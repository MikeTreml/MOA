from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from loguru import logger

from .schemas import FileChange, Profile, RunRecord


# IDs are minted by schemas.new_id() as "<prefix>_<32 hex>". Anything else in a
# run/change id reaching storage came from a URL path and must never be
# interpolated into a filesystem path (e.g. "..%5C..%5Cplanted").
_ID_RE = re.compile(r"^(run|change|step)_[0-9a-f]{32}$")


def _validate_id(value: str) -> str:
    if not _ID_RE.match(value or ""):
        raise KeyError(f"Invalid id: {value!r}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    A crash or power loss mid-write must never leave a half-written JSON file
    that permanently 500s every endpoint that parses it. We write to a sibling
    temp file, flush+fsync, then os.replace (atomic on Windows and POSIX).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _quarantine(path: Path) -> None:
    """Move a corrupt JSON file aside so it stops breaking every read.

    Renaming (not deleting) preserves the bytes for manual recovery while
    letting the app rebuild a fresh, valid file in its place."""
    try:
        backup = path.with_name(f"{path.name}.corrupt.{os.getpid()}")
        os.replace(path, backup)
        logger.warning(f"Quarantined corrupt file {path} -> {backup}")
    except OSError as exc:
        logger.error(f"Could not quarantine corrupt file {path}: {exc}")


class WorkbenchStorage:
    def __init__(self, root: Path | None = None):
        self.root = root or self.default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.changes_dir.mkdir(parents=True, exist_ok=True)
        # Sync endpoints run in FastAPI's threadpool, so read-modify-write of the
        # shared JSON files is genuinely concurrent. One re-entrant lock per
        # storage instance serializes writers and prevents lost updates.
        self._lock = threading.RLock()

    @staticmethod
    def default_root() -> Path:
        base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home() / "AppData" / "Local")
        )
        return Path(base) / "MoAWorkbench"

    @property
    def profiles_path(self) -> Path:
        return self.root / "profiles.json"

    @property
    def active_profile_path(self) -> Path:
        return self.root / "active_profile.txt"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def changes_dir(self) -> Path:
        return self.root / "changes"

    def list_profiles(self) -> list[Profile]:
        if not self.profiles_path.exists():
            default = Profile()
            self.save_profile(default)
            self.set_active_profile(default.name)
        try:
            data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
            return [Profile.model_validate(item) for item in data]
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            # A truncated/corrupt profiles.json would otherwise 500 every
            # profile-dependent endpoint forever. Quarantine it and rebuild the
            # default so the app stays usable.
            logger.error(f"profiles.json unreadable ({exc}); rebuilding default.")
            _quarantine(self.profiles_path)
            default = Profile()
            self.save_profile(default)
            self.set_active_profile(default.name)
            return [default]

    def get_profile(self, name: str) -> Profile:
        for profile in self.list_profiles():
            if profile.name == name:
                return profile
        raise KeyError(f"Profile not found: {name}")

    def get_active_profile(self) -> Profile:
        if self.active_profile_path.exists():
            name = self.active_profile_path.read_text(encoding="utf-8").strip()
            try:
                return self.get_profile(name)
            except KeyError:
                pass
        profiles = self.list_profiles()
        return profiles[0]

    def set_active_profile(self, name: str) -> None:
        self.get_profile(name)
        _atomic_write_text(self.active_profile_path, name)

    def save_profile(self, profile: Profile) -> Profile:
        with self._lock:
            profiles = [item for item in self.list_profiles() if item.name != profile.name] if self.profiles_path.exists() else []
            profiles.append(profile)
            profiles.sort(key=lambda item: item.name.lower())
            _atomic_write_text(
                self.profiles_path,
                json.dumps([item.model_dump(mode="json") for item in profiles], indent=2),
            )
            if not self.active_profile_path.exists():
                _atomic_write_text(self.active_profile_path, profile.name)
        return profile

    def delete_profile(self, name: str) -> None:
        """Remove a saved profile. Refuses to delete the last remaining profile
        (the app always needs one) and re-points the active pointer if needed."""
        with self._lock:
            profiles = self.list_profiles()
            if not any(item.name == name for item in profiles):
                raise KeyError(f"Profile not found: {name}")
            remaining = [item for item in profiles if item.name != name]
            if not remaining:
                raise ValueError("Cannot delete the only remaining profile.")
            _atomic_write_text(
                self.profiles_path,
                json.dumps([item.model_dump(mode="json") for item in remaining], indent=2),
            )
            active = None
            if self.active_profile_path.exists():
                active = self.active_profile_path.read_text(encoding="utf-8").strip()
            if active == name:
                _atomic_write_text(self.active_profile_path, remaining[0].name)

    def rename_profile(self, old_name: str, new_name: str) -> Profile:
        """Rename a saved profile in place, moving the active pointer with it.
        Refuses an empty name or a collision with another profile."""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("New profile name cannot be empty.")
        with self._lock:
            profiles = self.list_profiles()
            if not any(item.name == old_name for item in profiles):
                raise KeyError(f"Profile not found: {old_name}")
            # Case-insensitive collision check: profiles are sorted/compared by
            # lowercased name, so "Foo" and "foo" must not coexist.
            if new_name.lower() != old_name.lower() and any(
                item.name.lower() == new_name.lower() for item in profiles
            ):
                raise ValueError(f"A profile named {new_name!r} already exists.")
            renamed = None
            updated = []
            for item in profiles:
                if item.name == old_name:
                    renamed = item.model_copy(update={"name": new_name})
                    updated.append(renamed)
                else:
                    updated.append(item)
            updated.sort(key=lambda item: item.name.lower())
            _atomic_write_text(
                self.profiles_path,
                json.dumps([item.model_dump(mode="json") for item in updated], indent=2),
            )
            active = None
            if self.active_profile_path.exists():
                active = self.active_profile_path.read_text(encoding="utf-8").strip()
            if active == old_name:
                _atomic_write_text(self.active_profile_path, new_name)
            return renamed  # type: ignore[return-value]

    def save_run(self, record: RunRecord) -> RunRecord:
        with self._lock:
            path = self.runs_dir / f"{record.id}.json"
            _atomic_write_text(path, record.model_dump_json(indent=2))
            for change in record.file_changes:
                self.save_file_change(change)
        return record

    def authoritative_roots(self) -> list[str]:
        """The union of allowed_roots across every saved profile.

        This is the server-trusted set of directories the API is willing to read
        from or write to. Client-supplied roots are constrained against it so a
        caller can never widen scope past what the user has saved locally."""
        roots: list[str] = []
        for profile in self.list_profiles():
            for root in profile.allowed_roots:
                if root not in roots:
                    roots.append(root)
        return roots

    def get_run(self, run_id: str) -> RunRecord:
        _validate_id(run_id)
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            raise KeyError(f"Run not found: {run_id}")
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        records: list[RunRecord] = []
        for path in self.runs_dir.glob("*.json"):
            try:
                records.append(RunRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (ValueError, OSError) as exc:
                # One corrupt run file must not brick the entire history view.
                logger.warning(f"Skipping unreadable run file {path}: {exc}")
                continue
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records[:limit]

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its inlined change files. Idempotent-ish: a missing
        run raises KeyError so the API can surface a clean 404."""
        _validate_id(run_id)
        with self._lock:
            run_path = self.runs_dir / f"{run_id}.json"
            if not run_path.exists():
                raise KeyError(f"Run not found: {run_id}")
            try:
                record = self.get_run(run_id)
                for change in record.file_changes:
                    (self.changes_dir / f"{change.id}.json").unlink(missing_ok=True)
            except (ValueError, OSError):
                pass
            run_path.unlink(missing_ok=True)

    def save_file_change(self, change: FileChange) -> FileChange:
        with self._lock:
            path = self.changes_dir / f"{change.id}.json"
            _atomic_write_text(path, change.model_dump_json(indent=2))
            self._sync_change_into_run(change)
        return change

    def _sync_change_into_run(self, change: FileChange) -> None:
        """Keep the copy of a change embedded in its run record in sync.

        Runs are stored with their file changes inlined, and the GUI reads the
        run history rather than individual change files. Without this, applying
        or rejecting a change would never be reflected in the run view.
        """
        try:
            record = self.get_run(change.run_id)
        except KeyError:
            return
        replaced = False
        for index, existing in enumerate(record.file_changes):
            if existing.id == change.id:
                record.file_changes[index] = change
                replaced = True
        if not replaced:
            return
        path = self.runs_dir / f"{record.id}.json"
        _atomic_write_text(path, record.model_dump_json(indent=2))

    def get_file_change(self, change_id: str) -> FileChange:
        _validate_id(change_id)
        path = self.changes_dir / f"{change_id}.json"
        if not path.exists():
            raise KeyError(f"File change not found: {change_id}")
        return FileChange.model_validate_json(path.read_text(encoding="utf-8"))

