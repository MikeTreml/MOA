from __future__ import annotations

import difflib
from pathlib import Path

from .schemas import FileChange


class PathOutsideAllowedRoots(ValueError):
    pass


class FileChangedOnDisk(ValueError):
    """Raised when a file's on-disk content no longer matches what a proposed
    change was diffed against, so applying it would silently clobber edits."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed_path(path: str, allowed_roots: list[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    for root in allowed_roots:
        resolved_root = Path(root).expanduser().resolve()
        if candidate == resolved_root or _is_within(candidate, resolved_root):
            return candidate
    raise PathOutsideAllowedRoots(f"{candidate} is outside the allowed roots.")


def _resolve_roots(roots: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(Path(root).expanduser().resolve())
        except (OSError, ValueError, RuntimeError):
            continue
    return resolved


def constrain_roots(requested: list[str], authoritative: list[str]) -> list[str]:
    """Return the subset of ``requested`` roots that fall within ``authoritative``.

    This is the server-side guard for API callers: a client may *narrow* the
    server-trusted roots but never *widen* them. Passing an empty ``requested``
    list means "use every authoritative root". A requested root that is not
    contained by any authoritative root is dropped (fail closed), so a caller
    that supplies only out-of-scope roots gets an empty list and every
    subsequent path resolution raises ``PathOutsideAllowedRoots``.
    """
    authoritative_paths = _resolve_roots(authoritative)
    if not requested:
        return [str(path) for path in authoritative_paths]
    allowed: list[str] = []
    for candidate in _resolve_roots(requested):
        if any(
            candidate == root or _is_within(candidate, root)
            for root in authoritative_paths
        ):
            allowed.append(str(candidate))
    return allowed


def read_context_files(paths: list[str], allowed_roots: list[str], max_bytes: int = 20000):
    contexts = []
    for path in paths:
        resolved = resolve_allowed_path(path, allowed_roots)
        # Read at most max_bytes+1 so a multi-gigabyte file never balloons RSS;
        # the extra byte tells us whether the content was truncated.
        with open(resolved, "rb") as handle:
            raw = handle.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        contexts.append(
            {
                "path": str(resolved),
                "content": raw[:max_bytes].decode("utf-8", errors="replace"),
                "truncated": truncated,
            }
        )
    return contexts


def _read_text_verbatim(path: Path) -> str:
    """Read a file preserving its exact newlines (no universal-newline
    translation), so a proposed change never silently rewrites LF to CRLF.
    Raises UnicodeDecodeError for non-UTF-8 files rather than corrupting them."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def list_directory(path: str | None, allowed_roots: list[str]) -> dict:
    """List a directory's immediate children, constrained to allowed_roots.

    With ``path`` None, returns the roots themselves as the top level. A resolved
    path outside every root raises ``PathOutsideAllowedRoots``. ``parent`` is only
    returned when it still falls within a root, so a caller can never walk above
    the granted scope.
    """
    roots = _resolve_roots(allowed_roots)
    if not path:
        entries = [
            {"name": str(root), "path": str(root), "is_dir": True}
            for root in roots
            if root.exists()
        ]
        return {"path": None, "parent": None, "entries": entries}

    resolved = resolve_allowed_path(path, allowed_roots)
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved} does not exist.")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{resolved} is not a directory.")

    root_paths = _resolve_roots(allowed_roots)
    entries: list[dict] = []
    try:
        for child in sorted(resolved.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            try:
                is_dir = child.is_dir()
                # Skip a child that resolves (via symlink) outside the roots, so
                # the listing never even names a path the caller couldn't read.
                real = child.resolve()
                if not any(real == r or _is_within(real, r) for r in root_paths):
                    continue
            except OSError:
                continue
            entries.append({"name": child.name, "path": str(child), "is_dir": is_dir})
    except OSError as exc:
        raise OSError(f"Cannot read directory {resolved}: {exc}") from exc

    parent = resolved.parent
    parent_str = str(parent) if any(parent == r or _is_within(parent, r) for r in roots) else None
    return {"path": str(resolved), "parent": parent_str, "entries": entries}


def make_diff(path: str, original: str, proposed: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"{path} (current)",
            tofile=f"{path} (proposed)",
        )
    )


def create_file_change(
    run_id: str,
    path: str,
    proposed_content: str,
    allowed_roots: list[str],
) -> FileChange:
    resolved = resolve_allowed_path(path, allowed_roots)
    original = _read_text_verbatim(resolved) if resolved.exists() else ""
    return FileChange(
        run_id=run_id,
        path=str(resolved),
        original_content=original,
        proposed_content=proposed_content,
        diff=make_diff(str(resolved), original, proposed_content),
        allowed_roots=allowed_roots,
    )


def apply_file_change(change: FileChange) -> FileChange:
    resolved = resolve_allowed_path(change.path, change.allowed_roots)

    # Staleness (TOCTOU) guard: the proposed diff was computed against
    # original_content. If the file changed on disk since then — a manual edit,
    # or another applied change touching the same file — applying our whole-file
    # rewrite would silently destroy those edits. Refuse instead.
    current = _read_text_verbatim(resolved) if resolved.exists() else ""
    if current != change.original_content:
        raise FileChangedOnDisk(
            f"{resolved} changed on disk since this edit was proposed; "
            "re-run so the diff reflects the current file."
        )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    # newline="" writes proposed_content verbatim — no LF->CRLF translation.
    with open(resolved, "w", encoding="utf-8", newline="") as handle:
        handle.write(change.proposed_content)
    change.status = "applied"
    from .schemas import now_iso

    change.applied_at = now_iso()
    return change


def reject_file_change(change: FileChange) -> FileChange:
    change.status = "rejected"
    return change

