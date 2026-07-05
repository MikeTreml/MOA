import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SecurityBoundaryTests(unittest.TestCase):
    def make_client(self, appdata):
        from fastapi.testclient import TestClient
        from workbench.api import create_app

        with patch.dict(os.environ, {"LOCALAPPDATA": appdata}):
            return TestClient(create_app())

    def test_default_allowed_roots_is_repo_root_not_parent(self):
        from workbench.schemas import default_allowed_roots

        roots = default_allowed_roots()
        self.assertEqual(len(roots), 1)
        repo_root = Path(roots[0])
        # The repo root contains providers.py; its parent (a sibling-repo dir)
        # must NOT be the granted scope.
        self.assertTrue((repo_root / "providers.py").exists())
        self.assertNotEqual(repo_root, repo_root.parent)

    def test_files_read_cannot_widen_beyond_saved_roots(self):
        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as saved_root, tempfile.TemporaryDirectory() as secret_root:
            client = self.make_client(appdata)
            # Save a profile whose only allowed root is saved_root.
            from workbench.schemas import Profile

            client.post(
                "/api/profiles",
                json=Profile(name="Scoped", allowed_roots=[saved_root]).model_dump(mode="json"),
            )
            secret = Path(secret_root) / "secret.txt"
            secret.write_text("classified", encoding="utf-8")

            # Attacker asks to read the secret and claims a wide-open root.
            response = client.post(
                "/api/files/read",
                json={"paths": [str(secret)], "allowed_roots": [secret_root, "/"]},
            )
            # The read is refused because secret_root is outside every saved root.
            self.assertEqual(response.status_code, 400)

    def test_files_read_allows_narrowing_within_saved_roots(self):
        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as saved_root:
            client = self.make_client(appdata)
            from workbench.schemas import Profile

            client.post(
                "/api/profiles",
                json=Profile(name="Scoped", allowed_roots=[saved_root]).model_dump(mode="json"),
            )
            target = Path(saved_root) / "ok.txt"
            target.write_text("visible", encoding="utf-8")

            response = client.post(
                "/api/files/read",
                json={"paths": [str(target)], "allowed_roots": [saved_root]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["files"][0]["content"], "visible")

    def test_malformed_run_id_is_rejected_without_touching_disk(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            # A traversal payload in the id must 404 (invalid id), never resolve a path.
            response = client.get("/api/runs/..%2F..%2Fplanted")
            self.assertIn(response.status_code, (404, 400))

    def test_malformed_change_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.post("/api/file-changes/..%5C..%5Cplanted/apply")
            self.assertIn(response.status_code, (404, 400))

    def test_foreign_host_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.get("/api/provider-presets", headers={"host": "evil.example.com"})
            self.assertEqual(response.status_code, 400)

    def test_loopback_host_header_is_allowed(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            response = client.get("/api/provider-presets", headers={"host": "127.0.0.1:8008"})
            self.assertEqual(response.status_code, 200)

    def test_file_list_cannot_escape_saved_roots(self):
        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as saved_root, tempfile.TemporaryDirectory() as outside:
            client = self.make_client(appdata)
            from workbench.schemas import Profile

            client.post(
                "/api/profiles",
                json=Profile(name="Scoped", allowed_roots=[saved_root]).model_dump(mode="json"),
            )
            (Path(saved_root) / "sub").mkdir()
            (Path(saved_root) / "a.txt").write_text("x", encoding="utf-8")

            # No path -> lists the saved roots as the top level.
            top = client.post("/api/files/list", json={}).json()
            self.assertTrue(any(entry["is_dir"] for entry in top["entries"]))

            # Listing within the root works and shows children.
            inside = client.post("/api/files/list", json={"path": saved_root})
            self.assertEqual(inside.status_code, 200)
            names = {entry["name"] for entry in inside.json()["entries"]}
            self.assertIn("a.txt", names)
            self.assertIn("sub", names)

            # Listing a directory outside every saved root is refused.
            escaped = client.post("/api/files/list", json={"path": outside})
            self.assertEqual(escaped.status_code, 400)

    def test_rename_profile_endpoint(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)
            from workbench.schemas import Profile

            client.get("/api/profiles")  # seed default
            client.post("/api/profiles", json=Profile(name="Before").model_dump(mode="json"))
            renamed = client.post("/api/profiles/Before/rename", json={"new_name": "After"})
            self.assertEqual(renamed.status_code, 200)
            names = {p["name"] for p in client.get("/api/profiles").json()["profiles"]}
            self.assertIn("After", names)
            self.assertNotIn("Before", names)
            # Collision is a 409.
            client.post("/api/profiles", json=Profile(name="Other").model_dump(mode="json"))
            collision = client.post("/api/profiles/Other/rename", json={"new_name": "After"})
            self.assertEqual(collision.status_code, 409)

    def make_token_client(self, appdata, token):
        from fastapi.testclient import TestClient
        from workbench.api import create_app

        with patch.dict(os.environ, {"LOCALAPPDATA": appdata, "MOA_WORKBENCH_TOKEN": token}):
            return TestClient(create_app())

    def test_api_token_required_when_configured(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_token_client(appdata, "s3cret")

            # No token -> 401.
            self.assertEqual(client.get("/api/provider-presets").status_code, 401)
            # Wrong token -> 401.
            self.assertEqual(
                client.get("/api/provider-presets", headers={"X-MoA-Token": "nope"}).status_code, 401
            )
            # Correct token via header -> 200.
            self.assertEqual(
                client.get("/api/provider-presets", headers={"X-MoA-Token": "s3cret"}).status_code, 200
            )
            # Correct token via Authorization: Bearer -> 200.
            self.assertEqual(
                client.get("/api/provider-presets", headers={"Authorization": "Bearer s3cret"}).status_code, 200
            )
            # Correct token via query param (the EventSource path) -> 200.
            self.assertEqual(client.get("/api/provider-presets?token=s3cret").status_code, 200)

    def test_api_token_not_required_by_default(self):
        with tempfile.TemporaryDirectory() as appdata:
            client = self.make_client(appdata)  # no MOA_WORKBENCH_TOKEN
            self.assertEqual(client.get("/api/provider-presets").status_code, 200)

    def test_apply_already_applied_change_returns_409(self):
        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as workroot:
            client = self.make_client(appdata)
            from unittest.mock import AsyncMock

            from workbench.file_safety import create_file_change
            from workbench.schemas import Evaluation, RunRecord

            target = Path(workroot) / "out.txt"
            target.write_text("old", encoding="utf-8")
            record = RunRecord(
                profile_name="p", prompt="hi", workflow="hybrid", final_output="x",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            change = create_file_change(
                run_id=record.id, path=str(target),
                proposed_content="new", allowed_roots=[workroot],
            )
            record.file_changes = [change]
            with patch("workbench.api.run_workflow", new=AsyncMock(return_value=record)):
                client.post("/api/runs", json={"prompt": "hi", "profile": {"allowed_roots": [workroot]}})

            first = client.post(f"/api/file-changes/{change.id}/apply")
            self.assertEqual(first.status_code, 200)
            second = client.post(f"/api/file-changes/{change.id}/apply")
            self.assertEqual(second.status_code, 409)

    def test_reject_after_apply_returns_409(self):
        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as workroot:
            client = self.make_client(appdata)
            from unittest.mock import AsyncMock

            from workbench.file_safety import create_file_change
            from workbench.schemas import Evaluation, RunRecord

            target = Path(workroot) / "out.txt"
            target.write_text("old", encoding="utf-8")
            record = RunRecord(
                profile_name="p", prompt="hi", workflow="hybrid", final_output="x",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            change = create_file_change(
                run_id=record.id, path=str(target),
                proposed_content="new", allowed_roots=[workroot],
            )
            record.file_changes = [change]
            with patch("workbench.api.run_workflow", new=AsyncMock(return_value=record)):
                client.post("/api/runs", json={"prompt": "hi", "profile": {"allowed_roots": [workroot]}})

            client.post(f"/api/file-changes/{change.id}/apply")
            rejected = client.post(f"/api/file-changes/{change.id}/reject")
            self.assertEqual(rejected.status_code, 409)


if __name__ == "__main__":
    unittest.main()
