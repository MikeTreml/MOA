import concurrent.futures
import tempfile
import unittest
from pathlib import Path


class StorageDurabilityTests(unittest.TestCase):
    def make_storage(self, root):
        from workbench.storage import WorkbenchStorage

        return WorkbenchStorage(root=Path(root))

    def test_corrupt_profiles_file_is_quarantined_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as root:
            storage = self.make_storage(root)
            storage.list_profiles()  # materialize the default profiles.json
            storage.profiles_path.write_text("{ this is not json", encoding="utf-8")

            profiles = storage.list_profiles()  # must not raise

            self.assertGreaterEqual(len(profiles), 1)
            # The corrupt bytes are preserved alongside, not silently deleted.
            corrupt = list(Path(root).glob("profiles.json.corrupt.*"))
            self.assertTrue(corrupt)

    def test_corrupt_run_file_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            from workbench.schemas import Evaluation, RunRecord

            storage = self.make_storage(root)
            good = RunRecord(
                profile_name="p", prompt="ok", workflow="hybrid", final_output="x",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            storage.save_run(good)
            (storage.runs_dir / "run_deadbeef.json").write_text("garbage", encoding="utf-8")

            runs = storage.list_runs()  # must not raise

            ids = {run.id for run in runs}
            self.assertIn(good.id, ids)

    def test_writes_are_atomic_no_partial_file(self):
        # A successful write leaves no leftover temp files and valid JSON.
        with tempfile.TemporaryDirectory() as root:
            from workbench.schemas import Profile

            storage = self.make_storage(root)
            storage.save_profile(Profile(name="Alpha"))
            leftovers = list(Path(root).glob("*.tmp"))
            self.assertEqual(leftovers, [])
            # File parses cleanly.
            names = {p.name for p in storage.list_profiles()}
            self.assertIn("Alpha", names)

    def test_concurrent_profile_saves_do_not_lose_updates(self):
        with tempfile.TemporaryDirectory() as root:
            from workbench.schemas import Profile

            storage = self.make_storage(root)
            storage.list_profiles()  # seed default

            def save(i):
                storage.save_profile(Profile(name=f"P{i}"))

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(save, range(20)))

            names = {p.name for p in storage.list_profiles()}
            for i in range(20):
                self.assertIn(f"P{i}", names)

    def test_delete_run_removes_record(self):
        with tempfile.TemporaryDirectory() as root:
            from workbench.schemas import Evaluation, RunRecord

            storage = self.make_storage(root)
            record = RunRecord(
                profile_name="p", prompt="x", workflow="hybrid", final_output="x",
                evaluation=Evaluation(status="PASS", feedback="ok", score=1),
            )
            storage.save_run(record)
            storage.delete_run(record.id)
            self.assertRaises(KeyError, storage.get_run, record.id)

    def test_rename_profile_moves_active_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            from workbench.schemas import Profile

            storage = self.make_storage(root)
            storage.save_profile(Profile(name="Original"))
            storage.set_active_profile("Original")
            storage.rename_profile("Original", "Renamed")

            names = {p.name for p in storage.list_profiles()}
            self.assertIn("Renamed", names)
            self.assertNotIn("Original", names)
            self.assertEqual(storage.get_active_profile().name, "Renamed")

    def test_rename_profile_rejects_collision(self):
        with tempfile.TemporaryDirectory() as root:
            from workbench.schemas import Profile

            storage = self.make_storage(root)
            storage.save_profile(Profile(name="A"))
            storage.save_profile(Profile(name="B"))
            self.assertRaises(ValueError, storage.rename_profile, "A", "B")
            self.assertRaises(KeyError, storage.rename_profile, "Nope", "C")
            # Case-only collision is rejected (store compares case-insensitively).
            self.assertRaises(ValueError, storage.rename_profile, "A", "b")

    def test_delete_last_profile_refused(self):
        with tempfile.TemporaryDirectory() as root:
            storage = self.make_storage(root)
            profiles = storage.list_profiles()
            # Reduce to a single profile, then attempt to delete it.
            for extra in profiles[1:]:
                storage.delete_profile(extra.name)
            only = storage.list_profiles()
            self.assertEqual(len(only), 1)
            self.assertRaises(ValueError, storage.delete_profile, only[0].name)


if __name__ == "__main__":
    unittest.main()
