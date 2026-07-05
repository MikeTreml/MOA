import tempfile
import unittest
from pathlib import Path


class FileChangeSafetyTests(unittest.TestCase):
    def test_apply_refuses_when_file_changed_since_proposal(self):
        from workbench.file_safety import (
            FileChangedOnDisk,
            apply_file_change,
            create_file_change,
        )

        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "f.txt"
            target.write_text("original", encoding="utf-8")
            change = create_file_change(
                run_id="run_" + "0" * 32,
                path=str(target),
                proposed_content="proposed",
                allowed_roots=[root],
            )
            # Someone edits the file after the proposal was made.
            target.write_text("manual edit", encoding="utf-8")

            self.assertRaises(FileChangedOnDisk, apply_file_change, change)
            # The manual edit survives; the stale proposal did not overwrite it.
            self.assertEqual(target.read_text(encoding="utf-8"), "manual edit")

    def test_apply_succeeds_when_file_unchanged(self):
        from workbench.file_safety import apply_file_change, create_file_change

        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "f.txt"
            target.write_text("original", encoding="utf-8")
            change = create_file_change(
                run_id="run_" + "0" * 32,
                path=str(target),
                proposed_content="proposed",
                allowed_roots=[root],
            )
            applied = apply_file_change(change)
            self.assertEqual(applied.status, "applied")
            self.assertEqual(target.read_text(encoding="utf-8"), "proposed")

    def test_apply_preserves_lf_line_endings(self):
        from workbench.file_safety import apply_file_change, create_file_change

        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "f.txt"
            # Write LF bytes verbatim.
            target.write_bytes(b"line1\nline2\n")
            change = create_file_change(
                run_id="run_" + "0" * 32,
                path=str(target),
                proposed_content="line1\nline2\nline3\n",
                allowed_roots=[root],
            )
            apply_file_change(change)
            # Bytes must stay LF — no silent CRLF rewrite on Windows.
            self.assertEqual(target.read_bytes(), b"line1\nline2\nline3\n")

    def test_create_change_on_crlf_file_roundtrips_without_staleness(self):
        # A CRLF file must apply cleanly: original_content captured verbatim so
        # the staleness check matches exactly.
        from workbench.file_safety import apply_file_change, create_file_change

        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "f.txt"
            target.write_bytes(b"a\r\nb\r\n")
            change = create_file_change(
                run_id="run_" + "0" * 32,
                path=str(target),
                proposed_content="a\r\nb\r\nc\r\n",
                allowed_roots=[root],
            )
            applied = apply_file_change(change)  # must not raise FileChangedOnDisk
            self.assertEqual(applied.status, "applied")
            self.assertEqual(target.read_bytes(), b"a\r\nb\r\nc\r\n")

    def test_context_read_truncates_large_file_without_reading_all(self):
        from workbench.file_safety import read_context_files

        with tempfile.TemporaryDirectory() as root:
            big = Path(root) / "big.txt"
            big.write_text("x" * 50000, encoding="utf-8")
            out = read_context_files([str(big)], [root], max_bytes=100)
            self.assertTrue(out[0]["truncated"])
            self.assertEqual(len(out[0]["content"]), 100)


if __name__ == "__main__":
    unittest.main()
