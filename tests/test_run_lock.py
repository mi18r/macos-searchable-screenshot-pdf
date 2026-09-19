from pathlib import Path
import shutil
import unittest

from run_lock import BuildAlreadyRunningError, build_lock, lock_name, source_identity


class BuildLockTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / ".tmp" / self._testMethodName
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.cache_directory = self.root / "cache"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_lock_name_is_deterministic_for_resolved_source(self):
        source = self.root / "Screenshots"
        source.mkdir()

        self.assertEqual(lock_name(source), lock_name(source.resolve()))

    def test_aliases_for_same_directory_share_filesystem_identity_and_lock(self):
        source = self.root / "Screenshots"
        alias = self.root / "Screenshots Alias"
        source.mkdir()
        alias.symlink_to(source, target_is_directory=True)

        self.assertEqual(source_identity(source), source_identity(alias))
        self.assertEqual(lock_name(source), lock_name(alias))

    def test_lock_name_isolated_between_sources(self):
        source_a = self.root / "Screenshots A"
        source_b = self.root / "Screenshots B"
        source_a.mkdir()
        source_b.mkdir()

        self.assertNotEqual(lock_name(source_a), lock_name(source_b))

    def test_same_source_rejects_nonblocking_contention_then_releases(self):
        source = self.root / "Screenshots"
        source.mkdir()

        with build_lock(self.cache_directory, source):
            with self.assertRaisesRegex(BuildAlreadyRunningError, str(source.resolve())):
                with build_lock(self.cache_directory, source):
                    pass

        with build_lock(self.cache_directory, source):
            pass

    def test_different_sources_can_be_locked_together(self):
        source_a = self.root / "Screenshots A"
        source_b = self.root / "Screenshots B"
        source_a.mkdir()
        source_b.mkdir()

        with build_lock(self.cache_directory, source_a):
            with build_lock(self.cache_directory, source_b):
                pass

    def test_exceptional_exit_releases_lock(self):
        source = self.root / "Screenshots"
        source.mkdir()

        with self.assertRaisesRegex(RuntimeError, "build failed"):
            with build_lock(self.cache_directory, source):
                raise RuntimeError("build failed")

        with build_lock(self.cache_directory, source):
            pass


if __name__ == "__main__":
    unittest.main()
