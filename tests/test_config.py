"""Tests for config loading."""

import os
import tempfile
import unittest

from opentreesearch.tree_controller import DEFAULT_PUCT_C, load_config


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config function."""

    def test_default_puct(self):
        """Verify default puct_exploration_constant."""
        _, puct_c = load_config(None)
        self.assertEqual(puct_c, DEFAULT_PUCT_C)

    def test_extracts_puct_from_yaml(self):
        """Verify puct_exploration_constant is extracted from YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("database:\n  puct_exploration_constant: 2.5\n")
            f.flush()
            try:
                _, puct_c = load_config(f.name)
                self.assertEqual(puct_c, 2.5)
            finally:
                os.unlink(f.name)

    def test_default_when_absent(self):
        """Verify default is used when puct_exploration_constant is absent."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("database:\n  population_size: 100\n")
            f.flush()
            try:
                _, puct_c = load_config(f.name)
                self.assertEqual(puct_c, DEFAULT_PUCT_C)
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
