import unittest
from unittest.mock import Mock, patch

from caat.registry import Registry


class RegistryTests(unittest.TestCase):
    def test_entries_load_on_use_and_cannot_replace_a_builtin(self):
        installed = Mock(name="provider")
        installed.name = "registered"
        installed.load.return_value = "from provider"
        collision = Mock()
        collision.name = "reserved"
        with patch(
            "caat.registry.entry_points", return_value=[installed, collision]
        ) as points:
            registry = Registry("fixtures", str, {"reserved": "built-in value"})
            self.assertEqual(tuple(registry), ("reserved", "registered"))
            points.assert_called_once_with(group="caat.fixtures")
            installed.load.assert_not_called()
            self.assertEqual(registry["reserved"], "built-in value")
            collision.load.assert_not_called()
            self.assertEqual(registry["registered"], "from provider")
            self.assertEqual(registry["registered"], "from provider")
            installed.load.assert_called_once_with()

    def test_duplicate_entries_and_wrong_component_types_are_rejected(self):
        first, second, bad = Mock(), Mock(), Mock()
        first.name = second.name = "duplicate"
        bad.name = "bad"
        bad.load.return_value = []
        with patch("caat.registry.entry_points", return_value=[first, second, bad]):
            registry = Registry("fixtures", str, {})
            with self.assertRaisesRegex(ValueError, "multiple entry points"):
                registry["duplicate"]
            first.load.assert_not_called()
            with self.assertRaisesRegex(ValueError, "must expose a str instance"):
                registry["bad"]
