import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))  # ComfyUI root
SPEC = importlib.util.spec_from_file_location(
    "booru_tagger_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package
SPEC.loader.exec_module(package)
nodes = package.nodes


class ExcludeTagTests(unittest.TestCase):
    def setUp(self):
        self.names = ["photo (medium)", "star (symbol)", "solo", "alice (game)"]
        self.df = pd.DataFrame({"name": self.names, "category": [0, 0, 0, 4]})
        self.probs = np.array([0.9, 0.9, 0.9, 0.9])
        self.spec = SimpleNamespace(
            tags={"general": np.array([0, 1, 2]), "character": np.array([3]),
                  "rating": np.array([], dtype=int)},
            raw_names=np.asarray(self.names, dtype=object),
            escaped_names=np.asarray([nodes._escape(n) for n in self.names], dtype=object),
            best_threshold=None,
        )

    def test_plain_and_escaped_exclusions_in_both_paths(self):
        for spec in (None, self.spec):
            with self.subTest(spec=spec is not None):
                result = nodes.get_tag(
                    self.probs, self.df, spec, threshold=0.5,
                    character_threshold=0.5,
                    exclude_tags=r"photo \(medium\), STAR_(SYMBOL), alice (game)",
                )
                self.assertEqual(result["combined"], "solo")
                self.assertEqual(result["character"], "")

    def test_remaining_parentheses_are_escaped_once(self):
        for spec in (None, self.spec):
            with self.subTest(spec=spec is not None):
                result = nodes.get_tag(
                    self.probs, self.df, spec, threshold=0.5,
                    character_threshold=0.5, exclude_tags="solo",
                )
                self.assertEqual(
                    result["combined"],
                    r"alice \(game\), photo \(medium\), star \(symbol\)",
                )

    def test_old_double_escaped_output_can_be_excluded(self):
        for spec in (None, self.spec):
            with self.subTest(spec=spec is not None):
                result = nodes.get_tag(
                    self.probs, self.df, spec, threshold=0.5,
                    character_threshold=0.5,
                    exclude_tags=r"photo \\(medium\\)",
                )
                self.assertNotIn("photo", result["combined"])


if __name__ == "__main__":
    unittest.main()
