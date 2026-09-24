import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "booru_tagger_threshold_test", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules[PACKAGE_SPEC.name] = package
PACKAGE_SPEC.loader.exec_module(package)
nodes = package.nodes


def make_spec(df, model_name):
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="image")],
        get_outputs=lambda: [],
    )
    return nodes.ModelSpec(session, df, model_name, None, lambda image: image, "plain")


class CategoryThresholdTests(unittest.TestCase):
    def run_both_paths(self, model_name, df, probs, **kwargs):
        plain = nodes.get_tag(probs, df, model_name=model_name, **kwargs)
        fast = nodes.get_tag(probs, df, make_spec(df, model_name), **kwargs)
        self.assertEqual(plain, fast)
        return plain

    def test_pixai_six_distinct_recommendations_and_rating(self):
        df = pd.DataFrame({
            "name": ["general_low", "general_ok", "style_ok", "style_low",
                     "copyright_ok", "character_low", "meta_ok", "rating_safe"],
            "category_name": ["general", "general", "style", "style",
                              "copyright", "character", "meta", "rating"],
            "category": [0, 0, 0, 0, 4, 4, 0, 1],
        })
        probs = np.array([.16, .18, .16, .14, .25, .26, .18, .40], dtype=np.float32)
        result = self.run_both_paths(
            "pixai-tagger-v1.0", df, probs, threshold=.15,
            character_threshold=.24, use_best_threshold=True)
        self.assertEqual(result["general"], "general_ok, style_ok, meta_ok")
        self.assertEqual(result["character"], "copyright_ok")
        self.assertEqual(result["rating"], "")

        probs[-1] = .42
        result = self.run_both_paths(
            "pixai-tagger-v1.0", df, probs, threshold=.15,
            character_threshold=.24, use_best_threshold=True)
        self.assertEqual(result["rating"], "rating_safe")

    def test_disabling_recommendations_uses_manual_inputs(self):
        df = pd.DataFrame({"name": ["general", "style", "rating"],
                           "category_name": ["general", "style", "rating"],
                           "category": [0, 0, 1]})
        result = self.run_both_paths(
            "pixai-tagger-v1.0", df, np.array([.16, .16, .4]),
            threshold=.15, character_threshold=.24, use_best_threshold=False)
        self.assertEqual(result["general"], "general, style")
        self.assertEqual(result["rating"], "rating")

    def test_zero_inputs_use_model_recommendations(self):
        df = pd.DataFrame({"name": ["below", "above", "character_below"],
                           "category_name": ["general", "general", "character"],
                           "category": [0, 0, 4]})
        result = self.run_both_paths(
            "pixai-tagger-v1.0", df, np.array([.16, .18, .26]),
            use_best_threshold=True)
        self.assertEqual(result["general"], "above")
        self.assertEqual(result["character"], "")

    def test_manual_inputs_remain_minimums_and_missing_best_falls_back(self):
        df = pd.DataFrame({"name": ["general_low", "general_ok", "meta"],
                           "category_name": ["general", "general", "meta"],
                           "category": [0, 0, 3],
                           "best_threshold": [np.nan, .7, np.nan]})
        result = self.run_both_paths(
            "cl-tagger-v2-v2_00", df, np.array([.56, .65, .61]),
            threshold=.6, character_threshold=.55, use_best_threshold=True)
        self.assertEqual(result["general"], "meta")

    def test_camie_categories_have_their_output_group(self):
        df = pd.DataFrame({
            "name": ["general", "meta", "year", "character", "copyright", "artist"],
            "category_name": ["general", "meta", "year", "character", "copyright", "artist"],
            "category": [0, 3, 3, 4, 4, 4],
        })
        result = self.run_both_paths(
            "camie-tagger-v2", df, np.full(len(df), .6),
            threshold=.492, character_threshold=.492, use_best_threshold=True)
        self.assertEqual(result["general"], "general, meta, year")
        self.assertEqual(result["character"], "character, copyright, artist")

    def test_cl_v2_quality_uses_its_published_cutoff(self):
        df = pd.DataFrame({"name": ["quality_low", "quality_high"],
                           "category_name": ["quality", "quality"],
                           "category": [2, 2]})
        result = self.run_both_paths(
            "cl-tagger-v2-v2_00", df, np.array([.54, .56]),
            use_best_threshold=True)
        self.assertEqual(result["general"], "quality_high")

    def test_quality_without_a_published_cutoff_stays_excluded(self):
        df = pd.DataFrame({"name": ["quality", "meta"],
                           "category_name": ["quality", "meta"],
                           "category": [2, 3]})
        result = self.run_both_paths(
            "cl-tagger-v1-v1_00", df, np.array([.99, .56]),
            use_best_threshold=True)
        self.assertEqual(result["general"], "meta")


if __name__ == "__main__":
    unittest.main()
