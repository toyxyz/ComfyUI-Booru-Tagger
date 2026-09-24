import json
import pathlib
import unittest


CONFIG = json.loads((pathlib.Path(__file__).resolve().parents[1] / "models.json").read_text(encoding="utf-8"))


class ThresholdConfigTests(unittest.TestCase):
    def test_every_model_has_valid_thresholds_and_paths(self):
        models = set(CONFIG["model_url"])
        for key in ("model_path", "metadata_path", "remote_model_path",
                    "remote_metadata_path", "threshold", "character_threshold",
                    "tag_count", "input_size", "category_thresholds"):
            with self.subTest(key=key):
                self.assertEqual(set(CONFIG[key]), models)
        for key in ("threshold", "character_threshold"):
            for model, value in CONFIG[key].items():
                with self.subTest(key=key, model=model):
                    self.assertIsInstance(value, (int, float))
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)
        for key in ("tag_count", "input_size"):
            for model, value in CONFIG[key].items():
                with self.subTest(key=key, model=model):
                    self.assertIsInstance(value, int)
                    self.assertGreater(value, 0)
        for model, thresholds in CONFIG["category_thresholds"].items():
            with self.subTest(model=model):
                if model != "pixai-tagger-v1.0":
                    self.assertEqual(thresholds["general"], CONFIG["threshold"][model])
                    if "character" in thresholds:
                        self.assertEqual(thresholds["character"],
                                         CONFIG["character_threshold"][model])
                for value in thresholds.values():
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)

    def test_pixai_v1_has_all_six_published_category_thresholds(self):
        self.assertEqual(CONFIG["category_thresholds"]["pixai-tagger-v1.0"], {
            "general": 0.17, "character": 0.27, "style": 0.15,
            "copyright": 0.24, "meta": 0.17, "rating": 0.41,
        })

    def test_wd_defaults_are_explicit_for_each_model(self):
        for model in CONFIG["model_url"]:
            if model.startswith("wd-"):
                with self.subTest(model=model):
                    self.assertEqual(CONFIG["threshold"][model], 0.35)
                    self.assertEqual(CONFIG["character_threshold"][model], 0.85)


if __name__ == "__main__":
    unittest.main()
