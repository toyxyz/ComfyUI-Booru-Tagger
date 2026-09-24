import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
SPEC = importlib.util.spec_from_file_location(
    "booru_tagger_model_info_test", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package
SPEC.loader.exec_module(package)
from booru_tagger_model_info_test import model_info


class ModelInfoTests(unittest.TestCase):
    def test_all_models_have_display_details(self):
        for model in model_info.config["model_url"]:
            with self.subTest(model=model):
                info = model_info.get_model_info(model)
                self.assertGreater(info["tag_count"], 0)
                self.assertGreater(info["input_size"], 0)
                self.assertIn(info["format"], ("ONNX", "PyTorch"))
                self.assertEqual(info["threshold"], model_info.config["threshold"][model])
                self.assertEqual(info["character_threshold"],
                                 model_info.config["character_threshold"][model])
                self.assertEqual(info["category_thresholds"],
                                 model_info.config["category_thresholds"][model])
                self.assertTrue(info["repository"].startswith("https://"))

    def test_local_metadata_overrides_catalog_and_reports_groups(self):
        model = "wd-vit-tagger-v3"
        with tempfile.TemporaryDirectory() as temp, patch.object(model_info, "models_dir", temp):
            model_path = pathlib.Path(temp, model_info.config["model_path"][model])
            metadata_path = pathlib.Path(temp, model_info.config["metadata_path"][model])
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"test")
            metadata_path.write_text(
                "tag_id,name,category,count\n0,foo,0,1\n1,bar,4,1\n2,safe,9,1\n",
                encoding="utf-8",
            )
            info = model_info.get_model_info(model)
            self.assertEqual(info["tag_count"], 3)
            self.assertEqual(info["category_counts"],
                             {"General": 1, "Character": 1, "Rating": 1})
            self.assertTrue(info["installed"])

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(KeyError):
            model_info.get_model_info("unknown")


if __name__ == "__main__":
    unittest.main()
