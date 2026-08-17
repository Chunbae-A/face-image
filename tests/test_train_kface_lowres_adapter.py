from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "train_kface_lowres_adapter",
    SCRIPTS / "train_kface_lowres_adapter.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainKFaceLowResolutionAdapterTests(unittest.TestCase):
    def test_subject_split_is_disjoint_and_reproducible(self) -> None:
        subjects = [f"subject_{index:016x}" for index in range(20)]
        first = MODULE.split_subjects(subjects, seed=20260817)
        second = MODULE.split_subjects(subjects, seed=20260817)

        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(items) for name, items in first.items()},
            {"train": 12, "validation": 4, "test": 4},
        )
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is only required in the Kaggle training runtime",
    )
    def test_adapter_starts_as_identity_mapping(self) -> None:
        torch = MODULE._torch_module()
        model = MODULE.build_adapter(hidden_dimensions=16, residual_scale=0.25)
        values = torch.randn(4, 512)
        values = torch.nn.functional.normalize(values, dim=1)

        with torch.inference_mode():
            observed = model(values)

        self.assertTrue(torch.allclose(observed, values, atol=1e-6))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is only required in the Kaggle training runtime",
    )
    def test_small_locked_test_experiment_has_no_sensitive_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(20260817)
            for subject_index in range(12):
                subject = f"subject_{subject_index + 1:016x}"
                base = np.zeros(512, dtype=np.float32)
                base[subject_index] = 1.0
                medium = []
                low = []
                for _ in range(10):
                    medium_row = base + rng.normal(0, 0.01, size=512)
                    medium_row /= np.linalg.norm(medium_row)
                    low_row = medium_row + 0.02 * np.roll(medium_row, 1)
                    low_row /= np.linalg.norm(low_row)
                    medium.append(medium_row)
                    low.append(low_row)
                quality = np.tile(
                    np.asarray(
                        [[0.95, 0.10, 100.0, 128.0, 640.0, 640.0]],
                        dtype=np.float32,
                    ),
                    (10, 1),
                )
                np.savez_compressed(
                    root / f"{subject}__chunk_00000.npz",
                    image_indices=np.arange(10, dtype=np.int32),
                    low_embeddings=np.asarray(low, dtype=np.float32),
                    medium_embeddings=np.asarray(medium, dtype=np.float32),
                    low_quality=quality,
                    medium_quality=quality,
                )

            candidates = (
                MODULE.AdapterCandidate("paired", 1.0, 0.0),
                MODULE.AdapterCandidate("contrastive", 0.5, 0.5),
            )
            result = MODULE.run_experiment(
                root,
                candidates=candidates,
                split_seed=20260817,
                training_seed=20260817,
                reference_count=2,
                calibration_far=0.1,
                target_far=0.2,
                minimum_low_tar_improvement=1.0,
                hidden_dimensions=16,
                group_subjects=4,
                samples_per_subject=2,
                bins=1_000,
                device="cpu",
            )

            self.assertEqual(result["split"]["subject_overlap_count"], 0)
            self.assertFalse(
                result["split"]["test_used_for_training_or_candidate_selection"]
            )
            self.assertEqual(result["split"]["locked_test_evaluations"], 1)
            self.assertEqual(
                set(result["locked_test"]),
                {"baseline_raw_arcface", result["selection"]["selected_candidate"]},
            )
            self.assertFalse(result["contains_subject_identifiers"])
            self.assertFalse(result["contains_embeddings"])
            self.assertFalse(result["individual_scores_persisted"])
            self.assertFalse(result["model_weights_persisted"])
            self.assertIsNone(result["onnx_artifact"])


if __name__ == "__main__":
    unittest.main()
