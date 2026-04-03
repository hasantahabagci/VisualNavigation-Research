from __future__ import annotations

import tempfile
import unittest

import torch

from jackal_diffusion.dataset import JackalLowdimDataset
from jackal_diffusion.policy import JackalDiffusionLowdimPolicy, JackalMLPLowdimPolicy


class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = JackalLowdimDataset(
            n_demos_per_side=4,
            n_obs_steps=2,
            n_action_steps=8,
            stride=2,
            val_ratio=0.2,
            seed=123,
        )
        cls.normalizer = cls.dataset.get_normalizer()
        batch = [cls.dataset[0], cls.dataset[1]]
        cls.batch = {
            "obs": torch.stack([item["obs"] for item in batch]),
            "action": torch.stack([item["action"] for item in batch]),
        }

    def _check_policy(self, policy):
        policy.set_normalizer(self.normalizer)
        loss = policy.compute_loss(self.batch)
        self.assertGreaterEqual(float(loss.item()), 0.0)
        result = policy.predict_action({"obs": self.batch["obs"]})
        self.assertEqual(tuple(result["action"].shape), (2, 8, 2))
        self.assertEqual(tuple(result["action_pred"].shape), (2, 8, 2))

    def test_diffusion_policy_shapes_and_legacy_roundtrip(self):
        policy = JackalDiffusionLowdimPolicy(n_obs_steps=2, n_action_steps=8, obs_dim=16)
        self._check_policy(policy)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/diffusion.pt"
            policy.save_legacy(path)
            loaded = JackalDiffusionLowdimPolicy.load_legacy(path)
            torch.manual_seed(123)
            loaded_result = loaded.predict_action({"obs": self.batch["obs"]})
            torch.manual_seed(123)
            original_result = policy.predict_action({"obs": self.batch["obs"]})
            self.assertTrue(
                torch.allclose(
                    loaded_result["action"],
                    original_result["action"],
                    atol=1e-5,
                )
            )

    def test_mlp_policy_shapes_and_legacy_roundtrip(self):
        policy = JackalMLPLowdimPolicy(n_obs_steps=2, n_action_steps=8, obs_dim=16)
        self._check_policy(policy)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/mlp.pt"
            policy.save_legacy(path)
            loaded = JackalMLPLowdimPolicy.load_legacy(path)
            loaded_result = loaded.predict_action({"obs": self.batch["obs"]})
            original_result = policy.predict_action({"obs": self.batch["obs"]})
            self.assertTrue(
                torch.allclose(
                    loaded_result["action"],
                    original_result["action"],
                    atol=1e-5,
                )
            )


if __name__ == "__main__":
    unittest.main()
