from __future__ import annotations

import unittest

import numpy as np
import torch

from jackal_diffusion.dataset import JackalLowdimDataset


class JackalDatasetTest(unittest.TestCase):
    def test_split_and_shapes_are_deterministic(self):
        dataset_a = JackalLowdimDataset(
            n_demos_per_side=4,
            n_obs_steps=2,
            n_action_steps=8,
            stride=2,
            val_ratio=0.2,
            seed=123,
        )
        dataset_b = JackalLowdimDataset(
            n_demos_per_side=4,
            n_obs_steps=2,
            n_action_steps=8,
            stride=2,
            val_ratio=0.2,
            seed=123,
        )

        self.assertTrue(np.array_equal(dataset_a.train_indices, dataset_b.train_indices))
        self.assertTrue(np.array_equal(dataset_a.val_indices, dataset_b.val_indices))
        sample = dataset_a[0]
        self.assertEqual(tuple(sample["obs"].shape), (2, dataset_a.obs_dim))
        self.assertEqual(tuple(sample["action"].shape), (8, dataset_a.action_dim))

    def test_normalizer_round_trip(self):
        dataset = JackalLowdimDataset(
            n_demos_per_side=4,
            n_obs_steps=2,
            n_action_steps=8,
            stride=2,
            val_ratio=0.2,
            seed=123,
        )
        normalizer = dataset.get_normalizer()
        sample = dataset[0]
        nobs = normalizer["obs"].normalize(sample["obs"])
        obs = normalizer["obs"].unnormalize(nobs)
        naction = normalizer["action"].normalize(sample["action"])
        action = normalizer["action"].unnormalize(naction)
        self.assertTrue(torch.allclose(obs, sample["obs"], atol=1e-5))
        self.assertTrue(torch.allclose(action, sample["action"], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
