from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

from jackal_diffusion.config_utils import load_config
from jackal_diffusion.workspace import load_policy_from_checkpoint


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


class WorkspaceAndWrapperTest(unittest.TestCase):
    def _run_workspace(self, config_name: str, output_dir: str):
        cfg = load_config(config_name)
        cfg.training.device = "cpu"
        cfg.training.resume = False
        cfg.training.num_epochs = 1
        cfg.training.rollout_every = 1
        cfg.training.checkpoint_every = 1
        cfg.training.val_every = 1
        cfg.training.max_train_steps = 2
        cfg.training.max_val_steps = 2
        cfg.task.dataset.n_demos_per_side = 2
        cfg.task.env_runner.n_rollouts = 3

        module_name = cfg._target_.rsplit(".", 1)[0]
        class_name = cfg._target_.rsplit(".", 1)[1]
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        workspace = cls(cfg, output_dir=output_dir)
        workspace.run()
        return workspace

    def test_workspace_checkpoint_and_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = self._run_workspace("train_diffusion_lowdim_workspace", tmpdir)
            ckpt = workspace.get_checkpoint_path()
            self.assertTrue(os.path.exists(ckpt))
            _, policy = load_policy_from_checkpoint(str(ckpt), output_dir=tmpdir, device="cpu")
            self.assertIsNotNone(policy)

            eval_dir = os.path.join(tmpdir, "eval")
            cmd = [
                sys.executable,
                os.path.join(REPO_ROOT, "eval.py"),
                "--checkpoint",
                str(ckpt),
                "--output-dir",
                eval_dir,
            ]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)
            with open(os.path.join(eval_dir, "eval_log.json"), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("test_mean_score", payload)

    def test_train_and_live_wrappers(self):
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        with tempfile.TemporaryDirectory() as tmpdir:
            train_cmd = [
                sys.executable,
                os.path.join(REPO_ROOT, "train.py"),
                "--config-name=train_mlp_lowdim_workspace",
                "training.num_epochs=1",
                "training.max_train_steps=2",
                "training.max_val_steps=2",
                "training.rollout_every=1",
                "training.checkpoint_every=1",
                "training.val_every=1",
                "task.dataset.n_demos_per_side=2",
                "task.env_runner.n_rollouts=2",
                f"hydra.run.dir={tmpdir}/hydra_train",
            ]
            subprocess.run(train_cmd, check=True, cwd=REPO_ROOT, env=env)
            ckpt = os.path.join(tmpdir, "hydra_train", "checkpoints", "latest.ckpt")
            self.assertTrue(os.path.exists(ckpt))

            live_cmd = [
                sys.executable,
                os.path.join(REPO_ROOT, "live_sim.py"),
                "--checkpoint",
                ckpt,
                "--no-animate",
                "--max-steps",
                "20",
            ]
            subprocess.run(live_cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    unittest.main()
