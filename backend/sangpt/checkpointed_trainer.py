"""
Checkpointed trainer for SANCTA-GPT built on normalized dataset manifests.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from dataset_pipeline import DatasetManifestPipeline


class CheckpointedTrainer:
    """Runs training with manifest-backed corpora and periodic checkpoints."""

    def __init__(self, engine, pipeline: DatasetManifestPipeline, logger=None, checkpoint_dir: str = "./logs/checkpoints"):
        self.engine = engine
        self.pipeline = pipeline
        self.logger = logger or logging.getLogger("CheckpointedTrainer")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        steps: int = 100,
        mode: str = "all",
        checkpoint_interval: int = 50,
        progress_interval: int = 10,
        progress_callback=None,
    ) -> Dict[str, Any]:
        manifest = self.pipeline.ensure_manifest()
        documents = self.pipeline.load_training_documents(mode)
        if not documents:
            raise ValueError(f"No training documents available for mode: {mode}")

        self.engine.set_training_mode(mode.upper())
        self.engine.set_training_data(documents)

        # Reset the cosine LR schedule for this training run so the model
        # trains at full learning rate instead of being stuck at the floor.
        if hasattr(self.engine, 'begin_training_run'):
            self.engine.begin_training_run(max_steps=steps)

        started_at = datetime.now()
        loss = None
        checkpoints = []
        progress = []

        interval = max(1, progress_interval)
        for step_idx in range(steps):
            loss = self.engine.train_step()
            logical_step = step_idx + 1
            progress.append(loss)

            if progress_callback and (logical_step == 1 or logical_step % interval == 0 or logical_step == steps):
                progress_callback(logical_step, steps, loss)

            if logical_step % max(1, checkpoint_interval) == 0 or logical_step == steps:
                checkpoint_path = self._checkpoint_path(mode, self.engine.step_count)
                metadata = {
                    "mode": mode,
                    "logical_step": logical_step,
                    "engine_step": self.engine.step_count,
                    "loss": loss,
                    "corpus_size": len(documents),
                    "manifest_path": manifest["manifest_path"],
                    "generated_at": datetime.now().isoformat(),
                }
                self.engine.save_checkpoint(str(checkpoint_path), metadata=metadata)
                checkpoints.append(str(checkpoint_path))

        # Downsample the loss history to keep summaries manageable for long runs
        if len(progress) > 200:
            stride = len(progress) // 200
            sampled_loss = progress[::stride]
            # Always include the final value
            if progress[-1] != sampled_loss[-1]:
                sampled_loss.append(progress[-1])
        else:
            sampled_loss = list(progress)

        summary = {
            "status": "completed",
            "mode": mode,
            "steps_requested": steps,
            "steps_completed": steps,
            "corpus_size": len(documents),
            "manifest_path": manifest["manifest_path"],
            "checkpoint_interval": checkpoint_interval,
            "checkpoints": checkpoints,
            "final_loss": loss,
            "loss_history": sampled_loss,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
        }

        summary_path = self.checkpoint_dir / f"train_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        summary["summary_path"] = str(summary_path)
        self.logger.info(f"Training summary written to {summary_path}")
        return summary

    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        payload = self.engine.load_checkpoint(checkpoint_path)
        self.logger.info(f"Resumed training state from {checkpoint_path}")
        return payload

    def latest_checkpoint(self) -> Optional[str]:
        # Look for JSON checkpoints (metadata) — .pt weights are loaded automatically
        checkpoints = sorted(self.checkpoint_dir.glob("*_step*.json"))
        return str(checkpoints[-1]) if checkpoints else None

    def latest_pt_checkpoint(self) -> Optional[str]:
        """Return the latest .pt (PyTorch weights) checkpoint if available."""
        checkpoints = sorted(self.checkpoint_dir.glob("*_step*.pt"))
        return str(checkpoints[-1]) if checkpoints else None

    def _checkpoint_path(self, mode: str, step_count: int) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = mode.lower().replace(" ", "_")
        return self.checkpoint_dir / f"{safe_mode}_step{step_count}_{timestamp}.json"
