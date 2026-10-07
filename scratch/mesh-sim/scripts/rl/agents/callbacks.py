"""Training callbacks: bounded checkpoint retention and mask-aware evaluation."""

from pathlib import Path

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "checkpoint"


def list_checkpoints(directory, name_prefix: str = CHECKPOINT_PREFIX
                     ) -> list[tuple[int, Path]]:
    """Saved checkpoints as (num_timesteps, path), ascending; unparsable names ignored."""
    found = []
    for path in Path(directory).glob(f"{name_prefix}_*_steps.zip"):
        steps = path.name[len(name_prefix) + 1:-len("_steps.zip")]
        if steps.isdigit():
            found.append((int(steps), path))
    return sorted(found)


class BoundedCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that keeps only the newest keep_last checkpoint files."""

    def __init__(self, *args, keep_last: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if int(keep_last) < 1:
            raise ValueError(f"keep_last must be >= 1, got {keep_last!r}")
        self.keep_last = int(keep_last)

    def _on_step(self) -> bool:
        result = super()._on_step()
        retained = list_checkpoints(self.save_path, self.name_prefix)
        for _, path in retained[:-self.keep_last]:
            path.unlink(missing_ok=True)
        return result


def build_callbacks(out_dir: str, checkpoint_every: int, keep_last: int,
                    eval_env, eval_every: int, eval_episodes: int, verbose: int
                    ) -> tuple[list[BaseCallback], MaskableEvalCallback | None]:
    """Wire the cadence options; all units are SB3 timesteps (one env, one decision each)."""
    callbacks: list[BaseCallback] = []
    if checkpoint_every > 0:
        callbacks.append(BoundedCheckpointCallback(
            save_freq=checkpoint_every,
            save_path=str(Path(out_dir) / CHECKPOINT_DIR),
            name_prefix=CHECKPOINT_PREFIX,
            keep_last=keep_last,
            verbose=verbose,
        ))

    eval_callback = None
    if eval_every > 0 and eval_env is not None:
        eval_callback = MaskableEvalCallback(
            Monitor(eval_env),
            n_eval_episodes=eval_episodes,
            eval_freq=eval_every,
            best_model_save_path=str(out_dir),
            log_path=str(out_dir),
            deterministic=True,
            render=False,
            warn=False,
            use_masking=True,
            verbose=verbose,
        )
        callbacks.append(eval_callback)
    return callbacks, eval_callback
