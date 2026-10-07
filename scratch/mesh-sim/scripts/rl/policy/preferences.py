"""Capture a MaskablePPO policy's per-slot action logits during its own predict call."""

import numpy as np


class PreferenceCapture:
    """Forward hook on ``policy.action_net``; no extra forward, RNG, or parameter access."""

    def __init__(self):
        self.registered = False
        self.error: str | None = None
        self._handle = None
        self._last: np.ndarray | None = None

    def attach(self, model) -> None:
        try:
            import torch

            action_net = model.policy.action_net
            if not isinstance(action_net, torch.nn.Module):
                raise TypeError(f"policy.action_net is {type(action_net).__name__}, "
                                "not a torch module")
            self._handle = action_net.register_forward_hook(self._hook)
        except Exception as exc:
            self.registered = False
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self.registered = True
        self.error = None

    def _hook(self, module, inputs, output) -> None:
        # Copy out: the tensor's storage may be reused or mutated after predict returns.
        self._last = output.detach().cpu().numpy().astype(np.float32, copy=True).reshape(-1)

    def take(self) -> np.ndarray | None:
        last, self._last = self._last, None
        return last

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.registered = False
