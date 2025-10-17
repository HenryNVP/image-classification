from __future__ import annotations

from typing import Any

import torch


def get_device(prefer: str | None = None) -> str:
    if prefer:
        prefer = prefer.lower()
        if prefer == "cpu":
            return "cpu"
        if prefer in {"cuda", "gpu"} and torch.cuda.is_available():
            return "cuda"
        if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"

    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"[warn] Unable to read CUDA capability ({exc}); defaulting to CPU.")
        else:
            if (major, minor) >= (7, 0):
                return "cuda"
            print(
                f"[warn] CUDA capability {major}.{minor} is below 7.0; "
                "falling back to CPU."
            )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"

    return "cpu"


def to_device(obj: Any, device: str | torch.device | None = None):
    target = torch.device(device or get_device())

    if isinstance(obj, (torch.nn.Module, torch.Tensor)):
        return obj.to(target)
    return obj
