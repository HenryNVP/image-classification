from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]

SUPPORTED_CAPABILITY = (7, 0)


def get_device(prefer: Optional[str] = None) -> str:
    """
    Choose compute device. Safely refuse GPUs with too-old compute capability.
    """
    prefer = (prefer or "").lower()

    if torch is None:
        return "cpu"

    if prefer == "cpu":
        return "cpu"

    if prefer in {"cuda", "gpu"}:
        if _cuda_ready(warn=True):
            return "cuda"
        return "cpu"

    if _cuda_ready(warn=False):
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"

    return "cpu"


def to_device(obj: Any, device: Optional[str] = None) -> Any:
    """
    Move tensors or modules to the requested device. Pass-through for other objects.
    """
    if torch is None:
        return obj

    actual_device = torch.device(device or get_device())

    if isinstance(obj, torch.nn.Module):
        return obj.to(actual_device)

    if isinstance(obj, torch.Tensor):
        return obj.to(actual_device)

    return obj


def _cuda_ready(warn: bool) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False

    try:
        capability: Tuple[int, int] = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - depends on runtime
        if warn:
            print(f"[warn] CUDA not usable: {exc}. Using CPU.")
        return False

    if capability >= SUPPORTED_CAPABILITY:
        return True

    if warn:
        major, minor = capability
        print(
            f"[warn] Detected CUDA compute capability {major}.{minor} "
            "which this PyTorch build does not support. Using CPU."
        )
    return False
