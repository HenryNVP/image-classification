from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .metrics import top1_acc

@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    metrics: list[float] = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        metrics.append(top1_acc(outputs, targets))
    if not metrics:
        return 0.0
    return float(sum(metrics) / len(metrics))


def fit(
    model: torch.nn.Module,
    train_ld,
    val_ld,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    device: torch.device,
    save_dir,
    amp: bool = True,
    start_epoch: int = 0,
    best_accuracy: float = 0.0,
    label_smoothing: float = 0.1,
    log_interval: int | None = None,
) -> float:
    model.to(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    amp_enabled = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best = best_accuracy

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        running_correct = 0
        seen = 0
        model.train()
        for step, (inputs, targets) in enumerate(train_ld, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits, targets, label_smoothing=label_smoothing
                )
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += logits.argmax(dim=1).eq(targets).sum().item()
            seen += inputs.size(0)

            if log_interval and step % log_interval == 0:
                avg_loss = running_loss / seen
                avg_acc = running_correct / seen
                print(
                    f"Epoch {epoch + 1}/{epochs} Step {step}/{len(train_ld)} "
                    f"- loss: {avg_loss:.4f} acc: {avg_acc * 100:.2f}%"
                )

        val_top1 = evaluate(model, val_ld, device)
        if val_top1 > best:
            best = val_top1
            torch.save(model.state_dict(), save_path / "best.pth")
        current = epoch + 1
        print(
            f"Epoch {current}/{epochs} - val@1={val_top1 * 100:.2f}%  best={best * 100:.2f}%"
        )

    return best
