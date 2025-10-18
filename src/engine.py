from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    seen = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        seen += inputs.size(0)
    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, total_correct / seen


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
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> Tuple[float, List[dict]]:
    model.to(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    amp_enabled = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best = best_accuracy
    history: List[dict] = []

    patience_counter = 0

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

        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)
        val_loss, val_top1 = evaluate(model, val_ld, device)
        if val_top1 > best + early_stopping_min_delta:
            best = val_top1
            torch.save(model.state_dict(), save_path / "best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        current = epoch + 1
        print(
            f"Epoch {current}/{epochs} "
            f"- train loss {train_loss:.4f} acc {train_acc * 100:.2f}% "
            f"- val loss {val_loss:.4f} acc {val_top1 * 100:.2f}% "
            f"(best {best * 100:.2f}%)"
        )

        history.append(
            {
                "epoch": current,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_top1,
                "best_acc": best,
            }
        )

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(
                f"Early stopping triggered after {current} epochs: "
                f"no improvement for {patience_counter} epochs."
            )
            break

    return best, history
