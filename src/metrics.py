def top1_acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item()
