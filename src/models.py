import timm, torch.nn as nn

def create_model(cfg):
    # timm.create_model wrapper
    m = timm.create_model(cfg.name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)

    in_features = None
    if getattr(m, "get_classifier", None):
        classifier = m.get_classifier()
        if classifier is not None and hasattr(classifier, "in_features"):
            in_features = classifier.in_features
    if in_features is None and hasattr(m, "num_features"):
        in_features = m.num_features

    if hasattr(m, "reset_classifier"):
        m.reset_classifier(num_classes=cfg.num_classes)
    elif hasattr(m, "head") and isinstance(m.head, nn.Linear):
        m.head = nn.Linear(m.head.in_features, cfg.num_classes)
    elif in_features:
        m.classifier = nn.Linear(in_features, cfg.num_classes)
    else:
        raise RuntimeError("Unable to determine classifier head dimensions.")

    if getattr(cfg, "freeze_backbone", False):
        for name, param in m.named_parameters():
            if any(head_name in name for head_name in ("head", "fc", "classifier")):
                continue
            param.requires_grad = False

    return m
