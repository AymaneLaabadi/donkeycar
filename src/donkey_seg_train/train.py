"""Training entry point. (To do.)

    python -m donkey_seg_train.data.prepare      # once; reruns are no-ops while the data is unchanged
    python -m donkey_seg_train.train             # later

Get the loaders with:
    from donkey_seg_train.config import load_config
    from donkey_seg_train.data import build_dataloaders
    loaders = build_dataloaders(load_config("train.yaml"))   # loaders["train"], ["val"], ["test"]
"""
