"""Training side of the project: road segmentation on the data written by donkey_seg_datagen.

    data/      step-wise data preparation (cache, texture bank, optional materialised copies),
               augmentations, and the PyTorch Dataset / DataLoader builders
    model.py   network definition
    losses.py  loss functions
    train.py   training loop / entry point
    config.py  train.yaml loading (defaults + validation)
"""
