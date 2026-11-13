"""Segmentation network. (To do.)

Inputs are float32 Bx3xHxW in [0, 1]. Put the normalisation (mean/std from cache/train/info.json) inside
the model as its first layer, so the car can feed raw [0, 1] frames with no extra preprocessing.
"""
