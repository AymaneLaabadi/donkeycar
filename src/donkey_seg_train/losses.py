"""Loss functions. (To do.)

Masks are int64 BxHxW with 0 = background, 1 = road, 255 = ignore (pixels that the geometric
augmentation brought in from outside the frame). Every loss must skip them, for example
`F.cross_entropy(logits, mask, ignore_index=IGNORE_INDEX)`.
"""

from .data.augment import IGNORE_INDEX  # noqa: F401  (single source of truth for the ignore label)
