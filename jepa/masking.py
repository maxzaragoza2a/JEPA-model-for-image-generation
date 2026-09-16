"""Sampling the context and target masks -- the self-supervised task itself.

Nothing here touches an image. Everything works on the *patch grid*: for our
config a 32x32 image becomes an 8x8 grid of 64 patches, numbered row by row.

    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
   ...
   56 57 58 59 60 61 62 63

The task JEPA learns is: "here is part of the image -- predict what is in the
parts I hid". So we split those 64 numbers into two roles:

    context  the patches shown to the model
    targets  the patches it must predict

We hide whole *rectangles*, not scattered patches. Neighbouring patches look
alike, so an isolated hidden patch can be interpolated from the visible ones
around it -- the model would learn to blur, not to understand. A rectangle has
an interior with no visible neighbour at all, so it forces real reasoning.

One draw looks like this (T = target, C = context, . = used by neither):

    C  C  C  C  C  C  T  T
    C  C  T  T  T  C  T  T
    C  C  T  T  T  C  C  .
    C  C  T  T  T  C  C  .
    C  C  C  C  C  T  T  .
    T  T  T  T  C  T  T  .
    T  T  T  T  C  T  T  .
    .  .  .  .  .  .  .  .

Two things to notice. The context is a large rectangle with the targets *cut
out of it* -- otherwise we would hand the model the answer along with the
question. And some patches end up in neither role; that is fine, they are
simply unused this round.

An important asymmetry, easy to get wrong: only the CONTEXT encoder sees a
reduced sequence. The target encoder is later fed the whole unmasked image,
and the target representations are read out afterwards at these indices. That
is what makes the targets rich, and it is what separates JEPA from MAE.

A note on batching. The number of context patches changes from draw to draw
(24 above, 31 in the next one), and TensorFlow cannot stack rows of different
lengths. So we draw ONE mask per batch and share it across every image in it.
Simple, statically shaped, and close enough to the paper -- which also forces a
common length across a batch, varying only the positions.
"""

import numpy as np
import tensorflow as tf


def sample_block(grid_size: int, scale: tuple, aspect_ratio: tuple, rng) -> tuple:
    """Draw one random rectangle inside a grid_size x grid_size grid.

    Args:
        grid_size: side of the patch grid (8 for us).
        scale: (low, high) area of the rectangle as a fraction of the whole
            grid. (0.15, 0.20) on a 64-patch grid means roughly 10 to 13
            patches.
        aspect_ratio: (low, high) height / width. 1.0 is square, 0.75 is
            wider than tall, 1.5 is taller than wide.
        rng: a numpy Generator, so draws are reproducible from config.seed.

    Returns:
        (top, left, height, width) -- the rectangle's top-left corner and size,
        in patch units.

    How to get there:
      - draw an area and an aspect ratio uniformly in their ranges;
      - turn them into a height and a width (area = h * w, ratio = h / w),
        then round to whole patches;
      - clamp h and w to at least 1 and at most grid_size, or a large draw on
        a small grid can ask for a rectangle that does not fit;
      - draw top and left so the rectangle stays inside the grid.
    """
    area = rng.uniform(*scale) * grid_size**2
    ratio = rng.uniform(*aspect_ratio)
    height = int(round(np.sqrt(area * ratio)))
    width = int(round(np.sqrt(area / ratio)))
    height = np.clip(height, 1, grid_size)
    width = np.clip(width, 1, grid_size)
    top = rng.integers(0, grid_size - height + 1)
    left = rng.integers(0, grid_size - width + 1)
    return top, left, height, width




def block_to_indices(top: int, left: int, height: int, width: int,
                     grid_size: int) -> np.ndarray:
    """Turn a rectangle into the flat token indices it covers.

    The grid is numbered row by row, so the patch at row r, column c is token
    r * grid_size + c. This is the same row-major order extract_patches emits
    and the same order the positional table follows -- they must agree, and a
    mismatch here is completely silent.

    Returns:
        A 1-D int array of height * width token indices, sorted ascending.
    """
    rows = np.arange(top, top + height)
    cols = np.arange(left, left + width)

    # rows as a column, cols as a row: broadcasting fills the whole rectangle
    # in one go, laid out row by row. Flattening it therefore already yields
    # ascending indices -- no sort needed.
    return (rows[:, None] * grid_size + cols[None, :]).reshape(-1)


def sample_masks(config, rng) -> tuple:
    """Draw one full context / targets split, shared by a whole batch.

    Returns:
        context_indices: 1-D int array, the tokens shown to the context
            encoder.
        target_indices: a list of config.num_target_blocks 1-D int arrays, one
            per rectangle to predict. They are kept separate rather than
            merged: each block is predicted as its own little problem.

    How to get there:
      - draw num_target_blocks rectangles with target_scale and
        target_aspect_ratio, and convert each to indices;
      - draw one context rectangle with context_scale and a square ratio;
      - remove from the context every index appearing in any target block.

    Target blocks may overlap each other -- that is allowed, they stay separate
    predictions. What must never happen is a target index surviving inside the
    context.
    """
    grid_size = config.grid_size

    target_indices = [
        block_to_indices(
            *sample_block(grid_size, config.target_scale,
                          config.target_aspect_ratio, rng),
            grid_size,
        )
        for _ in range(config.num_target_blocks)
    ]

    # A square ratio for the context: it is meant to be one large view of the
    # image, not a stripe, so there is nothing to vary here.
    context_indices = block_to_indices(
        *sample_block(grid_size, config.context_scale, (1.0, 1.0), rng),
        grid_size,
    )

    # The cut-out. Blocks may overlap, so flatten them into one set first;
    # setdiff1d keeps the result sorted and unique.
    hidden = np.concatenate(target_indices)
    context_indices = np.setdiff1d(context_indices, hidden)

    return context_indices, target_indices


def gather_tokens(tokens, indices):
    """Read a token sequence at the given indices: (B, N, D) -> (B, M, D).

    Args:
        tokens: (batch, num_patches, embed_dim), the full sequence.
        indices: 1-D int array of M token indices from sample_masks. One draw
            is shared by the whole batch, so there is no batch dimension here
            and every image is read at the same positions.

    The same call is used at two opposite moments, which is the easy thing to
    get wrong:
      - the CONTEXT is gathered *before* the encoder, which therefore only ever
        sees the reduced sequence;
      - the TARGETS are gathered *after* the target encoder, which has just run
        on the whole unmasked image. That is what makes the target
        representations rich, and what separates JEPA from MAE.

    axis=1 because axis 0 is the batch: we select tokens, not images.
    """
    return tf.gather(tokens, indices, axis=1)
