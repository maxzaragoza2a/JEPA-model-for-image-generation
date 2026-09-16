"""Tests for the context / target sampling -- mostly about one silent bug.

`masking.py` never sees an image. It produces integers, and those integers are
only meaningful because everything else in the project numbers the patch grid
the same way: token i is the patch at row i // grid_size, column i % grid_size.
Nothing enforces that agreement. Transpose it and every shape still matches,
every test of the patch pipeline still passes, and the model simply learns from
a scrambled grid. So the centre of this file is
`test_gather_tokens_agrees_with_the_patch_ordering`, which asks for a rectangle
by its row and column and checks that the pixels handed back are the ones
actually living there.

The rest guards the two invariants the docstrings promise: a sampled block
always fits inside the grid, and no target index ever survives in the context --
if one did, the model would be handed part of the answer along with the
question, and the loss would fall for entirely the wrong reason.
"""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.masking import (
    block_to_indices,
    gather_tokens,
    sample_block,
    sample_masks,
)
from jepa.patch_embed import PatchEmbedding, extract_patches


@pytest.fixture
def cfg():
    return JEPAConfig()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---- block_to_indices ------------------------------------------------------

def test_block_to_indices_is_row_major():
    """The worked example: a 2x2 block at (1, 2) of a 4x4 grid.

        0  1  2  3
        4  5  6  7     <- row 1, columns 2 and 3
        8  9 10 11     <- row 2, columns 2 and 3
       12 13 14 15

    Hard-coded on purpose. A transposed implementation returns [9, 13, 10, 14]
    here, which is the whole point of the file.
    """
    assert list(block_to_indices(1, 2, 2, 2, 4)) == [6, 7, 10, 11]


@pytest.mark.parametrize("top,left,height,width,grid", [
    (0, 0, 1, 1, 4),      # a single patch in the corner
    (0, 0, 4, 4, 4),      # the whole grid
    (1, 2, 2, 2, 4),      # the worked example
    (3, 5, 4, 2, 8),      # taller than wide, off-centre
    (0, 7, 8, 1, 8),      # a one-column stripe against the right edge
])
def test_block_to_indices_covers_the_rectangle_exactly(top, left, height, width, grid):
    """Exactly the h*w cells of the rectangle, sorted, nothing else."""
    got = block_to_indices(top, left, height, width, grid)

    expected = [r * grid + c
                for r in range(top, top + height)
                for c in range(left, left + width)]

    assert got.shape == (height * width,)
    assert list(got) == expected
    assert list(got) == sorted(got), "the docstring promises ascending order"


# ---- sample_block ----------------------------------------------------------

@pytest.mark.parametrize("grid", [4, 8, 14])
def test_sample_block_always_fits_in_the_grid(rng, cfg, grid):
    """A rectangle running off the edge would index tokens that do not exist.

    Worth hammering: it only happens for extreme draws, so a handful of
    samples would miss it.
    """
    for _ in range(2000):
        top, left, height, width = sample_block(
            grid, cfg.target_scale, cfg.target_aspect_ratio, rng
        )
        assert 1 <= height <= grid and 1 <= width <= grid
        assert 0 <= top and top + height <= grid
        assert 0 <= left and left + width <= grid


def test_sample_block_area_tracks_the_requested_scale(rng, cfg):
    """The drawn area follows `scale` -- but only loosely, and that is expected.

    `sample_block` rounds a real-valued height and width to whole patches, and
    on an 8x8 grid one patch is over 1.5% of the total area, so the rounding
    alone pushes the realised fraction outside the requested window: measured
    0.141 to 0.250 for a requested 0.15 to 0.20. The band below is deliberately
    generous; it catches a scale that is ignored outright, not the rounding.
    """
    low, high = cfg.target_scale
    fractions = np.array([
        np.prod(sample_block(cfg.grid_size, cfg.target_scale,
                             cfg.target_aspect_ratio, rng)[2:]) / cfg.num_patches
        for _ in range(2000)
    ])

    assert fractions.min() >= low * 0.75
    assert fractions.max() <= high * 1.35
    assert low <= fractions.mean() <= high, "the centre should still land in range"


# ---- sample_masks ----------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_no_target_index_survives_in_the_context(cfg, seed):
    """The invariant that matters: the question must not contain the answer.

    Target blocks may overlap each other freely, but a single target index left
    in the context hands the model part of what it is asked to predict.
    """
    context, targets = sample_masks(cfg, np.random.default_rng(seed))

    leaked = np.intersect1d(context, np.concatenate(targets))
    assert leaked.size == 0, f"targets {leaked} survived in the context"


@pytest.mark.parametrize("seed", range(20))
def test_masks_stay_inside_the_grid_and_targets_stay_separate(cfg, seed):
    context, targets = sample_masks(cfg, np.random.default_rng(seed))

    assert len(targets) == cfg.num_target_blocks, "one array per block to predict"
    for block in targets:
        assert block.size > 0
        assert block.min() >= 0 and block.max() < cfg.num_patches

    assert list(context) == sorted(set(context)), "sorted and free of duplicates"
    if context.size:
        assert context.min() >= 0 and context.max() < cfg.num_patches


def test_masks_are_reproducible_from_the_seed(cfg):
    """Same seed, same split -- otherwise a failing run cannot be replayed."""
    first_ctx, first_tgts = sample_masks(cfg, np.random.default_rng(cfg.seed))
    second_ctx, second_tgts = sample_masks(cfg, np.random.default_rng(cfg.seed))

    assert np.array_equal(first_ctx, second_ctx)
    for a, b in zip(first_tgts, second_tgts):
        assert np.array_equal(a, b)


# ---- gather_tokens ---------------------------------------------------------

def test_gather_tokens_selects_without_transforming(cfg, rng):
    """Plain selection: the result must equal the rows it claims to copy."""
    tokens = rng.standard_normal((3, cfg.num_patches, cfg.embed_dim)).astype("float32")
    indices = np.array([0, 5, 17, 63])

    gathered = np.asarray(gather_tokens(tokens, indices))

    assert gathered.shape == (3, indices.size, cfg.embed_dim)
    assert np.array_equal(gathered, tokens[:, indices, :])


def test_gather_tokens_agrees_with_the_patch_ordering(cfg, rng):
    """The one that would catch a transposed grid.

    The expected value has to come from the image itself, never from
    `block_to_indices`. An earlier version of this test asked that function for
    the indices and then checked the gathered patches carried those same
    indices -- which is true for any convention, transposed or not, and caught
    nothing. So the reference here is the raw pixel crop: asking for the
    rectangle at rows 1-2, columns 2-3 must hand back precisely that region of
    the image, patch by patch.

    `extract_patches` is used raw, with no Dense in the way, so nothing can
    launder a mismatch.
    """
    p, g = cfg.patch_size, cfg.grid_size
    top, left, height, width = 1, 2, 2, 2
    images = rng.standard_normal(
        (1, cfg.image_size, cfg.image_size, cfg.num_channels)
    ).astype("float32")

    patches = np.asarray(extract_patches(images, p))
    indices = block_to_indices(top, left, height, width, g)

    gathered = np.asarray(gather_tokens(patches, indices))[0]

    for k in range(height * width):
        r, c = top + k // width, left + k % width
        crop = images[0, r * p:(r + 1) * p, c * p:(c + 1) * p, :].reshape(-1)
        assert np.array_equal(gathered[k], crop), (
            f"slot {k} of the block should be the patch at grid cell ({r}, {c}); "
            f"masking.py and extract_patches disagree on the grid order"
        )


def test_the_whole_pipeline_produces_the_expected_shapes(cfg):
    """image -> tokens -> context / targets, end to end.

    Also pins the asymmetry the module docstring describes: one mask is drawn
    per batch and shared, so the selected length is the same for every image
    and the tensors stay statically shaped.
    """
    batch = 2
    images = np.zeros(
        (batch, cfg.image_size, cfg.image_size, cfg.num_channels), "float32"
    )

    tokens = PatchEmbedding(cfg.patch_size, cfg.embed_dim)(images)
    assert tuple(tokens.shape) == (batch, cfg.num_patches, cfg.embed_dim)

    context_idx, target_idx = sample_masks(cfg, np.random.default_rng(cfg.seed))

    context = gather_tokens(tokens, context_idx)
    assert tuple(context.shape) == (batch, context_idx.size, cfg.embed_dim)

    for block in target_idx:
        target = gather_tokens(tokens, block)
        assert tuple(target.shape) == (batch, block.size, cfg.embed_dim)
