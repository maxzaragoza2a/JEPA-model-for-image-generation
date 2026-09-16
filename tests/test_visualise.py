"""Tests for the mask visualiser.

A picture is what you fall back on when you no longer trust the numbers, so it
has to be trustworthy itself. Two ways it could lie, both of them quiet:

  * `patch_region` could transpose the grid, drawing a sideways picture of a
    perfectly correct mask -- and sending you hunting for a bug in `masking.py`
    that is not there;
  * `context_view` could leave a target patch visible, which would make a real
    leak look like normal output.

Only the pure array functions are covered. `plot_masked_samples` reads the
dataset and opens matplotlib, and asserting on pixels of a rendered figure
would test matplotlib rather than this module.
"""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.data import standardise
from jepa.masking import block_to_indices, sample_masks
from jepa.visualise import (
    BLANK,
    DIM,
    block_bounds,
    context_view,
    denormalise,
    patch_region,
    target_view,
)


@pytest.fixture
def cfg():
    return JEPAConfig()


@pytest.fixture
def image(cfg):
    """A picture with no pixel equal to BLANK or to zero.

    Both matter: a blanked patch is recognised by holding exactly BLANK, and a
    dimmed one by holding exactly DIM times its old value. If the source image
    could already contain those values the assertions would be ambiguous.
    """
    rng = np.random.default_rng(0)
    shape = (cfg.image_size, cfg.image_size, cfg.num_channels)
    return rng.uniform(0.6, 1.0, shape).astype("float32")


# ---- geometry --------------------------------------------------------------

@pytest.mark.parametrize("index,expected_rows,expected_cols", [
    (0, (0, 4), (0, 4)),        # top-left corner
    (7, (0, 4), (28, 32)),      # end of the first row, not start of the second
    (8, (4, 8), (0, 4)),        # start of the second row
    (10, (4, 8), (8, 12)),      # row 1, column 2
    (63, (28, 32), (28, 32)),   # bottom-right corner
])
def test_patch_region_maps_row_major(cfg, index, expected_rows, expected_cols):
    """Token i covers row i // grid_size, column i % grid_size -- in pixels.

    Indices 7 and 8 are the ones that matter: they sit either side of a row
    boundary, which is exactly where a transposed mapping stops agreeing.
    """
    rows, cols = patch_region(index, cfg)

    assert (rows.start, rows.stop) == expected_rows
    assert (cols.start, cols.stop) == expected_cols


def test_block_bounds_recovers_the_rectangle(cfg):
    """A block is a rectangle, so its bounding box is the block itself."""
    p = cfg.patch_size
    indices = block_to_indices(1, 2, 3, 4, cfg.grid_size)

    left, top, width, height = block_bounds(indices, cfg)

    assert (left, top) == (2 * p, 1 * p)
    assert (width, height) == (4 * p, 3 * p)


# ---- the views -------------------------------------------------------------

def test_context_view_hides_every_target_patch(cfg, image):
    """The visual form of the invariant: the question must not show the answer.

    Drawn over a real draw rather than a hand-picked one, so the test also
    exercises whatever `sample_masks` actually produces.
    """
    context_indices, target_indices = sample_masks(cfg, np.random.default_rng(1))
    view = context_view(image, context_indices, cfg)

    for index in np.unique(np.concatenate(target_indices)):
        rows, cols = patch_region(index, cfg)
        assert np.all(view[rows, cols] == BLANK), (
            f"target patch {index} is still visible in the context view"
        )


def test_context_view_keeps_every_context_patch_untouched(cfg, image):
    """The other half: what the encoder does see must be the real pixels."""
    context_indices, _ = sample_masks(cfg, np.random.default_rng(1))
    view = context_view(image, context_indices, cfg)

    for index in context_indices:
        rows, cols = patch_region(index, cfg)
        assert np.array_equal(view[rows, cols], image[rows, cols])


def test_context_view_blanks_everything_outside_the_context(cfg, image):
    """Anything not in the context is hidden, whatever its role.

    Targets and unused patches are blanked alike, because the encoder does not
    receive either. The context list is written out here rather than sampled:
    on an 8x8 grid the context block is usually the whole grid, so a real draw
    often leaves no unused patch at all and the case would go untested.
    """
    context_indices = np.array([0, 1, 2, 9, 40])
    view = context_view(image, context_indices, cfg)

    for index in set(range(cfg.num_patches)) - set(context_indices.tolist()):
        rows, cols = patch_region(index, cfg)
        assert np.all(view[rows, cols] == BLANK), f"patch {index} should be hidden"


def test_target_view_highlights_targets_and_dims_the_rest(cfg, image):
    """Targets at full brightness, everything else multiplied by DIM."""
    context_indices, target_indices = sample_masks(cfg, np.random.default_rng(1))
    view = target_view(image, target_indices, cfg)

    hidden = set(np.concatenate(target_indices).tolist())
    for index in hidden:
        rows, cols = patch_region(index, cfg)
        assert np.array_equal(view[rows, cols], image[rows, cols])

    for index in set(range(cfg.num_patches)) - hidden:
        rows, cols = patch_region(index, cfg)
        assert np.allclose(view[rows, cols], image[rows, cols] * DIM)


# ---- denormalise -----------------------------------------------------------

def test_denormalise_inverts_standardise(cfg):
    """Round trip through the real normalisation, back to the [0, 1] range.

    Not exact: standardise works in float32, so the two affine steps leave a
    small residue. The tolerance is tight enough that a swapped mean and std
    would still be caught.
    """
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)

    restored = denormalise(standardise(raw, cfg), cfg)

    assert np.allclose(restored, raw / 255.0, atol=1e-5)


def test_denormalise_clips_into_the_displayable_range(cfg):
    """Augmentation can push values slightly outside [0, 1]; imshow needs them in."""
    extreme = np.array([[[[-9.0, 0.0, 9.0]]]], dtype="float32")

    out = denormalise(extreme, cfg)

    assert out.min() >= 0.0 and out.max() <= 1.0
