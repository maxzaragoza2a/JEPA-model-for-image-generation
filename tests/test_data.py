"""Tests for the CIFAR-10 input pipeline.

This is the one part of the preprocessing that touches actual pixels, and until
now nothing exercised it. The failures it can hide are all quiet ones: swapping
`mean` and `std`, dropping the `/255`, or letting augmentation leak into the
evaluation split would each degrade training without ever raising.

Two notes on how the assertions are written.

First, the normalisation constants are checked against the dataset itself
rather than against a copy of the same numbers -- otherwise the test would only
prove that config.py equals config.py. Standard CIFAR-10 statistics circulate
in at least two incompatible versions, so this is worth pinning.

Second, the statistics are accumulated in float64 on purpose. Over the full
50000-image train split there are 51 million values per channel, and a float32
accumulator loses enough precision to report a standard deviation of 0.93 for
data that is in fact scaled to 1.000. That is an artefact of the measurement,
not of the pipeline, and a test that ignored it would fail for the wrong reason.

These tests read the real dataset from the Keras cache (~170MB, downloaded on
first use), which makes them slower than the rest of the suite.
"""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.data import (
    build_pretrain_dataset,
    build_probe_dataset,
    load_cifar10,
    standardise,
)


@pytest.fixture(scope="session")
def cfg():
    return JEPAConfig()


@pytest.fixture(scope="session")
def cifar():
    """Loaded once for the whole session: each call re-reads 170MB from disk."""
    return load_cifar10()


# ---- the constants ---------------------------------------------------------

def test_normalisation_constants_match_the_dataset(cfg, cifar):
    """config.mean / config.std must be CIFAR-10's actual statistics.

    Checked against the pixels, not against another copy of the numbers. A
    widely copied alternative set of standard deviations exists -- roughly
    (0.2023, 0.1994, 0.2010) -- and silently using it would leave the inputs
    mis-scaled by about 20%.
    """
    (x_train, _), _ = cifar
    pixels = x_train.astype(np.float64) / 255.0

    assert np.allclose(pixels.mean(axis=(0, 1, 2)), cfg.mean, atol=1e-4)
    assert np.allclose(pixels.std(axis=(0, 1, 2)), cfg.std, atol=1e-4)


# ---- standardise -----------------------------------------------------------

def test_standardise_centres_and_scales_the_real_data(cfg, cifar):
    """The point of the operation: zero mean, unit variance, per channel."""
    (x_train, _), _ = cifar

    out = np.asarray(standardise(x_train, cfg))

    # float64 accumulation -- see the module docstring.
    assert np.allclose(out.mean(axis=(0, 1, 2), dtype=np.float64), 0.0, atol=1e-3)
    assert np.allclose(out.std(axis=(0, 1, 2), dtype=np.float64), 1.0, atol=1e-3)


def test_standardise_is_the_exact_affine_map(cfg):
    """A hand-computed case, so the formula cannot be subtly inverted.

    An all-255 image must land on (1.0 - mean) / std. Swapping mean and std, or
    forgetting the /255, both move this value a long way.
    """
    white = np.full((1, 4, 4, 3), 255, np.uint8)

    out = np.asarray(standardise(white, cfg))
    expected = (1.0 - np.array(cfg.mean)) / np.array(cfg.std)

    assert np.allclose(out[0, 0, 0], expected, atol=1e-5)
    assert np.allclose(out, expected, atol=1e-5), "a constant image stays constant"


def test_standardise_returns_float32(cfg):
    """uint8 in, float32 out -- the model never sees integers."""
    out = standardise(np.zeros((1, 4, 4, 3), np.uint8), cfg)

    assert out.dtype == "float32"


# ---- augment ---------------------------------------------------------------

def test_augment_preserves_shape_and_dtype(cfg, cifar):
    """Pad to 40x40 then crop back: the image must come out the size it went in."""
    import tensorflow as tf
    from jepa.data import augment

    (x_train, _), _ = cifar
    image = tf.constant(x_train[0])

    out = augment(image, cfg)

    assert tuple(out.shape) == (cfg.image_size, cfg.image_size, cfg.num_channels)
    assert out.dtype == image.dtype


def test_augment_only_moves_pixels_it_was_given(cfg, cifar):
    """Reflect-pad, crop and flip rearrange pixels; they never blend them.

    So every value coming out must already have been in the image. A resize or
    an interpolating transform slipped into the chain would invent intermediate
    values and break this.
    """
    import tensorflow as tf
    from jepa.data import augment

    (x_train, _), _ = cifar
    image = tf.constant(x_train[0])
    original = set(np.asarray(image).ravel().tolist())

    for _ in range(20):
        produced = set(np.asarray(augment(image, cfg)).ravel().tolist())
        assert produced <= original


def test_augment_actually_augments(cfg, cifar):
    """A no-op augmentation would be invisible everywhere else.

    Not every draw changes the image -- the crop can land back on centre and
    the flip is a coin toss -- so this asserts over a run of draws.
    """
    import tensorflow as tf
    from jepa.data import augment

    (x_train, _), _ = cifar
    image = tf.constant(x_train[0])
    reference = np.asarray(image)

    changed = sum(
        not np.array_equal(np.asarray(augment(image, cfg)), reference)
        for _ in range(20)
    )
    assert changed > 0, "augment returned the input image 20 times in a row"


# ---- the datasets ----------------------------------------------------------

@pytest.fixture(scope="module")
def pretrain_ds(cfg):
    return build_pretrain_dataset(cfg, "train")


@pytest.fixture(scope="module")
def probe_test_ds(cfg):
    return build_probe_dataset(cfg, "test")


def test_pretrain_dataset_yields_unlabelled_batches(cfg, pretrain_ds):
    """JEPA pretraining is self-supervised: a label here would be a design bug.

    The batch is also full rather than ragged -- drop_remainder is what keeps
    the token shapes static downstream.
    """
    batch = next(iter(pretrain_ds))

    assert not isinstance(batch, tuple), "pretraining must not carry labels"
    assert tuple(batch.shape) == (
        cfg.batch_size, cfg.image_size, cfg.image_size, cfg.num_channels
    )
    assert batch.dtype == "float32"


def test_pretrain_batches_differ_between_epochs(cfg, pretrain_ds):
    """Shuffling and augmentation must stay alive across iterations.

    If either froze, every epoch would show the model the same 128 images in
    the same order and pretraining would quietly collapse to memorisation.
    """
    first = np.asarray(next(iter(pretrain_ds)))
    second = np.asarray(next(iter(pretrain_ds)))

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("split", ["train", "test"])
def test_probe_dataset_yields_image_label_pairs(cfg, split):
    """The probe needs labels; the images must still be normalised floats."""
    images, labels = next(iter(build_probe_dataset(cfg, split)))

    assert tuple(images.shape)[1:] == (
        cfg.image_size, cfg.image_size, cfg.num_channels
    )
    assert images.dtype == "float32"
    assert tuple(labels.shape) == (tuple(images.shape)[0],)
    assert 0 <= int(np.min(labels)) and int(np.max(labels)) <= 9


def test_the_test_split_is_never_augmented(cfg, cifar, probe_test_ds):
    """The one that matters most, and the easiest to break by accident.

    `build_probe_dataset` couples shuffling and augmentation in a single
    `if shuffle:` block, so a change there could start augmenting evaluation
    data -- which would make the reported accuracy meaningless without any
    visible symptom.

    Comparing two passes only proves determinism; augmentation with a fixed
    seed would pass that. So the batch is compared against the raw test images
    put through `standardise` alone: anything else in the chain shows up.
    """
    _, (x_test, _) = cifar

    images, _ = next(iter(probe_test_ds))
    expected = np.asarray(standardise(x_test[: cfg.batch_size], cfg))

    assert np.array_equal(np.asarray(images), expected), (
        "the test split went through something other than standardise, "
        "or its order changed"
    )


def test_the_test_split_is_stable_across_passes(cfg, probe_test_ds):
    """Two evaluations of the same frozen encoder must be comparable."""
    first = np.asarray(next(iter(probe_test_ds))[0])
    second = np.asarray(next(iter(probe_test_ds))[0])

    assert np.array_equal(first, second)
