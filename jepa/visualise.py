"""Look at what the context encoder is actually given.

Everything before this point is numbers: patches, token indices, set
differences. This module turns one draw back into pictures, so a mask that is
subtly wrong -- a transposed grid, a target leaking into the context, blocks
that never move -- becomes visible instead of staying plausible.

Run it as a script:

    python -m jepa.visualise                  # 5 random images, random draw
    python -m jepa.visualise --seed 42        # reproduce a specific draw
    python -m jepa.visualise --out masks.png --no-show

Three rows are drawn for each image:

    input     the image as it leaves the data pipeline, un-normalised back to
              something the eye can read. This is the augmented, standardised
              tensor the model really gets, not the raw CIFAR file.
    context   only the patches the context encoder receives. Everything else is
              blanked, because the encoder genuinely never sees it.
    targets   what the predictor must reconstruct, at full brightness over a
              dimmed image, with one coloured outline per block.

The five images share ONE mask, and that is not a shortcut in the drawing: it
is how training works. The number of context patches changes from draw to draw
and TensorFlow cannot stack rows of different lengths, so a single mask is
drawn per batch and applied to every image in it. Seeing the same holes in all
five pictures is the pipeline being honest about itself.

Patches in neither role are left blank in both views. They are not a bug --
the context block is a rectangle with the targets cut out of it, so whatever
falls outside it is simply unused this round.
"""

import argparse

import numpy as np

from jepa.config import JEPAConfig
from jepa.data import build_pretrain_dataset
from jepa.masking import sample_masks

# One colour per target block. Chosen to stay distinguishable in greyscale and
# for the most common colour-vision deficiencies.
BLOCK_COLOURS = ["#e66100", "#5d3a9b", "#1aff92", "#40b0a6", "#dc267f", "#ffb000"]

# What a hidden patch is replaced by. Mid grey rather than black: black is a
# real pixel value in CIFAR, and a blanked region has to be unmistakable.
BLANK = 0.5

# How far the non-target areas are dimmed in the targets row.
DIM = 0.25


def denormalise(images, config: JEPAConfig):
    """Undo `standardise`, back to something displayable in [0, 1].

    The pipeline hands out standardised floats that run roughly -2 to +2.
    Multiplying by std and adding mean returns the original [0, 1] range; the
    clip only catches values that augmentation pushed slightly outside it.
    """
    images = np.asarray(images)
    return np.clip(images * np.array(config.std) + np.array(config.mean), 0.0, 1.0)


def patch_region(index: int, config: JEPAConfig):
    """Flat token index -> the pixel rows and columns it occupies.

    The same row-major convention as `extract_patches` and `block_to_indices`:
    token i lives at row i // grid_size, column i % grid_size. Getting this
    wrong here would draw a transposed picture of a correct mask, which is a
    good way to spend an afternoon chasing a bug that is not there.
    """
    row, col = divmod(int(index), config.grid_size)
    p = config.patch_size
    return slice(row * p, (row + 1) * p), slice(col * p, (col + 1) * p)


def block_bounds(indices, config: JEPAConfig):
    """Bounding box of one target block, in pixels: (left, bottom, w, h).

    A block is a rectangle, so its bounding box is the block itself. Returned
    in the corner-plus-size form matplotlib's Rectangle wants.
    """
    rows, cols = np.divmod(np.asarray(indices), config.grid_size)
    p = config.patch_size
    top, left = rows.min() * p, cols.min() * p
    height = (rows.max() - rows.min() + 1) * p
    width = (cols.max() - cols.min() + 1) * p
    return left, top, width, height


def context_view(image, context_indices, config: JEPAConfig):
    """The image reduced to the patches the context encoder is given."""
    view = np.full_like(image, BLANK)
    for index in context_indices:
        rows, cols = patch_region(index, config)
        view[rows, cols] = image[rows, cols]
    return view


def target_view(image, target_indices, config: JEPAConfig):
    """The image dimmed everywhere except the blocks to be predicted."""
    view = image * DIM
    for block in target_indices:
        for index in block:
            rows, cols = patch_region(index, config)
            view[rows, cols] = image[rows, cols]
    return view


def plot_masked_samples(config=None, seed=None, num_images=5, out=None, show=True):
    """Draw one mask, apply it to `num_images` images, and plot the result.

    Args:
        config: a JEPAConfig; the default one if omitted.
        seed: reproduces a specific draw. None picks a fresh one and prints it,
            so an interesting picture can be recovered later.
        num_images: how many images to show. They all share the one mask.
        out: path to save the figure to, or None to skip saving.
        show: open a window. Turn it off on a headless machine.

    Returns:
        The seed actually used, so a random draw can be replayed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    config = config or JEPAConfig()
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))

    rng = np.random.default_rng(seed)

    # Straight from the real pipeline, so what is drawn is what the model gets,
    # augmentation and normalisation included.
    batch = next(iter(build_pretrain_dataset(config, "train")))
    images = denormalise(batch, config)
    chosen = rng.choice(len(images), size=num_images, replace=False)

    context_indices, target_indices = sample_masks(config, rng)

    fig, axes = plt.subplots(3, num_images, figsize=(2.1 * num_images, 6.8))
    rows = ["input", f"context ({context_indices.size} patches)", "targets"]

    for column, image_index in enumerate(chosen):
        image = images[image_index]
        panels = [
            image,
            context_view(image, context_indices, config),
            target_view(image, target_indices, config),
        ]

        for row, panel in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(rows[row], fontsize=9)

        # Outline each block on the targets row, so overlapping blocks stay
        # readable as separate predictions rather than one merged shape.
        for block_number, block in enumerate(target_indices):
            left, top, width, height = block_bounds(block, config)
            axes[2, column].add_patch(Rectangle(
                (left - 0.5, top - 0.5), width, height,
                fill=False, linewidth=1.6,
                edgecolor=BLOCK_COLOURS[block_number % len(BLOCK_COLOURS)],
            ))

    hidden = np.unique(np.concatenate(target_indices)).size
    unused = config.num_patches - context_indices.size - hidden
    fig.suptitle(
        f"one mask, shared by the whole batch  --  seed {seed}\n"
        f"{context_indices.size} context, {hidden} hidden in "
        f"{len(target_indices)} blocks, {unused} unused  "
        f"(of {config.num_patches} patches)",
        fontsize=10,
    )
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved to {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None,
                        help="reproduce a specific draw (default: random)")
    parser.add_argument("--images", type=int, default=5,
                        help="how many images to show (default: 5)")
    parser.add_argument("--out", default="masked_samples.png",
                        help="where to save the figure")
    parser.add_argument("--no-show", action="store_true",
                        help="save without opening a window")
    args = parser.parse_args()

    seed = plot_masked_samples(
        seed=args.seed,
        num_images=args.images,
        out=args.out,
        show=not args.no_show,
    )
    print(f"seed {seed} -- pass --seed {seed} to draw this mask again")


if __name__ == "__main__":
    main()
