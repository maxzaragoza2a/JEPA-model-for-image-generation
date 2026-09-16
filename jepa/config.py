"""Every hyperparameter for the project, in one dataclass.

Defaults are deliberately small: TensorFlow >= 2.11 has no native-Windows GPU
support, so this trains on CPU. The shapes below (32x32 images, 4x4 patches,
64 tokens, 128-dim, 4 layers) keep a pretraining epoch tractable.
"""

from dataclasses import dataclass


@dataclass
class JEPAConfig:
    # ---- Data -------------------------------------------------------------
    image_size: int = 32          # CIFAR-10 is 32x32
    num_channels: int = 3
    patch_size: int = 4           # -> 8x8 grid = 64 patches per image

    # Per-channel CIFAR-10 statistics, used to standardise inputs.
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)

    # ---- Encoder (context and target share this architecture) -------------
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # ---- Predictor --------------------------------------------------------
    # I-JEPA makes the predictor deliberately narrow and shallow, so it cannot
    # simply memorise the target encoder; it has to rely on the context.
    predictor_embed_dim: int = 64
    predictor_depth: int = 2
    predictor_num_heads: int = 4

    # ---- Masking ----------------------------------------------------------
    # The self-supervised task: hide a few rectangular blocks of the patch grid
    # and ask the model to predict what is in them, given the rest.
    #
    # Blocks -- not scattered patches -- because neighbouring patches look alike,
    # so an isolated hidden patch can be interpolated from its visible
    # neighbours. A whole rectangle cannot: the model has to understand.
    num_target_blocks: int = 4        # how many rectangles to predict
    target_scale: tuple = (0.15, 0.20)   # area of one target, as a fraction of the grid
    target_aspect_ratio: tuple = (0.75, 1.5)  # height/width; 1.0 would be square
    context_scale: tuple = (0.85, 1.00)  # area of the single context block

    # ---- Training ---------------------------------------------------------
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3

    # Momentum for the EMA update of the target encoder, annealed to ema_end.
    ema_start: float = 0.996
    ema_end: float = 1.0

    seed: int = 42

    # ---- Derived shapes ---------------------------------------------------
    @property
    def grid_size(self) -> int:
        """Number of patches along one side (32 / 4 = 8)."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} is not divisible by "
                f"patch_size {self.patch_size}"
            )
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total tokens per image (8 * 8 = 64)."""
        return self.grid_size ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened size of one raw patch (4 * 4 * 3 = 48)."""
        return self.patch_size * self.patch_size * self.num_channels
