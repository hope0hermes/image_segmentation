from dataclasses import dataclass
from typing import Tuple, Optional

import tensorflow as tf


# These aliases are just a workaround to get pylance autocompletion :-/.
keras = tf.keras
layers = tf.keras.layers


###############################################################################
# Architecture configuration.
###############################################################################
SUBTYPE_CONFIG = dict(
    resnet18=dict(repetitions=[2, 2, 2, 2], block_type="basic"),
    resnet34=dict(repetitions=[3, 4, 6, 3], block_type="basic"),
    resnet50=dict(repetitions=[3, 4, 6, 3], block_type="bottleneck"),
    resnet101=dict(repetitions=[3, 4, 23, 3], block_type="bottleneck"),
    resnet152=dict(repetitions=[3, 8, 36, 3], block_type="bottleneck"),
)


###############################################################################
# Utils.
###############################################################################
@dataclass
class BlockLayerNames:
    """Layer names, common to a single resnet block."""
    base: str

    def __post_init__(self):
        self.base = self.base + "_"
        self.bn = self.base + "bn_"
        self.conv = self.base + "conv_"
        self.relu = self.base + "relu_"
        self.pool = self.base + "pool_"
        self.pad = self.base + "pad_"
        self.sc = self.conv + "shortcut"
        self.add = self.base + "add"


def _get_subtype(subtype: str):
    """
    Return block type and number of block repetitions for the selected ResNet
    architecture.
    """
    # Get architecture subtype.
    arch = SUBTYPE_CONFIG[subtype]
    repetitions = arch["repetitions"]
    if arch["block_type"] == "basic":
        return conv_block_basic, repetitions
    elif arch["block_type"] == "bottleneck":
        return conv_block_bottleneck, repetitions
    else:
        msn = (
            f'Supported blocks are "basic" and "bottleneck", got '
            f'{arch["block_type"]}, instead.'
        )
        raise ValueError(msn)


def _set_input(input_shape: Optional[Tuple[int, ...]], input_tensor: tf.Tensor):
    """
    Setup model's input depending on if a shape tuple or an input tensor are
    passed as input arguments.
    """
    if input_tensor is None:
        input_image = keras.layers.Input(shape=input_shape)
    else:
        if not tf.is_tensor(input_tensor):
            input_image = keras.layers.Input(input_tensor, shape=input_shape)
        else:
            input_image = input_tensor

    return input_image


def _default_bn_params(**params):
    """Default parameters for batch normalization layer."""
    defaults = dict(
        axis=3,
        momentum=0.99,
        epsilon=2e-5,
        center=True,
        scale=True,
    )
    defaults.update(params)
    return defaults


def _default_conv_params(**params):
    """Default parameters for Conv2D layer."""
    defaults = dict(
        kernel_initializer="he_uniform",
        use_bias=False,
        padding="same",
    )
    defaults.update(params)
    return defaults


###############################################################################
# Convolution blocks.
###############################################################################
def conv_block_basic(
    stage: int,
    block: int,
    filters: int,
    strides: Tuple[int, int] = (1, 1),
    clear_pass: bool = False,
):
    """Two-stack block used in resnet18 and resnet34 architectures."""
    params_conv = _default_conv_params()
    params_bn = _default_bn_params()
    names = BlockLayerNames(f"stage_{stage}_block_{block}")

    def layer(input_tensor):
        # Stack 1.
        x = layers.BatchNormalization(name=names.bn + "1", **params_bn)(input_tensor)
        x = layers.Activation("relu", name=names.relu + "1")(x)

        # Set shortcut.
        if clear_pass:
            shortcut = input_tensor
        else:
            shortcut = layers.Conv2D(
                filters, (1, 1), strides, name=names.sc, **params_conv
            )(x)

        x = layers.Conv2D(
            filters, (3, 3), strides, name=names.conv + "1", **params_conv
        )(x)

        # Stack 2.
        x = layers.BatchNormalization(name=names.bn + "2", **params_bn)(x)
        x = layers.Activation("relu", name=names.relu + "2")(x)
        x = layers.Conv2D(filters, (3, 3), name=names.conv + "2", **params_conv)(x)

        # Merge shortcut.
        x = layers.Add(name=names.add)([x, shortcut])

        return x

    return layer


def conv_block_bottleneck(
    stage: int,
    block: int,
    filters: int,
    strides: Tuple[int, int] = (1, 1),
    clear_pass: bool = False,
):
    """Three-stack, bottleneck block used in architectures deeper than resnet34."""
    params_conv = _default_conv_params()
    params_bn = _default_bn_params()
    names = BlockLayerNames(f"stage_{stage}_block_{block}")

    def layer(input_tensor):
        # Stack 1, reducing.
        x = layers.BatchNormalization(name=names.bn + "1", **params_bn)(input_tensor)
        x = layers.Activation("relu", name=names.relu + "1")(x)

        # Set shortcut.
        if clear_pass:
            shortcut = input_tensor
        else:
            shortcut = layers.Conv2D(4 * filters, (1, 1), strides, name=names.sc, **params_conv)(x)

        x = layers.Conv2D(filters, (1, 1), name=names.conv + "1", **params_conv)(x)

        # Stack 2, bottleneck.
        x = layers.BatchNormalization(name=names.bn + "2", **params_bn)(x)
        x = layers.Activation("relu", name=names.relu + "2")(x)
        x = layers.Conv2D(filters, (3, 3), strides, name=names.conv + "2", **params_conv)(x)

        # Stack 3, increase.
        x = layers.BatchNormalization(name=names.bn + "3", **params_bn)(x)
        x = layers.Activation("relu", name=names.relu + "3")(x)
        x = layers.Conv2D(4 * filters, (1, 1), name=names.conv + "3", **params_conv)(x)

        # Merge shortcut.
        x = layers.Add(name=names.add)([x, shortcut])

        return x

    return layer


###############################################################################
# Architecture parts.
###############################################################################
def stem_block(initial_filters: int = 64):
    """Initial resnet block, common to all architectures."""
    bn_params_no_scale = _default_bn_params(scale=False)
    bn_params = _default_bn_params()
    cv_params = _default_conv_params()
    names = BlockLayerNames(base="stem")

    def layer(input_tensor: tf.Tensor):
        x = layers.BatchNormalization(name=names.bn + "input", **bn_params_no_scale)(input_tensor)
        x = layers.Conv2D(initial_filters, 7, 2, name=names.conv + "1", **cv_params)(x)
        x = layers.BatchNormalization(name=names.bn + "1", **bn_params)(x)
        x = layers.Activation("relu", name=names.relu + "1")(x)
        x = layers.MaxPooling2D(3, 2, padding="same", name=names.pool + "1")(x)

        return x

    return layer


def body_blocks(subtype: str, initial_filters: int = 64):
    """Architecture subtype blocks"""
    # Select convolution block and corresponding number of repetitions. 2-stack,
    # basic block for resnet 18 and 34, and 3-stack bottleneck for deeper models.
    conv_block, repetitions = _get_subtype(subtype)

    def layer(input_tensor):
        x = input_tensor
        for stage, reps in enumerate(repetitions, start=1):
            filters = initial_filters * (2**(stage - 1))
            for block, _ in enumerate(range(reps), start=1):
                if stage == 1 and block == 1:
                    # The very 1st block doesn't reduce since there is a
                    # pooling operation right before it.
                    x = conv_block(stage, block, filters, strides=(1, 1))(x)
                elif block == 1:
                    x = conv_block(stage, block, filters, strides=(2, 2))(x)
                else:
                    x = conv_block(stage, block, filters, strides=(1, 1), clear_pass=True)(x)

        params_bn = _default_bn_params()
        names = BlockLayerNames(f"stage_{stage}_block_{block + 1}")
        x = layers.BatchNormalization(name=names.bn + "1", **params_bn)(x)
        x = layers.Activation("relu", name=names.relu + "1")(x)

        return x

    return layer


def top_block(include_top: bool = False, n_classes: Optional[int] = None):
    """Optional classifier block, common to all architectures."""
    if include_top and not isinstance(n_classes, int):
        msn = (
            "If the top classifier layer is to be included "
            "(i.e., `include_top == True`) then the number of classes, "
            f"`n_classes`, has to be provided, got {n_classes} instead"
        )
        raise ValueError(msn)

    names = BlockLayerNames(base="top")
    def layer(input_tensor):
        if include_top:
            x = layers.GlobalAveragePooling2D(name=names.pool)(input_tensor)
            x = layers.Dense(n_classes, name=names.base + "dense")(x)
            x = layers.Activation("softmax", name=names.base + "softmax")(x)

            return x
        else:
            return input_tensor

    return layer


###############################################################################
# ResNet builder.
###############################################################################
def resnet(
    subtype: str,
    input_shape: Optional[Tuple[int, ...]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    include_top: bool = False,
    n_classes: Optional[int] = None,
) -> tf.keras.Model:
    """
    Args:
        subtype: Architecture subtype. Supported subtypes are "resnet18",
            "resnet34", "resnet50", "resnet101" and "resnet152".
        input_shape: Optional, input images' shape tuple.
        input_tensor: Optional `tf.Tensor` to be used as model input. To be
            has to be provided in `input_shape` is set to `None`.
        include_top: Whether or not to include the fully connected layer at the
            top of the network.
        n_classes: Optional number of classes to classify images into. To be
            provided only if `include_top` is `True`.
    """
    # Set model input.
    model_input = _set_input(input_shape, input_tensor)

    # Build model.
    x = stem_block()(model_input)
    x = body_blocks(subtype)(x)
    x = top_block(include_top, n_classes)(x)

    return keras.Model(inputs=model_input, outputs=x, name=subtype)
