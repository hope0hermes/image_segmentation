import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Callable, Iterable, Dict

import tensorflow as tf


# These aliases are just a workaround to get pylance autocompletion :-/.
keras = tf.keras
layers = tf.keras.layers


###############################################################################
# Architecture configuration.
###############################################################################
KerasActivation = Callable


SUBTYPE_CONFIG = dict(
    efficientnet_B0=dict(width_coef=1.0, depth_coef=1.0, resolution=224, dropout_rate=0.2),
    efficientnet_B1=dict(width_coef=1.0, depth_coef=1.1, resolution=240, dropout_rate=0.2),
    efficientnet_B2=dict(width_coef=1.1, depth_coef=1.2, resolution=260, dropout_rate=0.3),
    efficientnet_B3=dict(width_coef=1.2, depth_coef=1.4, resolution=300, dropout_rate=0.3),
    efficientnet_B4=dict(width_coef=1.4, depth_coef=1.8, resolution=380, dropout_rate=0.4),
    efficientnet_B5=dict(width_coef=1.6, depth_coef=2.2, resolution=456, dropout_rate=0.4),
    efficientnet_B6=dict(width_coef=1.8, depth_coef=2.6, resolution=528, dropout_rate=0.5),
    efficientnet_B7=dict(width_coef=2.0, depth_coef=3.1, resolution=600, dropout_rate=0.5),
)


BASE_BLOCK_ARGS = [
    dict(kernel_size=3, repeats=1, filters_in=32, filters_out=16,
         expand_ratio=1, id_skip=True, strides=1, se_ratio=0.25),
    dict(kernel_size=3, repeats=2, filters_in=16, filters_out=24,
         expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    dict(kernel_size=5, repeats=2, filters_in=24, filters_out=40,
         expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    dict(kernel_size=3, repeats=3, filters_in=40, filters_out=80,
         expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    dict(kernel_size=5, repeats=3, filters_in=80, filters_out=112,
         expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25),
    dict(kernel_size=5, repeats=4, filters_in=112, filters_out=192,
         expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    dict(kernel_size=3, repeats=1, filters_in=192, filters_out=320,
         expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25),
]


CONV_KERNEL_INITIALIZER = dict(
    class_name="VarianceScaling",
    config=dict(
        scale=2.0,
        mode="fan_out",
        distribution="normal",
    ),
)


DENSE_KERNEL_INITIALIZER = dict(
    class_name="VarianceScaling",
    config=dict(
        scale=1. / 3.,
        mode="fan_out",
        distribution="uniform",
    ),
)


###############################################################################
# Utils.
###############################################################################
@dataclass
class BlockLayerNames:
    """Layer names, common to a single block."""
    base: str

    def __post_init__(self):
        self.base = self.base + "_"
        self.bn = self.base + "bn"
        self.conv = self.base + "conv"
        self.act = self.base + "act"
        self.pool = self.base + "pool"
        self.pad = self.base + "pad"
        self.sc = self.conv + "shortcut"
        self.add = self.base + "add"
        self.drop = self.base + "drop"


def _set_input(
    input_shape: Optional[Tuple[int, ...]] = None,
    input_tensor: tf.Tensor = None,
) -> tf.Tensor:
    """
    Setup model's input depending on if a shape tuple or an input tensor are
    passed as input arguments.

    Arguments:
        input_shape: Either `None` (in which case a default `channel_last`, rgb
            input will be set) or a user-provided shape to be validated.
        input_tensor: Tensor (i.e., the output of `layers.Input()`) to be used
            as model input.

    Returns:
        Validated model input.
    """
    # Validate input shape.
    if input_shape is None:
        input_shape = (None, None, 3)
    else:
        msn = (
            f"`input_shape` must be a `channel-last` tuple of 3 integers. "
            f"got {input_shape} instead."
        )
        if len(input_shape) != 3:
            raise ValueError(msn)
        if input_shape[2] is None:
            raise ValueError(msn)

    # If provided, use input tensor to setup model's input.
    if input_tensor is None:
        input_image = keras.layers.Input(shape=input_shape)
    else:
        if not tf.is_tensor(input_tensor):
            input_image = keras.layers.Input(input_tensor, shape=input_shape)
        else:
            input_image = input_tensor

    return input_image


def _get_subtype_parameters(subtype: str) -> dict:
    """Get configuration parameters for the selected architecture subtype."""
    if subtype in SUBTYPE_CONFIG.keys():
        return SUBTYPE_CONFIG[subtype]
    msn = (
        f"Invalid architecture subtype. Valid subtypes are: "
        f"{list(SUBTYPE_CONFIG.keys())}, got '{subtype}' instead."
    )
    raise ValueError(msn)


def round_filters(filters: int, width_coef: float, divisor: int) -> int:
    """
    Rounds the number of filters, based on the width multiplier.

    Arguments:
        filters: Number of input filters.
        width_coef: Scaling coefficient for the networks width.
        divisor: Divisor for channel rounding.

    Returns:
        Rounded number of filters.
    """
    filters_th = filters * width_coef
    filters_new = max(divisor, int(filters_th + divisor / 2) // divisor * divisor)
    # Make sure rounding doesn't go down by more than 10%.
    if filters_new < 0.9 * filters_th:
        filters_new += divisor
    return int(filters_new)


def round_repeats(repeats: int, depth_coef: float) -> int:
    """
    Rounds number of repeats, based on depth multiplier.

    Arguments:
        repeats: Number of input, block repeats.
        depth_coef: Scaling coefficient for the network's depth.

    Return:
        Scaled number of block repetitions.
    """

    return int(math.ceil(depth_coef * repeats))


def correct_pad(
    inputs: tf.Tensor, kernel_size: Union[Tuple[int, int], int]
) -> Tuple[Tuple[int, ...], ...]:
    """
    Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments:
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    Returns:
        A tuple.
    """
    img_dim = 1
    input_size = keras.backend.int_shape(inputs)[img_dim: (img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


###############################################################################
# Squeeze and excitation block.
###############################################################################
def squeeze_and_excitation_block(
    filters: int,
    activation: Union[KerasActivation, str] = "swish",
    base_name: str = "",
    kernel_initializer: Union[Dict, str] = "glorot_uniform",
):
    """
    Squeeze and excitation block with convolution bottleneck.

    Arguments:
        filters: Number of filters in the squeezed projection (bottleneck).
        activation: Bottleneck's activation function.
        base_name: Preffix for layer names.
        kernel_initializer: Configuration parameters for the initialization of
            the convolutional kernel.

    Return:
        Squeeze and excitation block on top of the input tensor.
    """
    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        name = "se_" if base_name == "" else base_name
        filters_org = input_tensor.shape[-1]

        # Reshape.
        squeeze_shape = (1, 1, filters_org)
        x = layers.GlobalAveragePooling2D(name=name + "squeeze")(input_tensor)
        x = layers.Reshape(squeeze_shape, name=name + "reshape")(x)

        # Compute weights.
        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name + "reduce",
        )(x)
        x = layers.Conv2D(
            filters=filters_org,
            kernel_size=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=kernel_initializer,
            name=name + "expand",
        )(x)

        # Excite.
        x = layers.multiply([input_tensor, x], name=name + "excite")

        return x

    return layer


###############################################################################
# Inverted residual block.
###############################################################################
def mb_block(
    activation: Union[KerasActivation, str] = "swish",
    drop_rate: float = 0.0,
    base_name: str = "",
    filters_in: int = 32,
    filters_out: int = 16,
    kernel_size: int = 3,
    strides: int = 1,
    expand_ratio: int = 1,
    se_ratio: float = 0.0,
    id_skip: bool = True,
):
    """
    Inverted residual block, with squeeze and excitation.

    Arguments:
        activation: Activation function.
        drop_rate: Fraction of intput units to be drop.
        base_name: Block name preffix.
        filters_in: Number of input filters.
        filters_out: Number of output filters.
        kernel_size: Size of the convolution window.
        strides: Size of the convolution stride.
        expand_ratio: Scaling coefficient for the input filters.
        se_ratio: Fraction to squeeze the input filters.
        id_skip: Boolean flag for the shortcut connection.

    Returns:
        Inverted residual block.
    """

    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        # Expansion phase.
        names = BlockLayerNames(base=base_name + "_expand")
        if expand_ratio > 1:
            x = layers.Conv2D(
                filters=filters_in * expand_ratio,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                name=names.conv,
            )(input_tensor)
            x = layers.BatchNormalization(name=names.bn)(x)
            x = layers.Activation(activation, name=names.act)(x)
        else:
            x = input_tensor

        # Depthwise phase.
        names = BlockLayerNames(base=base_name + "_dw")
        if strides == 2:
            x = layers.ZeroPadding2D(
                padding=correct_pad(x, kernel_size),
                name=names.pad,
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"

        x = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=names.conv
        )(x)
        x = layers.BatchNormalization(name=names.bn)(x)
        x = layers.Activation(activation, name=names.act)(x)

        # Squeeze and excitation phase.
        if 0 < se_ratio < 1:
            x = squeeze_and_excitation_block(
                filters=max(1, int(filters_in * se_ratio)),
                activation=activation,
                base_name=base_name + "_se_",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
            )(x)

        # Output phase.
        names = BlockLayerNames(base=base_name + "_out")
        x = layers.Conv2D(
            filters=filters_out,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=names.conv,
        )(x)
        x = layers.BatchNormalization(name=names.bn)(x)
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = layers.Dropout(
                    rate=drop_rate,
                    noise_shape=(None, 1, 1, 1),
                    name=names.drop,
                )(x)
            x = layers.add([x, input_tensor], name=names.add)

        return x

    return layer


###############################################################################
# Architecture parts.
###############################################################################
def stem_block(
    width_coef: float,
    divisor: int,
    input_filters: int = 32,
    activation: Union[KerasActivation, str] = "swish",
):
    """
    Efficientnet stem block.

    Arguments:
        width_coef: Scaling coefficient for the networks width.
        divisor: Divisor for channel rounding.
        activation: Activation function.
        input_filters: Initial number of filters.

    Returns:
        Stem block.
    """
    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        names = BlockLayerNames(base="stem")
        x = layers.ZeroPadding2D(
            padding=correct_pad(input_tensor, kernel_size=3),
            name=names.pad,
        )(input_tensor)
        x = layers.Conv2D(
            round_filters(input_filters, width_coef, divisor),
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=names.conv,
        )(x)
        x = layers.BatchNormalization(name=names.bn)(x)
        x = layers.Activation(activation, name=names.act)(x)

        return x

    return layer


def body_blocks(
    width_coef: float,
    depth_coef: float,
    resolution: int,
    drop_connect_rate: float = 0.2,
    divisor: int = 8,
    base_blocks: Iterable = BASE_BLOCK_ARGS,
    activation: Union[KerasActivation, str] = "swish",
    base_name: str = "block",
):
    """
    Efficientnet body blocks.

    Arguments:
        width_coef: Scaling coefficient for the network width.
        depth_coef: Scaling coefficient for the network depth.
        resolution: Input image size.
        drop_connect_rate: Dropout rate at the skip connections.
        divisor: Network width unit.
        base_blocks: Set of configuration parameters for the base block, i.e.,
            efficientnet_B0 architecture.
        activation: Activation function.
        base_name: Block name preffix.

    Returns:
        Stack of efficientnet body blocks.
    """
    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        tot_reps = sum(
            round_repeats(block["repeats"], depth_coef) for block in base_blocks
        )

        x = input_tensor
        cnt_reps = 0
        for block_idx, block in enumerate(base_blocks):
            block = deepcopy(block)

            # Update width and depth based on scaling coefficients.
            block["filters_in"] = round_filters(block["filters_in"], width_coef, divisor)
            block["filters_out"] = round_filters(block["filters_out"], width_coef, divisor)
            block["repeats"] = round_repeats(block["repeats"], depth_coef)

            for rep in range(block["repeats"]):
                # Only the first repetition of each block takes care of stride
                # and filter size increase.
                if rep > 0:
                    block["strides"] = 1
                    block["filters_in"] = block["filters_out"]

                x = mb_block(
                    activation=activation,
                    drop_rate=drop_connect_rate * cnt_reps / tot_reps,
                    base_name=f"{base_name}_{block_idx + 1}{chr(rep + 97)}",
                    filters_in=block["filters_in"],
                    filters_out=block["filters_out"],
                    kernel_size=block["kernel_size"],
                    strides=block["strides"],
                    expand_ratio=block["expand_ratio"],
                    se_ratio=block["se_ratio"],
                    id_skip=block["id_skip"],
                )(x)
                print(f"{base_name}_{block_idx + 1}{chr(rep + 97)}")
                cnt_reps += 1

        return x

    return layer


def top_block(
    width_coef: int,
    divisor: int = 8,
    include_top: bool = True,
    dropout_rate: float = 0.2,
    pooling: Optional[str] = None,
    activation: Union[KerasActivation, str] = "swish",
    classifier_activation: Union[KerasActivation, str] = "softmax",
    n_classes: int = 1000,
):
    """
    Efficientnet top block.

    Arguments:
        width_coef: Scaling coefficient for the network width.
        divisor: Network width unit.
        include_top: Whether or not to include the fully connected layer at the
            top of the network.
        dropout_rate: Dropout rate.
        pooling: Pooling mode for feature extraction when `include_top` is `False`.
            - `None` means the model's output will be the 4D tensor from the
                last convolutional layer.
            - `avg` will apply a global pooling averaging on the output of the
                last convolutional layer, resulting in a 2D output tensor.
            - `max` behaves as `avg` but applies max global pooling instead.
        activation: Activation function for the last hidden layer.
        classifier_activation: Activation function for the classifier layer. If
            `None`, logits of the top layer will be returned.
        n_classes: Number of classes to classify images into. To be provided
            only if `include_top` is `True.`

    Returns:
        Top block.
    """
    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        names = BlockLayerNames(base="top")
        x = layers.Conv2D(
            filters=round_filters(1280, width_coef, divisor),
            kernel_size=1,
            padding="same",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=names.conv
        )(input_tensor)
        x = layers.BatchNormalization(name=names.bn)(x)
        x = layers.Activation(activation, name=names.act)(x)

        if include_top:
            if dropout_rate > 0:
                x = layers.Dropout(rate=dropout_rate, name=names.drop)(x)
            x = layers.Dense(
                units=n_classes,
                activation=classifier_activation,
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                name=names.base + "pred"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name=names.pool)(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name=names.pool)(x)

        return x

    return layer


###############################################################################
# EfficientNet builder.
###############################################################################
def efficientnet(
    subtype: str,
    input_shape: Optional[Tuple[int, ...]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    divisor: int = 8,
    include_top: bool = False,
    n_classes: Optional[int] = None,
    pooling: Optional[str] = None,
    activation: Union[KerasActivation, str] = "swish",
    classifier_activation: Union[KerasActivation, str] = "softmax",
    normalize_input: bool = False,
) -> tf.keras.Model:
    """
    Arguments:
        subtype: Architecture subtype. Supported subtypes are "efficientnet_BX"
            with X in [0, ..., 7].
        input_shape: Input shape tuple. If `None`, a 3-channel image, with the
            default resolution for the specified architecture subtype will be
            used.
        input_tensor: `tf.Tensor` to be used as model input. Has to be provided
            if `input_shape` is set to `None`.
        divisor: Network's width unit.
        include_top: Whether or not to include the fully connected layer at the
            top of the network.
        n_classes: Number of classes to classify images into. To be provided
            only if `include_top` is `True.`
        pooling: Pooling mode for feature extraction when `include_top` is `False`.
            - `None` means the model's output will be the 4D tensor from the
                last convolutional layer.
            - `avg` will apply a global pooling averaging on the output of the
                last convolutional layer, resulting in a 2D output tensor.
            - `max` behaves as `avg` but applies max global pooling instead.
        activation: Activation function to be used for feature extraction.
        classifier_activation: Activation function for the classifier layer. If
            `None`, logits of the top layer will be returned.
        normalize_input: Wether or not scale (8-bit scaling) and normalize the
            model input.

    Returns.
        A `keras.Model` instance.
    """
    # Select architecture subtype, scaling coefficients.
    arch_coefs = _get_subtype_parameters(subtype)

    # Set model input.
    if input_shape is None:
        input_shape = (arch_coefs["resolution"], arch_coefs["resolution"], 3)
    model_input = _set_input(input_shape, input_tensor)

    # Normalize input.
    x = model_input
    if normalize_input:
        x = layers.experimental.preprocessing.Rescaling(1. / 255.)(x)
        x = layers.BatchNormalization()(x)

    # Build model.
    x = stem_block(
        width_coef=arch_coefs["width_coef"],
        divisor=divisor,
        input_filters=32,
        activation=activation,
    )(x)

    x = body_blocks(
        width_coef=arch_coefs["width_coef"],
        depth_coef=arch_coefs["depth_coef"],
        resolution=arch_coefs["resolution"],
        divisor=divisor,
        base_blocks=BASE_BLOCK_ARGS,
        activation=activation,
        base_name="block",
    )(x)

    x = top_block(
        width_coef=arch_coefs["width_coef"],
        divisor=divisor,
        include_top=include_top,
        dropout_rate=arch_coefs["dropout_rate"],
        pooling=pooling,
        activation=activation,
        classifier_activation=classifier_activation,
        n_classes=n_classes,
    )(x)

    return keras.Model(inputs=model_input, outputs=x, name=subtype)