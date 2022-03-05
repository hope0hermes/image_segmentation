from typing import Optional, Union, Tuple, Callable, Iterable

import numpy as np
import tensorflow as tf

from backbone_factory import BackboneFactory
from classification_models.resnet import SUBTYPE_CONFIG as SUBTYPES_RESNET
from classification_models.efficientnet import SUBTYPE_CONFIG as SUBTYPES_EFFICIENTNET


KerasActivation = Callable


# These aliases are just a workaround to get pylance autocompletion :-/.
keras = tf.keras
layers = tf.keras.layers


###############################################################################
# Defaults.
###############################################################################
DEFAULT_SKIPS_RESNET = (
    "stage_4_block_1_relu_1",
    "stage_3_block_1_relu_1",
    "stage_2_block_1_relu_1",
    "stem_relu_1",
    None,
)


DEFAULT_SKIPS_EFFICIENTNET = (
    "block_6a_expand_act",
    "block_4a_expand_act",
    "block_3a_expand_act",
    "block_2a_expand_act",
    None,
)


###############################################################################
# Utils.
###############################################################################
def _validate_skip_connetion_names(backbone: tf.keras.Model, skips: Iterable[str]):
    backbone_layers = set([x.name for x in backbone.layers] + [None])
    invalid_skips = set(skips) - backbone_layers

    if len(invalid_skips) > 0:
        raise ValueError(
            f"Can't find requested skip connections:\n\n{invalid_skips}\n\n"
            f"in {backbone.name} layers:\n\n{backbone_layers}\n\n"
        )


def _get_default_backbone_skip_names(backbone: tf.keras.Model) -> Tuple:
    if backbone.name in SUBTYPES_RESNET.keys():
        return DEFAULT_SKIPS_RESNET
    elif backbone.name in SUBTYPES_EFFICIENTNET.keys():
        return DEFAULT_SKIPS_EFFICIENTNET
    else:
        raise ValueError(f"Invalid backbone {backbone.name}")


def _get_skip_connections(
    backbone: tf.keras.Model, skip_names: Optional[Iterable[str]]
) -> Iterable[tf.Tensor]:
    if skip_names is None:
        skip_names = _get_default_backbone_skip_names(backbone)
    _validate_skip_connetion_names(backbone, skip_names)

    return [
        None if x is None else backbone.get_layer(name=x).output for x in skip_names
    ]


def _check_all_iterables_same_size(**kwargs):
    iter_names = [*kwargs]
    iter_vals = [kwargs[name] for name in iter_names]
    if any([len(x) != len(y) for x, y in zip(iter_vals, iter_vals[1:])]):
        sizes = "\n".join(
            [f"len({key}) = {len(val)} | {key} = {val}" for key, val in kwargs.items()]
        )
        raise ValueError(
            f"All iterables {iter_names} must have the same length. "
            f"Instead we got:\n{sizes}"
        )


###############################################################################
# Decoder blocks.
###############################################################################
def conv_norm_act(
    filters: int,
    use_bachnorm: bool,
    kernel_size: int = 3,
    activation: str = "relu",
    kernel_initializer: str = "he_uniform",
    padding: str = "same",
    name: str = "",
):
    with tf.name_scope("conv_norm_act"):
        def layer(input_tensor):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                kernel_initializer=kernel_initializer,
                name=name + "_conv",
            )(input_tensor)

            if use_bachnorm:
                x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)

            x = tf.keras.layers.Activation(activation, name=name + "_act")(x)

            return x

        return layer


def decoder_upsample_block(filters: int, up_rate: int, use_batchnorm: bool, name: str):
    def layer(input_tensor: tf.Tensor, skip=tf.Tensor):
        x = tf.keras.layers.UpSampling2D(
            size=up_rate, name=name + "_upsample"
        )(input_tensor)
        if skip is not None:
            x = tf.keras.layers.Concatenate(name=name + "_concat")([x, skip])
        x = conv_norm_act(filters, use_batchnorm, name=name + "_block_1")(x)
        x = conv_norm_act(filters, use_batchnorm, name=name + "_block_2")(x)
        return x
    return layer


def decoder_transpose_block(filters: int, up_rate: int, use_batchnorm: bool, name: str):
    def layer(input_tensor: tf.Tensor, skip=tf.Tensor):
        # Upsample-batchnorm-activate.
        x = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel=4,
            strides=up_rate,
            padding="same",
            name=name + "_transpose",
            use_bias=not use_batchnorm,
        )(input_tensor)
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
        x = tf.keras.layers.Activation("relu", name=name + "_act")(x)

        # Merge.
        if skip is not None:
            x = tf.keras.layers.Concatenate(name=name + "_concat")([x, skip])

        x = conv_norm_act(filters, use_batchnorm, name=name + "_block_1")(x)

        return x

    return layer


def decoder_block_selector(decoder_name):
    decoders = dict(
        upsample=decoder_upsample_block,
        transpose=decoder_transpose_block,
    )

    decoder_block = decoders.get(decoder_name, None)

    if decoder_block is None:
        raise ValueError(
            f"Invalid decoder block `{decoder_name}`. Valid decoders are "
            f"{list(decoders.keys())}"
        )

    return decoder_block

###############################################################################
# Model.
###############################################################################
def unet(
    backbone_name: str,
    input_shape: Tuple[int, int, int],
    n_classes: int,
    activation: Union[KerasActivation, str] = "sigmoid",
    skips: Optional[Iterable[str]] = None,
    filters: Iterable[int] = (256, 128, 64, 32, 16),
    up_rates: Iterable[int] = (2, 2, 2, 2, 2),
    decoder_type: str = "upsample",
    use_batchnorm: bool = False,
    # DEV: to delete.
    include_top: bool = False,
):
    """
    Arguments:
        backbone: Backbone architecture.
        input_shape:
        n_classes:
        activation:
        encoder_freeze:
        decoder_block:
        decoder_batchnorm:

    Returns:
        ``keras.Model``: **Unet**.
    """
    # Setup backbone
    config = dict(
        input_shape=input_shape,
        include_top=include_top,
        n_classes=n_classes,
    )
    backbone = BackboneFactory(arch=backbone_name, config=config).get()

    # Skip connections.
    skip_tensors = _get_skip_connections(backbone, skips)
    for x in skip_tensors:
        print(x)

    # Verify upsampling rates, filters and skip connections, all have the same length.
    _check_all_iterables_same_size(skips=skip_tensors, filters=filters, up_rates=up_rates)

    # Select decoder block type.
    decoder_block = decoder_block_selector(decoder_type)

    # Build decoder.
    input_tensor = backbone.input
    x = backbone.output
    for b_idx, (b_flt, b_urt, b_skp) in enumerate(zip(filters, up_rates, skip_tensors)):
        x = decoder_block(
            filters=b_flt,
            up_rate=b_urt,
            use_batchnorm=use_batchnorm,
            name=f"decoder_block_{b_idx}",
        )(x, b_skp)

    x = tf.keras.layers.Conv2D(
        filters=n_classes,
        kernel_size=3,
        padding="same",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        name="final_conv",
    )(x)
    x = tf.keras.layers.Activation(activation=activation, name=activation)(x)

    model = tf.keras.Model(input_tensor, x)

    # backbone.summary()
    keras.utils.plot_model(
        model,
        f"graph_unet_{backbone_name}.png",
        show_shapes=True,
        expand_nested=False,
    )

    return model


###############################################################################
# Builder.
###############################################################################
def main():
    backbone_name = "efficientnet_B0"
    backbone_name = "resnet18"
    shape = (224, 224, 3)
    model = unet(
        backbone_name=backbone_name,
        input_shape=shape,
        n_classes=2,
        include_top=False,
        # up_rates=(1, 2, 3, 4, 5, 6, 7),
    )

    # To delete.
    from pathlib import Path
    from datetime import datetime
    from time import sleep
    from shutil import rmtree

    logdir = Path("logs/fit/")
    if logdir.is_dir():
        rmtree(logdir)
        sleep(1.0)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

    model.fit(
        np.zeros(shape=(1, *shape)),
        np.random.randint(0, 1, size=(1)),
        callbacks=[tensorboard_callback],
    )

    model.save(f"model_{backbone_name}.hdf5")


if __name__ == "__main__":
    main()
