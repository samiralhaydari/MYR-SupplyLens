"""V2.4 attention-based multi-horizon joint CNN-LSTM architecture."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import DISPLAY_HORIZONS, V24_AUXILIARY_LOSS_WEIGHT


@keras.utils.register_keras_serializable(package="SupplyLensV24")
class MultiHeadTemporalAttentionPooling(layers.Layer):
    """Pool the LSTM sequence through several independently learned time-attention heads."""

    def __init__(self, num_heads=4, attention_units=24, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.attention_units = int(attention_units)

    def build(self, input_shape):
        feature_count = int(input_shape[-1])
        self.attention_kernel = self.add_weight(
            name="attention_kernel",
            shape=(feature_count, self.num_heads, self.attention_units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.attention_bias = self.add_weight(
            name="attention_bias",
            shape=(self.num_heads, self.attention_units),
            initializer="zeros",
            trainable=True,
        )
        self.context_vector = self.add_weight(
            name="context_vector",
            shape=(self.num_heads, self.attention_units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        projected = tf.tanh(
            tf.einsum("bti,ihd->bthd", inputs, self.attention_kernel)
            + self.attention_bias
        )
        scores = tf.squeeze(
            tf.einsum("bthd,hdo->btho", projected, self.context_vector),
            axis=-1,
        )
        weights = tf.nn.softmax(scores, axis=1)
        contexts = tf.einsum("bth,bti->bhi", weights, inputs)
        return tf.reshape(
            contexts, (-1, self.num_heads * int(inputs.shape[-1]))
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_heads * int(input_shape[-1]))

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "attention_units": self.attention_units,
        })
        return config


@keras.utils.register_keras_serializable(package="SupplyLensV24")
class CurrencySpecificHeads(layers.Layer):
    """Independent compact decoder head for each currency.

    Each currency predicts a 63-step daily return path plus direct cumulative
    return targets at the configured evaluation horizons.
    """

    def __init__(
        self,
        n_currencies=7,
        horizon=63,
        auxiliary_horizons=None,
        hidden_units=96,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_currencies = int(n_currencies)
        self.horizon = int(horizon)
        self.auxiliary_horizons = [
            int(h) for h in (auxiliary_horizons or DISPLAY_HORIZONS)
            if int(h) <= self.horizon
        ]
        self.hidden_units = int(hidden_units)
        self.output_width = self.horizon + len(self.auxiliary_horizons)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.hidden_kernel = self.add_weight(
            name="hidden_kernel",
            shape=(self.n_currencies, input_dim, self.hidden_units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.hidden_bias = self.add_weight(
            name="hidden_bias",
            shape=(self.n_currencies, self.hidden_units),
            initializer="zeros",
            trainable=True,
        )
        self.output_kernel = self.add_weight(
            name="output_kernel",
            shape=(self.n_currencies, self.hidden_units, self.output_width),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.output_bias = self.add_weight(
            name="output_bias",
            shape=(self.n_currencies, self.output_width),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        hidden = tf.einsum("bi,nih->bnh", inputs, self.hidden_kernel)
        hidden = tf.nn.gelu(hidden + self.hidden_bias)
        outputs = tf.einsum("bnh,nho->bno", hidden, self.output_kernel)
        outputs = outputs + self.output_bias
        return tf.reshape(
            outputs, (-1, self.n_currencies * self.output_width)
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_currencies * self.output_width)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_currencies": self.n_currencies,
            "horizon": self.horizon,
            "auxiliary_horizons": self.auxiliary_horizons,
            "hidden_units": self.hidden_units,
        })
        return config


@keras.utils.register_keras_serializable(package="SupplyLensV24")
class ResidualDilatedBlock(layers.Layer):
    """Two causal convolutions with a residual path and per-step normalization."""
    def __init__(self, kernel_size=3, dilation_rate=1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.dilation_rate = int(dilation_rate)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        width = int(input_shape[-1])
        self.kernel1 = self.add_weight(name="kernel1", shape=(self.kernel_size, width, width), initializer="glorot_uniform")
        self.bias1 = self.add_weight(name="bias1", shape=(width,), initializer="zeros")
        self.kernel2 = self.add_weight(name="kernel2", shape=(self.kernel_size, width, width), initializer="glorot_uniform")
        self.bias2 = self.add_weight(name="bias2", shape=(width,), initializer="zeros")
        self.gamma = self.add_weight(name="gamma", shape=(width,), initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(width,), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs):
        def conv(x, kernel, bias):
            pad = self.dilation_rate * (self.kernel_size - 1)
            x = tf.pad(x, [[0, 0], [pad, 0], [0, 0]])
            return tf.nn.convolution(x, kernel, padding="VALID", dilations=[self.dilation_rate]) + bias
        hidden = tf.nn.gelu(conv(inputs, self.kernel1, self.bias1))
        hidden = inputs + conv(hidden, self.kernel2, self.bias2)
        mean, variance = tf.nn.moments(hidden, axes=[-1], keepdims=True)
        return tf.nn.gelu((hidden - mean) * tf.math.rsqrt(variance + self.epsilon) * self.gamma + self.beta)

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate, "epsilon": self.epsilon}


def build_joint_cnn_lstm_v24(
    lookback,
    n_input_features,
    n_currencies,
    forecast_horizon=63,
    auxiliary_horizons=None,
    architecture="residual",
):
    """Build the V2.4 multi-scale CNN-LSTM with attention and currency heads."""
    auxiliary_horizons = [
        int(h) for h in (auxiliary_horizons or DISPLAY_HORIZONS)
        if int(h) <= int(forecast_horizon)
    ]
    if architecture not in {"residual", "original"}:
        raise ValueError("architecture must be residual or original")
    model = keras.Sequential(
        [
            keras.Input(shape=(int(lookback), int(n_input_features))),
            # Multi-scale causal convolutions: wider kernels and dilation rates
            # expand the receptive field without looking beyond the input window.
            layers.Conv1D(
                96, 3, padding="causal", activation="gelu",
            ),
            *([ResidualDilatedBlock(5, 2), ResidualDilatedBlock(9, 3),
               ResidualDilatedBlock(3, 8)] if architecture == "residual" else [
                layers.Conv1D(96, 5, padding="causal", dilation_rate=2, activation="gelu"),
                layers.Conv1D(96, 9, padding="causal", dilation_rate=3, activation="gelu"),
            ]),
            layers.LSTM(
                96,
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=0.0,
                recurrent_dropout=0.0,
            ),
            MultiHeadTemporalAttentionPooling(
                num_heads=4,
                attention_units=24,
                name="multi_head_temporal_attention_pooling",
            ),
            layers.Dropout(0.25),
            layers.Dense(192, activation="gelu"),
            layers.Dropout(0.20),
            layers.Dense(128, activation="gelu"),
            CurrencySpecificHeads(
                n_currencies=int(n_currencies),
                horizon=int(forecast_horizon),
                auxiliary_horizons=auxiliary_horizons,
                hidden_units=96,
                name="currency_specific_heads",
            ),
        ],
        name="MYR_SupplyLens_CNN_LSTM_V2_4",
    )
    return model


