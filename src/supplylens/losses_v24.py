"""Explicit horizon-balanced objective; retains the existing output encoding."""
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="SupplyLensV24")
class HorizonBalancedHuber(keras.losses.Loss):
    def __init__(self, horizon=63, auxiliary_horizons=(5, 21, 42, 63),
                 target_multiplier=2.5, auxiliary_weight=2.5,
                 path_weight=1.0, blend=0.35, delta=1.0,
                 name="horizon_balanced_huber", **kwargs):
        super().__init__(name=name, **kwargs)
        self.horizon = int(horizon)
        self.auxiliary_horizons = list(auxiliary_horizons)
        self.target_multiplier = float(target_multiplier)
        self.auxiliary_weight = float(auxiliary_weight)
        self.path_weight = float(path_weight)
        self.blend = float(blend)
        self.delta = float(delta)
        if not self.auxiliary_horizons or self.target_multiplier <= 0 or self.delta <= 0:
            raise ValueError("Positive scales and at least one auxiliary horizon required.")
        if sorted(set(self.auxiliary_horizons)) != self.auxiliary_horizons:
            raise ValueError("Auxiliary horizons must be unique and increasing.")
        if min(self.auxiliary_horizons) < 1 or max(self.auxiliary_horizons) > self.horizon:
            raise ValueError("Auxiliary horizons must be within the forecast path.")
        if min(self.auxiliary_weight, self.path_weight) < 0 or not 0 <= self.blend <= 1:
            raise ValueError("Invalid loss weights or blend.")

    def call(self, y_true, y_pred):
        width = self.horizon + len(self.auxiliary_horizons)
        error = tf.reshape(y_pred - tf.cast(y_true, y_pred.dtype),
                           (tf.shape(y_pred)[0], -1, width))
        daily = error[..., :self.horizon]
        # Undo legacy target encoding BEFORE applying the explicit loss weight.
        auxiliary = error[..., self.horizon:] / self.target_multiplier
        cumulative = tf.gather(tf.cumsum(daily, axis=-1),
                               [h - 1 for h in self.auxiliary_horizons], axis=-1)
        cumulative /= tf.sqrt(tf.cast(self.auxiliary_horizons, y_pred.dtype))
        blended = (1 - self.blend) * cumulative + self.blend * auxiliary
        def huber(value):
            absolute = tf.abs(value)
            quadratic = tf.minimum(absolute, self.delta)
            return tf.reduce_mean(0.5 * quadratic ** 2 +
                                  self.delta * (absolute - quadratic), axis=[1, 2])
        return (huber(daily) + self.auxiliary_weight * huber(auxiliary) +
                self.path_weight * huber(blended)) / (1 + self.auxiliary_weight + self.path_weight)

    def get_config(self):
        return {**super().get_config(), **{key: getattr(self, key) for key in (
            "horizon", "auxiliary_horizons", "target_multiplier", "auxiliary_weight",
            "path_weight", "blend", "delta")}}
