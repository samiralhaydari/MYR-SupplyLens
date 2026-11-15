"""Small NumPy-only prediction ensemble wrapper used by V2.4 inference."""
import numpy as np


class PredictionEnsemble:
    """Average same-shape predictions from independently trained members."""

    def __init__(self, models):
        self.models = list(models)
        if not self.models:
            raise ValueError("At least one model is required for an ensemble.")

    def predict(self, X, verbose=0):
        predictions = [
            np.asarray(model.predict(X, verbose=verbose), dtype=np.float64)
            for model in self.models
        ]
        shape = predictions[0].shape
        if any(prediction.shape != shape for prediction in predictions):
            raise ValueError("Ensemble members returned inconsistent prediction shapes.")
        stacked = np.stack(predictions, axis=0)
        result = np.mean(stacked, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Ensemble prediction contains non-finite values.")
        return result.astype(np.float32)
