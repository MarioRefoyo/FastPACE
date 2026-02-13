from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import silhouette_samples
import torch


class OutlierCalculator(ABC):
    def __init__(self, model, calibration_data, data_format):
        assert data_format in ["torch", "tf"]
        self.model = model
        self.data_format = data_format
        if data_format == "tf":
            self.length = calibration_data.shape[1]
            self.n_channels = calibration_data.shape[2]
        elif data_format == "torch":
            self.length = calibration_data.shape[2]
            self.n_channels = calibration_data.shape[1]

    @abstractmethod
    def _get_raw_outlier_scores(self, data):
        pass

    def _scale_score(self, data):
        scaled_scores = (data - self.min_score) / (self.max_score - self.min_score)
        return scaled_scores

    def _clip_score(self, data):
        scaled_scores = np.clip(data, a_min=self.min_score, a_max=self.max_score)
        return scaled_scores

    def get_outlier_scores(self, data):
        data_reconstruction_errors = self._get_raw_outlier_scores(data)
        scores = self._scale_score(data_reconstruction_errors).flatten()
        # scores = self._clip_score(scores)
        return scores


class AEOutlierCalculator(OutlierCalculator):
    def __init__(self, model, calibration_data, backend="tf", data_format="tf"):
        assert backend in ["tf", "torch"]

        super().__init__(model, calibration_data, data_format)
        self.backend = backend
        if backend == "torch":
            # Trace it with JIT
            example_input = np.random.rand(1, calibration_data.shape[1], calibration_data.shape[2]).astype(np.float32)
            self.compile_with_jit(example_input)

        # Calibrate to get outlier score as a number between 0 and 1
        calibration_scores = self._get_raw_outlier_scores(calibration_data)
        self.min_score = min(0, calibration_scores.min())
        self.max_score = calibration_scores.max()

    def _get_raw_outlier_scores(self, data):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        if self.backend != self.data_format:
            data = np.transpose(data, (0, 2, 1))
        if self.backend == "tf":
            data_reconstruction = self.model.predict(data, verbose=0)
        else:
            data_tensor = torch.from_numpy(data).float().to(self.device)
            data_reconstruction = self.model(data_tensor)
            data_reconstruction = data_reconstruction.detach().cpu().numpy()
        reconstruction_errors = np.mean(np.abs(data - data_reconstruction), axis=(1, 2))
        return reconstruction_errors
    
    def compile_with_jit(self, example_input: np.ndarray):
        device = next(self.model.parameters()).device
        self.device = device
        if self.backend != "torch":
            raise RuntimeError("JIT compilation is only available for PyTorch models.")

        # Prepare example input for tracing
        if len(example_input.shape) == 2:
            example_input = np.expand_dims(example_input, axis=0)
        if self.backend != self.data_format:
            example_input = np.transpose(example_input, (0, 2, 1))

        example_tensor = torch.from_numpy(example_input).float().to(device)

        # Trace the model and replace the original
        with torch.inference_mode():
            traced = torch.jit.trace(self.model, example_tensor)
            traced = torch.jit.freeze(traced.eval())
            self.base_model = self.model
            self.model = traced
            self.model = torch.jit.optimize_for_inference(self.model)


class IFOutlierCalculator(OutlierCalculator):

    def __init__(self, model, calibration_data, data_format):
        assert data_format in ["tf"]
        super().__init__(model, calibration_data, data_format)

        # Calibrate to get outlier score as a number between 0 and 1
        calibration_scores = self._get_raw_outlier_scores(calibration_data)
        self.min_score = calibration_scores.min()
        self.max_score = calibration_scores.max()

    def _get_raw_outlier_scores(self, data):
        data_flat = data.reshape(-1, self.length*self.n_channels)        
        outlier_score = self.model.score_samples(data_flat)
        # Multiply by minus one as lower is more anomaly
        outlier_score = -1 * outlier_score
        return outlier_score


class LOFOutlierCalculator(OutlierCalculator):
    def __init__(self, model, calibration_data, data_format):
        assert data_format in ["tf"]
        super().__init__(model, calibration_data, data_format)

        # Calibrate to get outlier score as a number between 0 and 1
        calibration_scores = self._get_raw_outlier_scores(calibration_data)
        self.min_score = calibration_scores.min()
        self.max_score = calibration_scores.max()

    def _get_raw_outlier_scores(self, data):
        data_flat = data.reshape(-1, self.length*self.n_channels)
        outlier_score = self.model.score_samples(data_flat)
        # Multiply by minus one as lower is more anomaly
        outlier_score = -1 * outlier_score
        return outlier_score
