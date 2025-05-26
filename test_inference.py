import pytest
from unittest import mock
import numpy as np
import inference


# Mock model class for predict_default
def mock_model_predict(return_values):
    class MockModel:
        def predict(self, dmatrix):
            return np.array(return_values)

    return MockModel()


def test_load_model_file_not_found(monkeypatch):
    # Simulate missing model file
    monkeypatch.setattr("os.path.exists", lambda path: False)
    with pytest.raises(FileNotFoundError):
        inference.load_model()


def test_load_model_success(monkeypatch):
    # Simulate model file exists and mock xgb.Booster
    monkeypatch.setattr("os.path.exists", lambda path: True)
    mock_booster = mock.MagicMock()
    monkeypatch.setattr("xgboost.Booster", lambda: mock_booster)
    monkeypatch.setattr(mock_booster, "load_model", lambda path: None)
    result = inference.load_model()
    assert result == mock_booster


def test_predict_default_basic(monkeypatch):
    # Prepare input data
    input_data = [[1] * len(inference.FEATURE_NAMES)]

    # Mock StandardScaler to just return the input unchanged
    class DummyScaler:
        def fit_transform(self, X):
            return X

    monkeypatch.setattr("sklearn.preprocessing.StandardScaler", lambda: DummyScaler())

    # Mock xgb.DMatrix to just return the input
    monkeypatch.setattr("xgboost.DMatrix", lambda X: X)

    # Use a mock model that returns fixed probabilities
    model = mock_model_predict([0.8, 0.2])
    # Duplicate input_data to match output size
    input_data = [input_data[0], input_data[0]]
    results = inference.predict_default(model, input_data)
    assert results[0]["probability"] == 0.8
    assert results[0]["prediction"] == "DEFAULT"
    assert results[0]["risk_level"] == "HIGH"
    assert results[1]["probability"] == 0.2
    assert results[1]["prediction"] == "NO DEFAULT"
    assert results[1]["risk_level"] == "LOW"
