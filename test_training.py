import pytest
from unittest import mock
import numpy as np
import pandas as pd

import training
class DummyFlow:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scale_pos_weight = None
        self.model = None
        self.metrics = None
        self.auc_threshold = 0.75
        self.cv_metrics = {}  # Needed for .output()
        self.feature_engineering = lambda: None  # Stub for .start()
        self.cross_validate = lambda: None       # Stub for .prepare()
        self.end = lambda: None                  # Stub for .output()

    def start(self):
        training.CreditRiskTrainingFlow.start(self)

    def prepare(self):
        training.CreditRiskTrainingFlow.prepare(self)

    def train(self):
        # Ensure .X exists for .train()
        self.X = pd.DataFrame(np.ones((8, 2)), columns=["a", "b"])
        training.CreditRiskTrainingFlow.train(self)

    def output(self):
        training.CreditRiskTrainingFlow.output(self)

def test_start_download(monkeypatch):
    flow = DummyFlow()
    # Mock os.path.exists to always return False (force download)
    monkeypatch.setattr("os.path.exists", lambda path: False)
    # Mock requests.get and zipfile.ZipFile
    monkeypatch.setattr("requests.get", lambda url: mock.Mock(content=b"data"))

    class DummyZip:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def extractall(self, path):
            pass

    monkeypatch.setattr("zipfile.ZipFile", lambda b: DummyZip())
    # Mock pd.read_excel to return a DataFrame
    df = pd.DataFrame({"A": [1, 2], "default payment next month": [0, 1]})
    monkeypatch.setattr("pandas.read_excel", lambda path, header=1: df)
    flow.next = lambda step: None  # Do nothing
    flow.start()
    assert hasattr(flow, "X") and hasattr(flow, "y")
    assert list(flow.y) == [0, 1]


def test_prepare(monkeypatch):
    flow = DummyFlow()
    # Provide dummy X and y
    flow.X = pd.DataFrame(np.ones((10, 2)), columns=["a", "b"])
    flow.y = pd.Series([0, 1] * 5)
    # Mock train_test_split
    monkeypatch.setattr(
        "sklearn.model_selection.train_test_split",
        lambda X, y, test_size, random_state, stratify: (
            X.iloc[:8],
            X.iloc[8:],
            pd.Series(y[:8]),
            pd.Series(y[8:]),
        ),
    )

    # Mock StandardScaler
    class DummyScaler:
        def fit_transform(self, X):
            return X

        def transform(self, X):
            return X

    monkeypatch.setattr("sklearn.preprocessing.StandardScaler", lambda: DummyScaler())
    flow.next = lambda step: None
    flow.prepare()
    assert hasattr(flow, "X_train") and hasattr(flow, "X_test")
    assert hasattr(flow, "scale_pos_weight")

def test_feature_engineering():
    flow = DummyFlow()
    # Set up dummy input DataFrame with necessary columns
    flow.X = pd.DataFrame({
        "PAY_0": [0, 1],
        "PAY_2": [0, 2],
        "PAY_3": [0, 2],
        "PAY_4": [0, 2],
        "PAY_5": [0, 2],
        "PAY_6": [0, 3],
        "BILL_AMT1": [100, 200],
        "BILL_AMT2": [100, 200],
        "BILL_AMT3": [100, 200],
        "BILL_AMT4": [100, 200],
        "BILL_AMT5": [100, 200],
        "BILL_AMT6": [100, 200],
        "PAY_AMT1": [50, 100],
        "PAY_AMT2": [50, 100],
        "PAY_AMT3": [50, 100],
        "PAY_AMT4": [50, 100],
        "PAY_AMT5": [50, 100],
        "PAY_AMT6": [50, 100],
        "LIMIT_BAL": [1000, 2000]
    })

    # Stub next
    flow.next = lambda step: None

    # Call actual method
    training.CreditRiskTrainingFlow.feature_engineering(flow)

    expected_features = [
        "PAY_MEAN", "PAY_TREND", "MAX_PAY_DELAY", "PAY_DELAY_VOLATILITY",
        "AVG_BILL_AMT", "AVG_PAY_AMT", "TOTAL_BILL_AMT", "TOTAL_PAY_AMT",
        "UTILIZATION_RATIO", "REMAINING_BALANCE", "BILL_VOLATILITY", "PAY_VOLATILITY"
    ]

    for feature in expected_features:
        assert feature in flow.X.columns

def test_cross_validate(monkeypatch):
    flow = DummyFlow()
    flow.X_train = np.random.rand(100, 5)
    flow.y_train = np.random.randint(0, 2, size=100)
    flow.scale_pos_weight = 1.0

    # Monkeypatch DMatrix and xgb.cv
    monkeypatch.setattr("xgboost.DMatrix", lambda X, label: mock.Mock())
    
    mock_cv_df = pd.DataFrame({
        "test-auc-mean": [0.7, 0.75, 0.8],
        "test-auc-std": [0.01, 0.02, 0.03],
        "test-logloss-mean": [0.5, 0.45, 0.4],
        "test-logloss-std": [0.02, 0.015, 0.01],
    })

    monkeypatch.setattr("xgboost.cv", lambda *args, **kwargs: mock_cv_df)

    flow.next = lambda step: None
    training.CreditRiskTrainingFlow.cross_validate(flow)

    assert isinstance(flow.cv_metrics, dict)
    assert round(flow.cv_metrics["cv_auc_mean"], 4) == 0.8
    assert round(flow.cv_metrics["cv_logloss_mean"], 4) == 0.4

def test_train(monkeypatch):
    flow = DummyFlow()
    flow.X = pd.DataFrame(np.ones((8, 2)), columns=["a", "b"])  # Needed for feature names
    flow.X_train = np.ones((8, 2))
    flow.X_test = np.ones((2, 2))
    flow.y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    flow.y_test = pd.Series([0, 1])
    flow.scale_pos_weight = 1.0

    monkeypatch.setattr("xgboost.DMatrix", lambda *args, **kwargs: (args, kwargs))

    class DummyModel:
        def save_model(self, path):
            self.saved_path = path

        def predict(self, dval):
            return np.array([0.6, 0.4])

        def get_score(self, importance_type="weight"):
            return {"a": 10, "b": 5}

        best_iteration = 42

    monkeypatch.setattr(
        "xgboost.train",
        lambda params, dtrain, num_boost_round, evals, early_stopping_rounds, verbose_eval: DummyModel(),
    )

    monkeypatch.setattr(training, "precision_score", lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(training, "recall_score", lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(training, "f1_score", lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(training, "roc_auc_score", lambda y_true, y_proba: 0.9)

    flow.next = lambda step: None
    flow.train()

    assert hasattr(flow, "model")
    assert flow.metrics["auc"] == 0.9

def test_output(monkeypatch, tmp_path):
    flow = DummyFlow()
    flow.metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": 0.9}
    flow.auc_threshold = 0.75
    flow.model = mock.Mock(best_iteration=10)
    flow.scale_pos_weight = 1.0
    # Patch OUTPUT_DIR to tmp_path
    monkeypatch.setattr(training, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(training, "USE_GPU", False)
    flow.next = lambda step: None
    flow.output()
    # Check that metrics.json and params.json were written
    metrics_path = tmp_path / "metrics.json"
    params_path = tmp_path / "params.json"
    assert metrics_path.exists()
    assert params_path.exists()
    with open(metrics_path) as f:
        metrics = f.read()
        assert "roc_auc" in metrics
    with open(params_path) as f:
        params = f.read()
        assert "best_iteration" in params
