import pytest
from unittest import mock
import numpy as np
import pandas as pd

import training

class DummyFlow(training.CreditRiskTrainingFlow):
    def __init__(self):
        # Don't call FlowSpec.__init__
        pass

def test_start_download(monkeypatch):
    flow = DummyFlow()
    # Mock os.path.exists to always return False (force download)
    monkeypatch.setattr('os.path.exists', lambda path: False)
    # Mock requests.get and zipfile.ZipFile
    monkeypatch.setattr('requests.get', lambda url: mock.Mock(content=b'data'))
    class DummyZip:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def extractall(self, path): pass
    monkeypatch.setattr('zipfile.ZipFile', lambda b: DummyZip())
    # Mock pd.read_excel to return a DataFrame
    df = pd.DataFrame({"A": [1,2], "default payment next month": [0,1]})
    monkeypatch.setattr('pandas.read_excel', lambda path, header=1: df)
    flow.next = lambda step: None  # Do nothing
    flow.start()
    assert hasattr(flow, 'X') and hasattr(flow, 'y')
    assert list(flow.y) == [0,1]

def test_prepare(monkeypatch):
    flow = DummyFlow()
    # Provide dummy X and y
    flow.X = pd.DataFrame(np.ones((10, 2)), columns=["a", "b"])
    flow.y = pd.Series([0,1]*5)
    # Mock train_test_split
    monkeypatch.setattr('sklearn.model_selection.train_test_split', lambda X, y, test_size, random_state, stratify: (X.iloc[:8], X.iloc[8:], pd.Series(y[:8]), pd.Series(y[8:])))
    # Mock StandardScaler
    class DummyScaler:
        def fit_transform(self, X): return X
        def transform(self, X): return X
    monkeypatch.setattr('sklearn.preprocessing.StandardScaler', lambda: DummyScaler())
    flow.next = lambda step: None
    flow.prepare()
    assert hasattr(flow, 'X_train') and hasattr(flow, 'X_test')
    assert hasattr(flow, 'scale_pos_weight')

def test_train(monkeypatch):
    flow = DummyFlow()
    flow.X_train = np.ones((8,2))
    flow.X_test = np.ones((2,2))
    flow.y_train = pd.Series([0,1,0,1,0,1,0,1])
    flow.y_test = pd.Series([0,1])
    flow.scale_pos_weight = 1.0
    # Mock xgb.DMatrix
    monkeypatch.setattr('xgboost.DMatrix', lambda X, label=None: (X,label))
    # Mock xgb.train to return a mock model
    class DummyModel:
        def save_model(self, path): self.saved_path = path
        def predict(self, dval): return np.array([0.6, 0.4])
        best_iteration = 42
    monkeypatch.setattr('xgboost.train', lambda params, dtrain, num_boost_round, evals, early_stopping_rounds, verbose_eval: DummyModel())
    # Mock metrics in the training module namespace
    monkeypatch.setattr(training, 'precision_score', lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(training, 'recall_score', lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(training, 'f1_score', lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(training, 'roc_auc_score', lambda y_true, y_proba: 0.9)
    flow.next = lambda step: None
    flow.train()
    assert hasattr(flow, 'model')
    assert flow.metrics['auc'] == 0.9

def test_output(monkeypatch, tmp_path):
    flow = DummyFlow()
    flow.metrics = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'auc': 0.9}
    flow.auc_threshold = 0.75
    flow.model = mock.Mock(best_iteration=10)
    flow.scale_pos_weight = 1.0
    # Patch OUTPUT_DIR to tmp_path
    monkeypatch.setattr(training, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(training, 'USE_GPU', False)
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
