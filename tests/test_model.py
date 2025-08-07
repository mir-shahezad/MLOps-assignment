import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.train import load_and_preprocess,train_model


@pytest.fixture(scope="module")
def data():
    return load_and_preprocess(data_path="../data/iris_processed.csv")

def test_data_split(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

@pytest.mark.parametrize("model_class", [LogisticRegression, RandomForestClassifier])
def test_model_training_accuracy(data, model_class):
    X_train, X_test, y_train, y_test = data
    model = model_class()
    acc, trained_model = train_model(model, model_class.__name__, X_train, X_test, y_train, y_test)
    assert 0.5 <= acc <= 1.0  # iris should usually give >90%
    y_pred = trained_model.predict(X_test)
    assert len(y_pred) == len(y_test)
    assert accuracy_score(y_test, y_pred) == acc