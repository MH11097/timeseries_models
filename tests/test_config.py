"""Tests for config loading."""

from src.utils.config import _deep_merge, load_config, make_param_slug


def test_load_base_config():
    config = load_config()
    assert "data" in config
    assert "split" in config
    assert "seed" in config


def test_load_model_config():
    config = load_config("arima")
    assert "model" in config
    assert config["model"]["name"] == "arima"


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 99
    assert result["b"]["d"] == 3
    assert result["e"] == 5


def test_overrides():
    config = load_config(overrides={"store_type": "c"})
    assert config["store_type"] == "c"


def test_make_param_slug_arima():
    config = {"model": {"name": "arima", "order": [1, 1, 1]}}
    assert make_param_slug(config) == "order_1-1-1"


def test_make_param_slug_xgboost():
    config = {"model": {"name": "xgboost", "max_depth": 7, "n_estimators": 1000, "learning_rate": 0.1}}
    assert make_param_slug(config) == "max_depth_7__n_estimators_1000__lr_0p1"


def test_make_param_slug_lstm():
    config = {"model": {"name": "lstm", "hidden_size": 128, "num_layers": 2}}
    assert make_param_slug(config) == "hidden_size_128__num_layers_2"


def test_make_param_slug_sarimax():
    config = {"model": {"name": "sarimax", "order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7]}}
    assert make_param_slug(config) == "order_1-1-1__seasonal_order_1-1-1-7"


def test_make_param_slug_unknown_model():
    config = {"model": {"name": "unknown_model"}}
    assert make_param_slug(config) == "default"


def test_make_param_slug_no_model_section():
    config = {}
    assert make_param_slug(config) == "default"
