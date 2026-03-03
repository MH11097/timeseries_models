"""Tests for data cleaning functions."""

import numpy as np
import pandas as pd

from src.data.cleaner import (
    add_competition_days,
    add_holiday_proximity,
    add_log_sales,
    add_promo_month,
    add_store_stats,
    fill_missing,
    fix_types,
    merge_store,
    remove_outliers,
    validate,
)


def _make_train(n=6):
    """Minimal train-like DataFrame."""
    return pd.DataFrame({
        "Store": [1] * n,
        "Date": pd.date_range("2015-01-01", periods=n, freq="D"),
        "Sales": [5000, 6000, 0, 7000, 8000, 30000],
        "Customers": [300, 400, 0, 500, 600, 700],
        "Open": [1, 1, 0, 1, 1, 1],
        "Promo": [1, 0, 0, 1, 0, 1],
        "StateHoliday": ["0", "a", "0", "0", "b", "0"],
        "SchoolHoliday": [0, 0, 1, 0, 0, 1],
    })


def _make_store():
    """Minimal store-like DataFrame."""
    return pd.DataFrame({
        "Store": [1],
        "StoreType": ["a"],
        "Assortment": ["c"],
        "CompetitionDistance": [np.nan],
        "CompetitionOpenSinceMonth": [6.0],
        "CompetitionOpenSinceYear": [2014.0],
        "Promo2": [1],
        "Promo2SinceWeek": [10.0],
        "Promo2SinceYear": [2013.0],
        "PromoInterval": ["Jan,Apr,Jul,Oct"],
    })


def test_fill_missing():
    store = pd.concat([_make_store(), _make_store()], ignore_index=True)
    store.loc[0, "CompetitionDistance"] = 500.0
    store.loc[1, "CompetitionDistance"] = np.nan
    result = fill_missing(store)
    assert result["CompetitionDistance"].isnull().sum() == 0
    assert result.loc[1, "CompetitionDistance"] == 500.0  # median of [500]
    assert result["Promo2SinceWeek"].dtype == int


def test_merge_store():
    train = _make_train()
    store = _make_store()
    merged = merge_store(train, store)
    assert "StoreType" in merged.columns
    assert len(merged) == len(train)


def test_remove_outliers():
    train = _make_train()
    result = remove_outliers(train)
    assert (result["Open"] == 1).all()
    assert (result["Sales"] > 0).all()
    assert result["Sales"].max() <= 25000


def test_fix_types():
    train = _make_train()
    store = _make_store()
    merged = merge_store(train, store)
    result = fix_types(merged)
    assert result["StateHoliday"].dtype in [int, np.int64]
    assert result["StoreType"].dtype in [int, np.int64]
    assert result["Assortment"].dtype in [int, np.int64]


def test_add_holiday_proximity():
    train = _make_train()
    train = fix_types(train)
    result = add_holiday_proximity(train)
    assert "AfterStateHoliday" in result.columns
    assert "BeforeStateHoliday" in result.columns
    assert "AfterSchoolHoliday" in result.columns
    assert "BeforeSchoolHoliday" in result.columns


def test_add_store_stats():
    train = _make_train()
    result = add_store_stats(train)
    assert "SalesPerDay" in result.columns
    assert "CustomersPerDay" in result.columns
    assert result["SalesPerDay"].notnull().all()


def test_add_promo_month():
    train = _make_train()
    store = _make_store()
    merged = merge_store(train, store)
    result = add_promo_month(merged)
    assert "IsPromoMonth" in result.columns
    # Jan 1-6, PromoInterval includes Jan → should be 1 for Promo2=1
    assert result["IsPromoMonth"].iloc[0] == 1


def test_add_competition_days():
    train = _make_train()
    store = _make_store()
    merged = merge_store(train, store)
    merged = fill_missing(merged)
    result = add_competition_days(merged)
    assert "CompetitionDaysOpen" in result.columns
    assert (result["CompetitionDaysOpen"] >= 0).all()
    assert (result["CompetitionDaysOpen"] <= 3 * 365).all()


def test_add_log_sales():
    train = _make_train()
    result = add_log_sales(train)
    assert "LogSales" in result.columns
    assert np.isclose(result["LogSales"].iloc[0], np.log1p(5000))


def test_validate():
    train = _make_train()
    report = validate(train)
    assert report["rows"] == 6
    assert report["stores"] == 1
    assert "sales_min" in report
