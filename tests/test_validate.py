import pandas as pd

from risk.features import build_features
from risk.validate import load_tables, validate


def test_load_and_validate_counts():
    distributors, orders, payments = load_tables("data")
    assert len(distributors) == 120
    assert len(orders) == 1819
    assert len(payments) == 1271

    distributors, orders, payments = validate(distributors, orders, payments)
    assert len(payments) == 1266
    assert not payments["payment_id"].astype(str).str.contains("TRAP").any()
    assert int(orders["order_before_onboarding"].sum()) == 135
    assert list(distributors.loc[distributors["zero_credit_limit"], "distributor_id"]) == ["DIST-015"]


def test_bad_payment_is_dropped_and_the_rest_survives():
    distributors, orders, payments = load_tables("data")
    extra = payments.iloc[[0]].copy()
    extra["payment_id"] = "PAY-BAD"
    extra["distributor_id"] = "DIST-999"
    extra["invoice_amount_inr"] = -50
    payments = pd.concat([payments, extra], ignore_index=True)

    _, _, cleaned = validate(distributors, orders, payments)
    assert "PAY-BAD" not in set(cleaned["payment_id"])
    assert len(cleaned) == 1266


def test_features_one_row_per_distributor():
    distributors, orders, payments = validate(*load_tables("data"))
    features = build_features(distributors, orders, payments)
    assert len(features) == 120
    assert features["distributor_id"].is_unique

    empty = features[features["distributor_id"] == "DIST-119"].iloc[0]
    assert bool(empty["no_history"])
    assert pd.isna(features.loc[features["distributor_id"] == "DIST-015", "utilisation"]).all()
