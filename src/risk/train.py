import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from risk.features import build_features
from risk.validate import AS_OF, load_tables, validate

log = logging.getLogger(__name__)

# Features use everything up to this day. The label uses invoices issued after it.
CUTOFF = pd.Timestamp("2025-06-30")

NUM = [
    "credit_limit_inr", "credit_terms_days", "zero_credit_limit", "tenure_days",
    "n_invoices", "avg_days_late", "max_days_late", "share_paid_late",
    "n_partial", "share_partial", "n_written_off", "n_past_due", "max_days_past_due",
    "outstanding_inr", "past_due_inr", "utilisation",
    "n_orders", "n_orders_last_90d", "n_orders_prev_90d",
    "cancel_rate", "avg_order_value", "days_since_last_order", "no_history",
]
CAT = ["region", "channel"]


def label_distributors(payments, cutoff, as_of):
    """1 if an invoice issued after cutoff was still not fully paid by as_of, or was written off."""
    rows = payments[(payments["invoice_date"] > cutoff) & (payments["due_date"] <= as_of)].copy()
    settled = (
        rows["payment_date"].notna()
        & (rows["payment_date"] <= as_of)
        & (rows["amount_paid_inr"] >= rows["invoice_amount_inr"])
        & (rows["payment_status"] != "written_off")
    )
    rows["bad"] = ~settled
    labels = rows.groupby("distributor_id", as_index=False)["bad"].any()
    labels["high_risk"] = labels["bad"].astype(int)
    return labels[["distributor_id", "high_risk"]]


def training_table(data_dir, cutoff=CUTOFF, as_of=AS_OF):
    distributors, orders, payments = validate(*load_tables(data_dir))
    distributors = distributors[distributors["onboarded_date"] <= cutoff]
    features = build_features(distributors, orders, payments, as_of=cutoff)
    labels = label_distributors(payments, cutoff, as_of)
    table = features.merge(labels, on="distributor_id", how="inner")
    for col in ("zero_credit_limit", "no_history"):
        table[col] = table[col].astype(int)
    return table


def make_model():
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUM),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def choose_threshold(y, proba):
    # Missing a bad distributor costs more than flagging a good one,
    # so take the highest threshold that still catches 80% of the bad ones.
    chosen = 0.3
    best_recall = -1
    for step in range(20, 81):
        t = step / 100
        recall = recall_score(y, proba >= t, zero_division=0)
        if recall >= 0.8:
            chosen = t
        if recall > best_recall:
            best_recall = recall
            if recall < 0.8:
                chosen = t
    return chosen


def scores_at(y, proba, threshold):
    pred = proba >= threshold
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 3),
        "avg_precision": round(float(average_precision_score(y, proba)), 3),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "threshold": threshold,
        "n": int(len(y)),
        "n_high_risk": int(y.sum()),
    }


def train(data_dir="data", artifacts_dir="artifacts"):
    table = training_table(data_dir)
    X = table[NUM + CAT]
    y = table["high_risk"]
    log.info("training rows %d, high_risk %d", len(table), int(y.sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=7
    )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    train_proba = cross_val_predict(
        make_model(), X_train, y_train, cv=folds, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, train_proba)

    model = make_model()
    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    held_out = scores_at(y_test, test_proba, threshold)
    cv_scores = scores_at(y_train, train_proba, threshold)

    tree = Pipeline([
        ("prep", make_model().named_steps["prep"]),
        ("model", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=7)),
    ])
    tree.fit(X_train, y_train)
    tree_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    log.info("held-out logistic %s", held_out)
    log.info("held-out tree roc_auc %.3f", tree_auc)

    # refit on every labeled row for the model we save
    model.fit(X, y)
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = sorted(
        zip(names, model.named_steps["model"].coef_[0]),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )

    out = Path(artifacts_dir)
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {
        "cutoff": str(CUTOFF.date()),
        "as_of": str(AS_OF.date()),
        "label": (
            "high_risk = 1 when any invoice issued after the cutoff "
            "was written off, or was still not fully paid by the as-of date. "
            "Distributors with no invoice in that window are not in the training set."
        ),
        "why_logistic": (
            "Logistic regression over a depth-3 tree because the dataset is about 90 rows "
            "and the coefficients can be explained. The tree was only a comparison."
        ),
        "false_negative": (
            "A false negative approves a distributor who then does not pay, which loses the invoice. "
            "A false positive blocks a distributor who would have paid, which loses the sale. "
            "The unpaid invoice costs more, so the threshold is set to keep recall at 80% or better."
        ),
        "held_out": held_out,
        "cv_on_train": cv_scores,
        "tree_held_out_roc_auc": round(float(tree_auc), 3),
        "coefficients": [{ "feature": name, "coef": round(float(coef), 3)} for name, coef in coefs],
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    table.to_csv(out / "training_set.csv", index=False)
    log.info("saved model, meta, training_set to %s", out)
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()
