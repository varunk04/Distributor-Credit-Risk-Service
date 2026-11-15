import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

AS_OF = pd.Timestamp("2026-06-09")


def load_tables(data_dir):
    data_dir = Path(data_dir)
    distributors = pd.read_csv(data_dir / "distributors.csv")
    orders = pd.read_csv(data_dir / "orders.csv")
    payments = pd.read_csv(data_dir / "payments.csv")

    distributors["onboarded_date"] = pd.to_datetime(
        distributors["onboarded_date"], format="%Y-%m-%d", errors="coerce"
    )
    orders["order_date"] = pd.to_datetime(
        orders["order_date"], format="%Y-%m-%d", errors="coerce"
    )
    for col in ["invoice_date", "due_date", "payment_date"]:
        payments[col] = pd.to_datetime(payments[col], format="%Y-%m-%d", errors="coerce")

    return distributors, orders, payments


def validate(distributors, orders, payments, as_of=AS_OF):
    log.info(
        "validating %d distributors, %d orders, %d payments",
        len(distributors), len(orders), len(payments),
    )

    # duplicate ids
    n = len(distributors)
    distributors = distributors.drop_duplicates("distributor_id")
    if len(distributors) < n:
        log.warning("dropped %d duplicate distributor ids", n - len(distributors))

    n = len(orders)
    orders = orders.drop_duplicates("order_id")
    if len(orders) < n:
        log.warning("dropped %d duplicate order ids", n - len(orders))

    n = len(payments)
    payments = payments.drop_duplicates("payment_id")
    if len(payments) < n:
        log.warning("dropped %d duplicate payment ids", n - len(payments))

    # dates that did not parse (payment_date is allowed to be blank)
    bad = distributors["onboarded_date"].isna()
    if bad.any():
        log.warning("dropped %d distributors with a bad onboarded_date", bad.sum())
        distributors = distributors[~bad]

    bad = orders["order_date"].isna()
    if bad.any():
        log.warning("dropped %d orders with a bad order_date", bad.sum())
        orders = orders[~bad]

    bad = payments["invoice_date"].isna() | payments["due_date"].isna()
    if bad.any():
        log.warning("dropped %d payments with a bad invoice_date or due_date", bad.sum())
        payments = payments[~bad]

    # amounts that make no sense
    bad = (
        payments["invoice_amount_inr"].isna()
        | (payments["invoice_amount_inr"] <= 0)
        | payments["amount_paid_inr"].isna()
        | (payments["amount_paid_inr"] < 0)
    )
    if bad.any():
        log.warning("dropped %d payments with a bad invoice or paid amount", bad.sum())
        payments = payments[~bad]

    bad = orders["order_value_inr"] <= 0
    if bad.any():
        log.warning("set %d zero or negative order values to missing", bad.sum())
        orders["order_value_inr"] = orders["order_value_inr"].where(~bad)

    # orphans: no distributor to attach them to
    known = set(distributors["distributor_id"])

    bad = ~orders["distributor_id"].isin(known)
    if bad.any():
        log.warning(
            "dropped %d orders with an unknown distributor: %s",
            bad.sum(), list(orders.loc[bad, "order_id"]),
        )
        orders = orders[~bad]

    bad = ~payments["distributor_id"].isin(known)
    if bad.any():
        log.warning(
            "dropped %d payments with an unknown distributor: %s",
            bad.sum(), list(payments.loc[bad, "payment_id"]),
        )
        payments = payments[~bad]

    # from here on nothing is dropped, rows are only flagged
    distributors = distributors.copy()
    orders = orders.copy()
    payments = payments.copy()

    distributors["zero_credit_limit"] = distributors["credit_limit_inr"] <= 0
    log.info(
        "%d distributors with a zero credit limit: %s",
        distributors["zero_credit_limit"].sum(),
        list(distributors.loc[distributors["zero_credit_limit"], "distributor_id"]),
    )

    onboarded = distributors.set_index("distributor_id")["onboarded_date"]
    terms = distributors.set_index("distributor_id")["credit_terms_days"]

    orders["null_order_value"] = orders["order_value_inr"].isna()
    log.info("%d orders have no value", orders["null_order_value"].sum())

    orders["order_before_onboarding"] = orders["order_date"] < orders["distributor_id"].map(onboarded)
    log.info("%d orders are dated before onboarding", orders["order_before_onboarding"].sum())

    payments["invoice_before_onboarding"] = (
        payments["invoice_date"] < payments["distributor_id"].map(onboarded)
    )
    log.info("%d invoices are dated before onboarding", payments["invoice_before_onboarding"].sum())

    payments["payment_before_invoice"] = payments["payment_date"] < payments["invoice_date"]
    log.info("%d payments are dated before their invoice", payments["payment_before_invoice"].sum())

    payments["overpaid"] = payments["amount_paid_inr"] > payments["invoice_amount_inr"]
    log.info("%d invoices were overpaid", payments["overpaid"].sum())

    gap = (payments["due_date"] - payments["invoice_date"]).dt.days
    payments["terms_mismatch"] = gap != payments["distributor_id"].map(terms)
    log.info(
        "%d invoices have a due date that does not match the distributor's terms: %s",
        payments["terms_mismatch"].sum(),
        sorted(payments.loc[payments["terms_mismatch"], "distributor_id"].unique()),
    )

    payments["payment_after_as_of"] = payments["payment_date"] > as_of
    log.info(
        "%d payments are dated after %s, they will count as unpaid",
        payments["payment_after_as_of"].sum(), as_of.date(),
    )

    seen = set(orders["distributor_id"]) | set(payments["distributor_id"])
    no_history = sorted(known - seen)
    log.info("%d distributors have no orders and no payments: %s", len(no_history), no_history)

    log.info(
        "done: %d distributors, %d orders, %d payments kept",
        len(distributors), len(orders), len(payments),
    )
    return distributors, orders, payments
