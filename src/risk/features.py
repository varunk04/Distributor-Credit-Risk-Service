import logging

import pandas as pd

from risk.validate import AS_OF

log = logging.getLogger(__name__)


def build_features(distributors, orders, payments, as_of=AS_OF):
    as_of = pd.Timestamp(as_of)

    # only use what had happened by the as-of date
    orders = orders[orders["order_date"] <= as_of].copy()
    payments = payments[payments["invoice_date"] <= as_of].copy()

    # ---- invoices ----
    # a payment dated after as_of has not happened yet, so it counts as unpaid
    paid_by_now = payments["payment_date"] <= as_of
    payments["paid_so_far"] = payments["amount_paid_inr"].where(paid_by_now, 0)
    payments["paid_so_far"] = payments["paid_so_far"].clip(upper=payments["invoice_amount_inr"])

    # for written-off rows payment_date is the write-off date
    payments["is_written_off"] = (payments["payment_status"] == "written_off") & paid_by_now
    payments["is_partial"] = (payments["payment_status"] == "partial") & paid_by_now

    # money still owed, written-off money is closed so it is not exposure
    payments["outstanding"] = payments["invoice_amount_inr"] - payments["paid_so_far"]
    payments["outstanding"] = payments["outstanding"].where(~payments["is_written_off"], 0)

    payments["is_past_due"] = (payments["outstanding"] > 0) & (payments["due_date"] < as_of)
    payments["past_due_amount"] = payments["outstanding"].where(payments["is_past_due"], 0)
    payments["days_past_due"] = (as_of - payments["due_date"]).dt.days.where(payments["is_past_due"])

    # days late only for invoices that got some money, negative means early
    got_paid = paid_by_now & (payments["amount_paid_inr"] > 0) & ~payments["payment_before_invoice"]
    payments["days_late"] = (payments["payment_date"] - payments["due_date"]).dt.days.where(got_paid)
    payments["paid_late"] = (payments["days_late"] > 0).astype(float).where(payments["days_late"].notna())

    pay = payments.groupby("distributor_id").agg(
        n_invoices=("payment_id", "size"),
        avg_days_late=("days_late", "mean"),
        max_days_late=("days_late", "max"),
        share_paid_late=("paid_late", "mean"),
        n_partial=("is_partial", "sum"),
        n_written_off=("is_written_off", "sum"),
        n_past_due=("is_past_due", "sum"),
        max_days_past_due=("days_past_due", "max"),
        outstanding_inr=("outstanding", "sum"),
        past_due_inr=("past_due_amount", "sum"),
        first_invoice=("invoice_date", "min"),
    )

    # ---- orders ----
    orders["is_cancelled"] = orders["order_status"] == "cancelled"
    orders["value_if_not_cancelled"] = orders["order_value_inr"].where(~orders["is_cancelled"])
    orders["in_last_90d"] = orders["order_date"] > as_of - pd.Timedelta(days=90)
    orders["in_prev_90d"] = (
        (orders["order_date"] <= as_of - pd.Timedelta(days=90))
        & (orders["order_date"] > as_of - pd.Timedelta(days=180))
    )

    ords = orders.groupby("distributor_id").agg(
        n_orders=("order_id", "size"),
        n_orders_last_90d=("in_last_90d", "sum"),
        n_orders_prev_90d=("in_prev_90d", "sum"),
        cancel_rate=("is_cancelled", "mean"),
        avg_order_value=("value_if_not_cancelled", "mean"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    )
    ords["days_since_last_order"] = (as_of - ords["last_order"]).dt.days

    # ---- put it together, one row per distributor ----
    cols = [
        "distributor_id", "region", "channel", "credit_limit_inr",
        "credit_terms_days", "zero_credit_limit", "onboarded_date",
    ]
    feats = distributors[cols].set_index("distributor_id")
    feats = feats.join(pay).join(ords)

    # no history means zero counts, but rates and averages stay missing
    count_cols = [
        "n_invoices", "n_partial", "n_written_off", "n_past_due",
        "outstanding_inr", "past_due_inr",
        "n_orders", "n_orders_last_90d", "n_orders_prev_90d",
    ]
    feats[count_cols] = feats[count_cols].fillna(0)

    feats["no_history"] = (feats["n_orders"] + feats["n_invoices"]) == 0

    # exposure as a share of the limit, missing when the limit is 0
    feats["utilisation"] = feats["outstanding_inr"] / feats["credit_limit_inr"].where(
        feats["credit_limit_inr"] > 0
    )
    feats["share_partial"] = feats["n_partial"] / feats["n_invoices"].where(feats["n_invoices"] > 0)

    # onboarded_date is sometimes later than the first order or invoice
    first_seen = feats[["onboarded_date", "first_order", "first_invoice"]].min(axis=1)
    feats["tenure_days"] = (as_of - first_seen).dt.days

    feats = feats.drop(columns=["onboarded_date", "first_order", "first_invoice", "last_order"])

    log.info(
        "built %d feature rows as of %s, %d with no history",
        len(feats), as_of.date(), feats["no_history"].sum(),
    )
    return feats.reset_index()
