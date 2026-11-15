import logging
from pathlib import Path

from risk.features import build_features
from risk.validate import AS_OF, load_tables, validate


def run(data_dir="data", as_of=AS_OF):
    distributors, orders, payments = load_tables(data_dir)
    distributors, orders, payments = validate(distributors, orders, payments, as_of)
    return build_features(distributors, orders, payments, as_of)


if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/pipeline.log", mode="w")],
    )
    features = run()
    features.to_csv("artifacts/features.csv", index=False)
    logging.info("wrote %d rows to artifacts/features.csv", len(features))
