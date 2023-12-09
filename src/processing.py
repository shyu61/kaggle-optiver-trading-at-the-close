from itertools import combinations

import polars as pl


def preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("target").is_not_nan() & pl.col("target").is_not_null())


def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    drop_cols = ["stock_id", "date_id", "row_id", "time_id"]
    df = df.with_columns(
        (
            (pl.col("bid_size") - pl.col("ask_size"))
            / (pl.col("bid_size") + pl.col("ask_size"))
        ).alias("imb_s1"),
        (
            (pl.col("imbalance_size") - pl.col("matched_size"))
            / (pl.col("imbalance_size") + pl.col("matched_size"))
        ).alias("imb_s2"),
    )

    prices = [
        "wap",
        "bid_price",
        "ask_price",
        "near_price",
        "far_price",
        "reference_price",
    ]

    for comb in combinations(prices, 2):
        df = df.with_columns(
            (
                (pl.col(comb[0]) - pl.col(comb[1]))
                / (pl.col(comb[0]) + pl.col(comb[1]))
            ).alias(f"{comb[0]}_{comb[1]}_imb")
        )

    for comb in combinations(prices, 3):
        _max = df.select(comb).max_horizontal()
        _min = df.select(comb).min_horizontal()
        _mid = df.select(comb).sum_horizontal() - _max - _min
        df = df.with_columns(
            ((_max - _mid) / (_mid - _min)).alias(f"{comb[0]}_{comb[1]}_{comb[2]}_imb2")
        )

    return df.drop(drop_cols)
