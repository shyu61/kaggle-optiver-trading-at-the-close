from itertools import combinations

import polars as pl

CAST_DTYPES = {
    "stock_id": pl.UInt16,
    "date_id": pl.UInt16,
    "seconds_in_bucket": pl.UInt16,
    "imbalance_size": pl.Float32,
    "imbalance_buy_sell_flag": pl.Int8,
    "reference_price": pl.Float32,
    "matched_size": pl.Float32,
    "far_price": pl.Float32,
    "near_price": pl.Float32,
    "bid_price": pl.Float32,
    "bid_size": pl.Float32,
    "ask_price": pl.Float32,
    "ask_size": pl.Float32,
    "wap": pl.Float32,
    "target": pl.Float32,
    "time_id": pl.UInt32,
}


def preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    cast_dtypes = CAST_DTYPES.copy()
    if "target" in df.columns:  # train
        df = df.filter(pl.col("target").is_not_nan() & pl.col("target").is_not_null())
    else:  # inference
        cast_dtypes.pop("target")

    return df.with_columns(
        [pl.col(col).cast(dtype) for col, dtype in cast_dtypes.items()]
    )


# fmt: off
STOCK_WEIGHTS = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006,
    0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004, 0.002, 0.002,
    0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004, 0.004, 0.004, 0.006,
    0.002, 0.002, 0.04, 0.002, 0.002, 0.004, 0.04, 0.002, 0.001, 0.006, 0.004, 0.004, 0.006,
    0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004, 0.006, 0.004, 0.002, 0.001, 0.002,
    0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004, 0.006, 0.002, 0.004, 0.004, 0.002, 0.004,
    0.004, 0.004, 0.001, 0.002, 0.002, 0.008, 0.02, 0.004, 0.006, 0.002, 0.02, 0.002, 0.002,
    0.006, 0.004, 0.002, 0.001, 0.02, 0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006,
    0.004, 0.006, 0.001, 0.002, 0.004, 0.006, 0.006, 0.001, 0.04, 0.006, 0.002, 0.004, 0.002,
    0.002, 0.006, 0.002, 0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002,
    0.006, 0.002, 0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008,
    0.002, 0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002, 0.04,
    0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02, 0.004, 0.002, 0.006, 0.02, 0.001, 0.002,
    0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04, 0.002, 0.008, 0.002,
    0.004, 0.001, 0.004, 0.006, 0.004
]
# fmt: on


def __add_index_wap(df: pl.DataFrame) -> pl.DataFrame:
    """
    index_wapはtime_idごとに、各銘柄のwapに重みSTOCK_WEIGHTSをかけたものの和で算出する
    """
    stock_weights_df = pl.DataFrame(
        {
            "stock_id": pl.arange(0, len(STOCK_WEIGHTS), eager=True).cast(pl.UInt16),
            "weight": pl.Series(STOCK_WEIGHTS).cast(pl.Float32),
        }
    )
    df = df.join(stock_weights_df, on="stock_id")
    index_wap_df = df.group_by("time_id").agg(
        (pl.col("wap") * pl.col("weight")).sum().alias("index_wap")
    )
    df = df.join(index_wap_df, on="time_id").sort("stock_id", "time_id").drop("weight")
    return df


def feature_engineering(
    df: pl.DataFrame, maintain_stock_id: bool = False
) -> pl.DataFrame:
    drop_cols = ["date_id", "row_id", "time_id"]
    if not maintain_stock_id:
        drop_cols.append("stock_id")

    df = __add_index_wap(df)

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
