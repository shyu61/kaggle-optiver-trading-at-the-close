from itertools import combinations
from typing import List

import numpy as np
import polars as pl

# import talib

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

PRICES = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
SIZES = ["matched_size", "bid_size", "ask_size", "imbalance_size"]


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
            "stock_weights": pl.Series(STOCK_WEIGHTS).cast(pl.Float32),
        }
    )
    df = df.join(stock_weights_df, on="stock_id")
    # index_wap_df = df.group_by("time_id").agg(
    #     (pl.col("wap") * pl.col("stock_weights")).sum().alias("index_wap")
    # )
    # df = df.join(index_wap_df, on="time_id").sort("stock_id", "time_id")
    df = df.with_columns(
        (pl.col("wap") * pl.col("stock_weights")).alias("weighted_wap"),
    )
    df = df.with_columns(
        pl.col("weighted_wap").pct_change(6).over("stock_id").alias("wap_momentum"),
    )
    return df


def __add_rolling(df: pl.DataFrame) -> pl.DataFrame:
    # fmt: off
    for col in ["bid_price", "ask_price", "bid_size", "ask_size"]:
        df = df.with_columns(
            pl.col(f"{col}_diff_3").rolling_mean(3).over("stock_id").alias(f"rolling_diff_{col}_3"),
            pl.col(f"{col}_diff_5").rolling_mean(5).over("stock_id").alias(f"rolling_diff_{col}_5"),
            pl.col(f"{col}_diff_10").rolling_mean(10).over("stock_id").alias(f"rolling_diff_{col}_10"),  # noqa

            pl.col(f"{col}_diff_3").rolling_std(3).over("stock_id").alias(f"rolling_std_diff_{col}_3"),  # noqa
            pl.col(f"{col}_diff_5").rolling_std(5).over("stock_id").alias(f"rolling_std_diff_{col}_5"),  # noqa
            pl.col(f"{col}_diff_10").rolling_std(10).over("stock_id").alias(f"rolling_std_diff_{col}_10"),  # noqa
        )

    # df = df.with_columns(
    #     pl.col("wap").rolling_mean(3).over("stock_id", "date_id").cast(pl.Float32).alias("wap_30s_rolling_mean"),  # noqa
    #     pl.col("wap").rolling_mean(6).over("stock_id", "date_id").cast(pl.Float32).alias("wap_60s_rolling_mean"),  # noqa
    #     pl.col("wap").rolling_std(3).over("stock_id", "date_id").cast(pl.Float32).alias("wap_30s_rolling_std"),  # noqa
    #     pl.col("wap").rolling_std(6).over("stock_id", "date_id").cast(pl.Float32).alias("wap_60s_rolling_std"),  # noqa

    #     pl.col("wap").diff().over("stock_id", "date_id").cast(pl.Float32).alias("wap_diff"),  # noqa
    #     pl.col("wap").diff().rolling_mean(3).over("stock_id", "date_id").cast(pl.Float32).alias("wap_diff_30s_rolling_mean"),  # noqa
    #     pl.col("wap").diff().rolling_mean(6).over("stock_id", "date_id").cast(pl.Float32).alias("wap_diff_60s_rolling_mean"),  # noqa

    #     (pl.col("wap").shift(1).over("stock_id", "date_id") / pl.col("wap")).cast(pl.Float32).alias("wap_shift_1_ratio"),  # noqa
    #     (pl.col("wap").shift(2).over("stock_id", "date_id") / pl.col("wap")).cast(pl.Float32).alias("wap_shift_2_ratio"),  # noqa
    #     (pl.col("wap").shift(3).over("stock_id", "date_id") / pl.col("wap")).cast(pl.Float32).alias("wap_shift_3_ratio"),  # noqa
    #     (pl.col("wap").shift(4).over("stock_id", "date_id") / pl.col("wap")).cast(pl.Float32).alias("wap_shift_4_ratio"),  # noqa
    #     (pl.col("wap").shift(5).over("stock_id", "date_id") / pl.col("wap")).cast(pl.Float32).alias("wap_shift_5_ratio"),  # noqa
    #     (pl.col("wap").shift(6).over("stock_id", "date_id") / pl.col("wap")).cast(pl.Float32).alias("wap_shift_6_ratio"),  # noqa

    #     pl.col("wap").shift(1).over("stock_id", "date_id").cast(pl.Float32).alias("wap_shift_1"),  # noqa
    #     pl.col("wap").shift(2).over("stock_id", "date_id").cast(pl.Float32).alias("wap_shift_2"),  # noqa
    #     pl.col("wap").shift(3).over("stock_id", "date_id").cast(pl.Float32).alias("wap_shift_3"),  # noqa
    #     pl.col("wap").shift(4).over("stock_id", "date_id").cast(pl.Float32).alias("wap_shift_4"),  # noqa
    #     pl.col("wap").shift(5).over("stock_id", "date_id").cast(pl.Float32).alias("wap_shift_5"),  # noqa
    #     pl.col("wap").shift(6).over("stock_id", "date_id").cast(pl.Float32).alias("wap_shift_6"),  # noqa
    # )
    # # closing auction中で、sizeがどう変化しているかの情報を追加する
    # targets = [
    #     "imbalance_size",
    #     "matched_size",
    #     "bid_size",
    #     "ask_size",
    #     "bid_price",
    #     "ask_price",
    #     "near_price",
    #     "far_price",
    #     "reference_price"
    # ]
    # for col in targets:
    #     df = df.with_columns(
    #         pl.col(col).rolling_mean(3).over("stock_id", "date_id").cast(pl.Float32).alias(f"{col}_30s_rolling_mean"),  # noqa
    #         pl.col(col).rolling_mean(6).over("stock_id", "date_id").cast(pl.Float32).alias(f"{col}_60s_rolling_mean"),  # noqa
    #         pl.col(col).rolling_std(3).over("stock_id", "date_id").cast(pl.Float32).alias(f"{col}_30s_rolling_std"),  # noqa
    #         pl.col(col).rolling_std(6).over("stock_id", "date_id").cast(pl.Float32).alias(f"{col}_60s_rolling_std"),  # noqa

    #         pl.col(col).diff().over("stock_id", "date_id").cast(pl.Float32).alias(f"{col}_diff"),
    #         pl.col(col).diff().rolling_mean(3).over("stock_id", "date_id").cast(pl.Float32).alias(f"{col}_diff_30s_rolling_mean"),  # noqa
    #         pl.col(col).diff().rolling_mean(6).over("stock_id", "date_id").cast(pl.Float32).alias(f"{col}_diff_60s_rolling_mean"),  # noqa
    #     )
    return df


def __add_lag(df: pl.DataFrame) -> pl.DataFrame:
    # fmt: off
    for col in ["matched_size", "imbalance_size", "reference_price", "imbalance_buy_sell_flag"]:
        df = df.with_columns(
            pl.col(col).shift(1).over("stock_id").alias(f"{col}_shift_1"),
            pl.col(col).shift(3).over("stock_id").alias(f"{col}_shift_3"),
            pl.col(col).shift(5).over("stock_id").alias(f"{col}_shift_5"),
            pl.col(col).shift(10).over("stock_id").alias(f"{col}_shift_10"),
            pl.col(col).pct_change(1).over("stock_id").alias(f"{col}_ret_1"),
            pl.col(col).pct_change(3).over("stock_id").alias(f"{col}_ret_3"),
            pl.col(col).pct_change(5).over("stock_id").alias(f"{col}_ret_5"),
            pl.col(col).pct_change(10).over("stock_id").alias(f"{col}_ret_10"),
        )

    for col in ["ask_price", "bid_price", "ask_size", "bid_size", "weighted_wap", "price_spread"]:
        df = df.with_columns(
            pl.col(col).diff(1).over("stock_id").alias(f"{col}_diff_1"),
            pl.col(col).diff(3).over("stock_id").alias(f"{col}_diff_3"),
            pl.col(col).diff(5).over("stock_id").alias(f"{col}_diff_5"),
            pl.col(col).diff(10).over("stock_id").alias(f"{col}_diff_10"),
        )

    df = df.with_columns(
        (pl.col("bid_price_diff_3") - pl.col("ask_price_diff_3")).alias("price_change_diff_3"),
        (pl.col("bid_price_diff_5") - pl.col("ask_price_diff_5")).alias("price_change_diff_5"),
        (pl.col("bid_price_diff_10") - pl.col("ask_price_diff_10")).alias("price_change_diff_10"),

        (pl.col("bid_size_diff_3") - pl.col("ask_size_diff_3")).alias("size_change_diff_3"),
        (pl.col("bid_size_diff_5") - pl.col("ask_size_diff_5")).alias("size_change_diff_5"),
        (pl.col("bid_size_diff_10") - pl.col("ask_size_diff_10")).alias("size_change_diff_10"),
    )
    return df


def __add_pair_imbalance(df: pl.DataFrame, prices: List) -> pl.DataFrame:
    for comb in combinations(prices, 2):
        df = df.with_columns(
            (
                (pl.col(comb[0]) - pl.col(comb[1]))
                / (pl.col(comb[0]) + pl.col(comb[1]))
            ).alias(f"{comb[0]}_{comb[1]}_imb")
        )
    return df


def __add_imbalance(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("ask_size") + pl.col("bid_size")).alias("volume"),
        ((pl.col("ask_price") + pl.col("bid_price")) / 2).alias("mid_price"),
        (pl.col("ask_size") / pl.col("bid_size")).alias("size_imbalance"),
        (
            (pl.col("bid_size") - pl.col("ask_size"))
            / (pl.col("bid_size") + pl.col("ask_size"))
        ).alias("liquidity_imbalance"),
        (
            (pl.col("imbalance_size") - pl.col("matched_size"))
            / (pl.col("imbalance_size") + pl.col("matched_size"))
        ).alias("matched_imbalance"),
        (2 / (1 / pl.col("bid_size")) + (1 / pl.col("ask_size"))).alias(
            "harmonic_imbalance"
        ),
    )

    df = __add_pair_imbalance(df, PRICES)
    df = __add_triplet_imbalance(df)

    return df


def __add_triplet_imbalance(df: pl.DataFrame) -> pl.DataFrame:
    for comb in combinations(PRICES, 3):
        _max = df.select(comb).max_horizontal()
        _min = df.select(comb).min_horizontal()
        _mid = df.select(comb).sum_horizontal() - _max - _min
        if (_mid == _min).all():
            df = df.with_columns(
                pl.lit(np.nan).alias(f"{comb[0]}_{comb[1]}_{comb[2]}_imb")
            )
        else:
            # fmt: off
            df = df.with_columns(
                ((_max - _mid) / (_mid - _min)).alias(f"{comb[0]}_{comb[1]}_{comb[2]}_imb")
            )
    return df


def __add_pressure(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("price_spread")
    )
    # fmt: off
    df = df.with_columns(
        (pl.col("imbalance_size").diff().over("stock_id") / pl.col("matched_size")).alias("imbalance_momentum"),  # noqa
        pl.col("price_spread").diff().over("stock_id").alias("spread_intensity"),
        (pl.col("imbalance_size") * pl.col("price_spread")).alias("price_pressure"),
        (pl.col("liquidity_imbalance") * pl.col("price_spread")).alias("market_urgency"),
        ((pl.col("ask_size") - pl.col("bid_size")) * (pl.col("far_price") - pl.col("near_price"))).alias("depth_pressure"),  # noqa
        ((pl.col("ask_price") - pl.col("bid_price")) / (pl.col("ask_size") + pl.col("bid_size"))).alias("spread_depth_ratio"),  # noqa
        pl.col("mid_price").diff(5).apply(np.sign).cast(pl.Int8).alias("mid_price_movement"),
        (
            (
                (pl.col("bid_price") * pl.col("ask_size"))
                + (pl.col("ask_price") * pl.col("bid_size"))
            )
            / (pl.col("ask_size") + pl.col("bid_size"))
        ).alias("micro_price"),
        ((pl.col("ask_price") - pl.col("bid_price")) / pl.col("wap")).alias("relative_spread"),

    )
    df = df.with_columns(
        (pl.col("mid_price_movement") * pl.col("volume")).alias("mid_price*volume"),
    )
    return df


def __add_statistic_agg(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        df.select(PRICES).mean_horizontal().alias("all_prices_mean"),
        pl.Series(df.select(PRICES).to_pandas().std(axis=1)).alias("all_prices_std"),
        pl.Series(df.select(PRICES).to_pandas().skew(axis=1)).alias("all_prices_skew"),
        pl.Series(df.select(PRICES).to_pandas().kurt(axis=1)).alias("all_prices_kurt"),
        df.select(SIZES).mean_horizontal().alias("all_sizes_mean"),
        pl.Series(df.select(SIZES).to_pandas().std(axis=1)).alias("all_sizes_std"),
        pl.Series(df.select(SIZES).to_pandas().skew(axis=1)).alias("all_sizes_skew"),
        pl.Series(df.select(SIZES).to_pandas().kurt(axis=1)).alias("all_sizes_kurt"),
    )
    return df


def __add_time(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("date_id") % 5).alias("day_of_week"),
        (pl.col("seconds_in_bucket") % 60).alias("seconds"),
        (pl.col("seconds_in_bucket") // 60).alias("minutes"),
        (540 - pl.col("seconds_in_bucket")).alias("time_to_market_close"),
    )
    return df


def __add_stock_unit(df: pl.DataFrame) -> pl.DataFrame:
    # fmt: off
    stock_unit_df = df.group_by("stock_id").agg(
        (pl.col("bid_size").median() + pl.col("ask_size").median()).alias("stock_unit_median_size"),
        (pl.col("bid_size").std() + pl.col("ask_size").std()).alias("stock_unit_std_size"),
        (pl.col("bid_size").max() - pl.col("bid_size").min()).alias("stock_unit_ptp_size"),
        (pl.col("bid_price").median() + pl.col("ask_price").median()).alias("stock_unit_median_price"),  # noqa
        (pl.col("bid_price").std() + pl.col("ask_price").std()).alias("stock_unit_std_price"),
        (pl.col("bid_price").max() - pl.col("bid_price").min()).alias("stock_unit_ptp_price"),
    )
    df = df.join(stock_unit_df, on="stock_id")
    return df


# def __add_talib_feats(df: pl.DataFrame) -> pl.DataFrame:
#     # def add_macd(groups: pl.DataFrame) -> pl.DataFrame:
#     #     macd, macdsignal, macdhist = talib.MACD(
#     #         groups["wap"], fastperiod=12, slowperiod=26, signalperiod=9
#     #     )
#     #     groups = groups.with_columns(
#     #         macd.alias("macd"),
#     #         macdsignal.alias("macdsignal"),
#     #         macdhist.alias("macdhist"),
#     #     )
#     #     return groups

#     # return df.group_by("stock_id", "date_id").map_groups(add_macd)

#     return df.group_by("stock_id", "date_id").map_groups(
#         lambda group: group.with_columns(
#             talib.EMA(group["wap"], timeperiod=3).alias("wap_30s_ema")
#         )
#     )


def feature_engineering(df: pl.DataFrame, keep_stock_id: bool = False) -> pl.DataFrame:
    df = __add_imbalance(df)
    df = __add_pressure(df)
    df = __add_statistic_agg(df)
    df = __add_index_wap(df)

    df = __add_lag(df)
    df = __add_rolling(df)
    # df = __add_talib_feats(df)

    df = __add_time(df)
    df = __add_stock_unit(df)

    for col in df.columns:
        df = df.with_columns(
            pl.when((df[col] == np.inf) | (df[col] == -np.inf))
            .then(0)
            .otherwise(df[col])
            .alias(col)
        )

    if not keep_stock_id:
        df = df.drop("stock_id")
    return df.drop("row_id")
