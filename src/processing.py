import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    feats = [
        "seconds_in_bucket",
        "imbalance_buy_sell_flag",
        "imbalance_size",
        "matched_size",
        "bid_size",
        "ask_size",
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
        "imb_s1",
        "imb_s2",
    ]

    df["imb_s1"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["imb_s2"] = df.eval(
        "(imbalance_size-matched_size)/(matched_size+imbalance_size)"
    )

    prices = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
    ]

    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            if i > j:
                df[f"{a}_{b}_imb"] = df.eval(f"({a}-{b})/({a}+{b})")
                feats.append(f"{a}_{b}_imb")

    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            for k, c in enumerate(prices):
                if i > j and j > k:
                    max_ = df[[a, b, c]].max(axis=1)
                    min_ = df[[a, b, c]].min(axis=1)
                    mid_ = df[[a, b, c]].sum(axis=1) - min_ - max_

                    df[f"{a}_{b}_{c}_imb2"] = (max_ - mid_) / (mid_ - min_)
                    feats.append(f"{a}_{b}_{c}_imb2")

    return df[feats]
