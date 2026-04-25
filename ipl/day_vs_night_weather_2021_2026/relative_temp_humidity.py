"""
IPL 2021-2026: Day vs Night relative-anomaly weather analysis.

The absolute-temperature analysis showed near-zero per-match correlations
with 1st-innings RR. This script tests two anomaly framings to see whether
the signal hides in deviations rather than absolute values:

  1. Cohort anomaly: x_match - mean(x | same cohort)
     Subtracting the cohort mean does not change r within a cohort
     (constant shift), but pools day + night meaningfully.

  2. Venue + cohort anomaly: x_match - mean(x | same venue, same cohort)
     Also removes "this venue is naturally hotter/more-humid" as a
     confound. Mirrors the venue-relative-rainfall framing already used
     in ipl_men_2022_2026_relative_rainfall_scoring_* tables.

Outputs three figures (each with two panels: temp + humidity):
  - cohort-relative pooled
  - venue+cohort-relative pooled
  - venue+cohort-relative split by day vs night

Iterative 2-sigma outlier removal applied throughout.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DB = "/Users/abhinavarora/Downloads/cricket (1).db"
OUT = Path("/Users/abhinavarora/Documents/Claude/Projects/Cricket research/ipl_day_night_weather")
OUT.mkdir(parents=True, exist_ok=True)

DAY_COLOR = "#d97706"
NIGHT_COLOR = "#2563eb"
POOLED_COLOR = "#7c3aed"
FIT_COLOR = "#b91c1c"


def load() -> pd.DataFrame:
    sql = """
    WITH ipl AS (
        SELECT m.match_id, m.season, m.start_date, m.venue, m.city
        FROM matches m
        WHERE m.event_name LIKE '%Indian Premier League%'
          AND m.season IN ('2021','2022','2023','2024','2025','2026')
    ),
    classified AS (
        SELECT
            i.match_id, i.season, i.start_date, i.venue, i.city,
            CASE
                WHEN mdn.method = 'start_time_sunset' OR mdn.method IS NULL THEN
                    CASE
                        WHEN ws.source_description LIKE '%(D/N)%' OR ws.source_description LIKE '%(D)%' THEN 'day'
                        WHEN ws.source_description LIKE '%(N)%' THEN 'night'
                        ELSE 'unknown'
                    END
                ELSE
                    CASE
                        WHEN mdn.day_night IN ('day','day/night') THEN 'day'
                        WHEN mdn.day_night = 'night' THEN 'night'
                        ELSE 'unknown'
                    END
            END AS day_night
        FROM ipl i
        LEFT JOIN web_start_time_sources ws ON ws.match_id = i.match_id
        LEFT JOIN match_day_night mdn ON mdn.match_id = i.match_id
    )
    SELECT
        c.match_id, c.season, c.start_date, c.venue, c.city, c.day_night,
        inn.total_runs AS first_innings_runs,
        inn.total_legal_balls AS first_innings_balls,
        CASE WHEN inn.total_legal_balls > 0
             THEN 6.0 * inn.total_runs / inn.total_legal_balls END AS first_innings_rr,
        wm.average_temperature_2m_c AS temp_c,
        wm.average_relative_humidity_2m_pct AS humidity_pct
    FROM classified c
    LEFT JOIN innings inn
        ON inn.match_id = c.match_id AND inn.innings_num = 1
        AND COALESCE(inn.is_super_over, 0) = 0
    LEFT JOIN match_weather_match_summary wm ON wm.match_id = c.match_id
    """
    with sqlite3.connect(DB) as con:
        df = pd.read_sql(sql, con)
    df = df[df["day_night"].isin(["day", "night"])].copy()
    df = df.dropna(subset=["first_innings_rr", "temp_c", "humidity_pct"]).copy()
    df = df[df["first_innings_balls"] >= 60].copy()
    return df


def add_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["temp_c", "humidity_pct"]:
        # Cohort mean
        cm = df.groupby("day_night")[col].transform("mean")
        df[f"{col}_anom_cohort"] = df[col] - cm
        # Venue + cohort mean (only if at least 2 matches at venue+cohort, else cohort mean)
        vcm = df.groupby(["venue", "day_night"])[col].transform("mean")
        vc_n = df.groupby(["venue", "day_night"])[col].transform("size")
        df[f"{col}_venue_cohort_mean"] = np.where(vc_n >= 2, vcm, cm)
        df[f"{col}_anom_venue_cohort"] = df[col] - df[f"{col}_venue_cohort_mean"]
        df[f"{col}_venue_cohort_n"] = vc_n
    return df


def iterative_2sd(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    cur = df.dropna(subset=[x_col, y_col]).copy()
    iterations = 0
    while True:
        if len(cur) < 5:
            break
        mx, sx = cur[x_col].mean(), cur[x_col].std(ddof=0)
        my, sy = cur[y_col].mean(), cur[y_col].std(ddof=0)
        keep = (
            cur[x_col].between(mx - 2 * sx, mx + 2 * sx)
            & cur[y_col].between(my - 2 * sy, my + 2 * sy)
        )
        if keep.all():
            break
        cur = cur.loc[keep].copy()
        iterations += 1
    removed = df.dropna(subset=[x_col, y_col]).drop(index=cur.index, errors="ignore")
    return cur, removed, iterations


def regress(x: pd.Series, y: pd.Series) -> dict:
    if len(x) < 3 or x.std(ddof=0) == 0:
        return {"r": np.nan, "p": np.nan, "slope": np.nan, "intercept": np.nan, "n": int(len(x))}
    res = stats.linregress(x, y)
    return {"r": res.rvalue, "p": res.pvalue, "slope": res.slope, "intercept": res.intercept, "n": int(len(x))}


def make_buckets(kept: pd.DataFrame, x_col: str, n_buckets: int) -> pd.DataFrame:
    if len(kept) < n_buckets:
        n_buckets = max(3, len(kept) // 3)
    try:
        bins = pd.qcut(kept[x_col], q=n_buckets, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = kept.groupby(bins, observed=True)
    return pd.DataFrame({
        "x_mean": g[x_col].mean(),
        "y_mean": g["first_innings_rr"].mean(),
        "n": g.size(),
    }).reset_index(drop=True)


def plot_pooled(df: pd.DataFrame, x_col: str, x_label: str, title: str, fname: str) -> dict:
    """One row, two columns: temp anomaly | humidity anomaly. Day & night as same colour, single regression."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    summary: dict = {}

    for row, x_name, x_lbl in [
        (0, x_col + "_anom_cohort", x_label + " (cohort anomaly)"),
        (1, x_col + "_anom_venue_cohort", x_label + " (venue+cohort anomaly)"),
    ]:
        # Per-match scatter (pooled day + night)
        ax_top = axes[row, 0]
        kept, removed, iters = iterative_2sd(df, x_name, "first_innings_rr")
        reg = regress(kept[x_name], kept["first_innings_rr"])
        # color by cohort but single fit
        for dn, color in [("day", DAY_COLOR), ("night", NIGHT_COLOR)]:
            sub = kept[kept["day_night"] == dn]
            ax_top.scatter(sub[x_name], sub["first_innings_rr"], s=45, color=color, alpha=0.7,
                           edgecolor="white", linewidth=0.4, label=f"{dn} (n={len(sub)})")
        if len(removed):
            ax_top.scatter(removed[x_name], removed["first_innings_rr"], s=30, c="lightgray",
                           alpha=0.45, marker="x", linewidth=0.7, label=f"removed (n={len(removed)})")
        if not np.isnan(reg["slope"]):
            xs = np.linspace(kept[x_name].min(), kept[x_name].max(), 50)
            ax_top.plot(xs, reg["intercept"] + reg["slope"] * xs, color=FIT_COLOR, lw=2, label="pooled fit")
        if len(kept) >= 3:
            xmin, xmax = kept[x_name].min(), kept[x_name].max()
            pad = max(0.03 * (xmax - xmin), 1e-3)
            ax_top.set_xlim(xmin - pad, xmax + pad)
        ax_top.axvline(0, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.6)
        ax_top.set_title(
            f"{x_lbl} \u2014 per-match pooled (kept={len(kept)}/{len(df)})\n"
            f"r={reg['r']:.3f}, p={reg['p']:.3f}, 2\u03c3 iters={iters}",
            fontsize=10,
        )
        ax_top.set_xlabel(x_lbl)
        ax_top.set_ylabel("1st innings RR (runs/over)")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="best", fontsize=8)

        # Bucket panel
        ax_bot = axes[row, 1]
        bk = make_buckets(kept, x_name, 10)
        if not bk.empty and len(bk) >= 3:
            reg_bk = regress(bk["x_mean"], bk["y_mean"])
            sizes = 60 + 12 * bk["n"].astype(float)
            ax_bot.scatter(bk["x_mean"], bk["y_mean"], s=sizes, color=POOLED_COLOR, alpha=0.85,
                           edgecolor="white", linewidth=0.7)
            for _, row_b in bk.iterrows():
                ax_bot.annotate(f"{int(row_b['n'])}", (row_b["x_mean"], row_b["y_mean"]),
                                xytext=(5, 4), textcoords="offset points", fontsize=8, color="#374151")
            xs = np.linspace(bk["x_mean"].min(), bk["x_mean"].max(), 50)
            ax_bot.plot(xs, reg_bk["intercept"] + reg_bk["slope"] * xs, color=FIT_COLOR, lw=2)
            ax_bot.axvline(0, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.6)
            ax_bot.set_title(
                f"{x_lbl} \u2014 {len(bk)} buckets (kept sample)\n"
                f"r={reg_bk['r']:.3f}, p={reg_bk['p']:.3f}",
                fontsize=10,
            )
            summary[x_name] = {
                "n_initial": int(len(df.dropna(subset=[x_name, 'first_innings_rr']))),
                "n_kept": int(len(kept)),
                "n_removed": int(len(removed)),
                "iterations": int(iters),
                "match_r": float(reg["r"]) if not np.isnan(reg["r"]) else None,
                "match_p": float(reg["p"]) if not np.isnan(reg["p"]) else None,
                "match_slope": float(reg["slope"]) if not np.isnan(reg["slope"]) else None,
                "buckets": int(len(bk)),
                "bucket_r": float(reg_bk["r"]) if not np.isnan(reg_bk["r"]) else None,
                "bucket_p": float(reg_bk["p"]) if not np.isnan(reg_bk["p"]) else None,
            }
        else:
            ax_bot.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax_bot.transAxes, color="#6b7280")
            summary[x_name] = {"n_initial": int(len(df)), "n_kept": int(len(kept)), "iterations": int(iters)}
        ax_bot.set_xlabel(f"{x_lbl} (bucket mean)")
        ax_bot.set_ylabel("Bucket mean 1st innings RR")
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_split(df: pd.DataFrame, x_col: str, x_label: str, title: str, fname: str) -> dict:
    """Split by cohort. 2 rows (per-match | bucket), 2 cols (day | night). Uses venue+cohort anomaly."""
    x_name = f"{x_col}_anom_venue_cohort"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    summary: dict = {}
    for col, dn, color in [(0, "day", DAY_COLOR), (1, "night", NIGHT_COLOR)]:
        sub = df[df["day_night"] == dn]
        kept, removed, iters = iterative_2sd(sub, x_name, "first_innings_rr")
        reg_match = regress(kept[x_name], kept["first_innings_rr"])

        ax_top = axes[0, col]
        ax_top.scatter(kept[x_name], kept["first_innings_rr"], s=70, color=color, alpha=0.8,
                       edgecolor="white", linewidth=0.6, label=f"kept (n={len(kept)})")
        if len(removed):
            ax_top.scatter(removed[x_name], removed["first_innings_rr"], s=45, c="lightgray",
                           alpha=0.55, marker="x", linewidth=0.9, label=f"removed (n={len(removed)})")
        if not np.isnan(reg_match["slope"]):
            xs = np.linspace(kept[x_name].min(), kept[x_name].max(), 50)
            ax_top.plot(xs, reg_match["intercept"] + reg_match["slope"] * xs, color=FIT_COLOR, lw=2, label="OLS fit")
        if len(kept) >= 3:
            xmin, xmax = kept[x_name].min(), kept[x_name].max()
            pad = max(0.03 * (xmax - xmin), 1e-3)
            ax_top.set_xlim(xmin - pad, xmax + pad)
        ax_top.axvline(0, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.6)
        ax_top.set_title(
            f"{dn.title()} \u2014 per-match (kept={len(kept)}/{len(sub.dropna(subset=[x_name, 'first_innings_rr']))})\n"
            f"r={reg_match['r']:.3f}, p={reg_match['p']:.3f}, 2\u03c3 iters={iters}",
            fontsize=11,
        )
        ax_top.set_xlabel(f"{x_label} (venue+cohort anomaly)")
        if col == 0:
            ax_top.set_ylabel("1st innings RR (runs/over)")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="best", fontsize=9)

        ax_bot = axes[1, col]
        n_buckets = 10 if dn == "night" else 6
        bk = make_buckets(kept, x_name, n_buckets)
        if not bk.empty and len(bk) >= 3:
            reg_bk = regress(bk["x_mean"], bk["y_mean"])
            sizes = 60 + 20 * bk["n"].astype(float)
            ax_bot.scatter(bk["x_mean"], bk["y_mean"], s=sizes, color=color, alpha=0.85,
                           edgecolor="white", linewidth=0.7)
            for _, row_b in bk.iterrows():
                ax_bot.annotate(f"{int(row_b['n'])}", (row_b["x_mean"], row_b["y_mean"]),
                                xytext=(5, 4), textcoords="offset points", fontsize=8, color="#374151")
            xs = np.linspace(bk["x_mean"].min(), bk["x_mean"].max(), 50)
            ax_bot.plot(xs, reg_bk["intercept"] + reg_bk["slope"] * xs, color=FIT_COLOR, lw=2)
            ax_bot.axvline(0, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.6)
            ax_bot.set_title(f"{dn.title()} \u2014 {len(bk)} buckets\nr={reg_bk['r']:.3f}, p={reg_bk['p']:.3f}", fontsize=10)
            summary[dn] = {
                "n_initial": int(len(sub.dropna(subset=[x_name, 'first_innings_rr']))),
                "n_kept": int(len(kept)),
                "match_r": float(reg_match["r"]) if not np.isnan(reg_match["r"]) else None,
                "match_p": float(reg_match["p"]) if not np.isnan(reg_match["p"]) else None,
                "iterations": int(iters),
                "buckets": int(len(bk)),
                "bucket_r": float(reg_bk["r"]) if not np.isnan(reg_bk["r"]) else None,
                "bucket_p": float(reg_bk["p"]) if not np.isnan(reg_bk["p"]) else None,
            }
        else:
            ax_bot.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax_bot.transAxes, color="#6b7280")
        ax_bot.set_xlabel(f"{x_label} (venue+cohort anomaly, bucket mean)")
        if col == 0:
            ax_bot.set_ylabel("Bucket mean 1st innings RR")
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return summary


def main():
    df = load()
    df = add_anomalies(df)
    print(f"Loaded {len(df)} matches | day {(df.day_night=='day').sum()} | night {(df.day_night=='night').sum()}")

    df.to_csv(OUT / "ipl_relative_anomaly_dataset.csv", index=False)

    summaries: dict = {}

    summaries["temp_pooled"] = plot_pooled(
        df, "temp_c", "Temperature (\u00b0C)",
        "IPL 2021\u20132026: Temperature anomaly vs 1st innings RR (pooled day + night)",
        "ipl_temp_anomaly_pooled.png",
    )
    summaries["humidity_pooled"] = plot_pooled(
        df, "humidity_pct", "Humidity (%)",
        "IPL 2021\u20132026: Humidity anomaly vs 1st innings RR (pooled day + night)",
        "ipl_humidity_anomaly_pooled.png",
    )
    summaries["temp_split"] = plot_split(
        df, "temp_c", "Temperature (\u00b0C)",
        "IPL 2021\u20132026: Venue+cohort temperature anomaly vs 1st innings RR (split)",
        "ipl_temp_anomaly_split.png",
    )
    summaries["humidity_split"] = plot_split(
        df, "humidity_pct", "Humidity (%)",
        "IPL 2021\u20132026: Venue+cohort humidity anomaly vs 1st innings RR (split)",
        "ipl_humidity_anomaly_split.png",
    )

    print("\nResults:")
    for name, s in summaries.items():
        print(f"\n  {name}:")
        for key, val in s.items():
            if isinstance(val, dict):
                mr = val.get("match_r")
                mp = val.get("match_p")
                br = val.get("bucket_r")
                bp = val.get("bucket_p")
                nk = val.get("n_kept")
                ni = val.get("n_initial")
                it = val.get("iterations")
                print(f"    {key:30s}  kept={nk}/{ni} (iters {it})  match r={mr if mr is None else round(mr,3)}, p={mp if mp is None else round(mp,3)}  bucket r={br if br is None else round(br,3)}, p={bp if bp is None else round(bp,3)}")

    # Flat summary CSV
    rows = []
    for chart, by_var in summaries.items():
        for var, s in by_var.items():
            if isinstance(s, dict):
                rows.append({"chart": chart, "variable": var, **s})
    pd.DataFrame(rows).to_csv(OUT / "ipl_relative_correlation_summary.csv", index=False)
    print(f"\nSaved figures to {OUT}")


if __name__ == "__main__":
    main()
