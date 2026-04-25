"""
IPL 2021-2026: Day vs Night weather analysis.

For each weather variable (temperature, humidity, 7d rainfall EMA, 30d rainfall EMA),
plot first-innings run rate vs the variable, separately for day and night cohorts,
after iteratively removing points >2 SD from the mean (on either axis) until stable.

Two panels per variable:
  - Per-match scatter with iterative 2sigma outlier removal (matches the user's
    written instruction).
  - Bucket scatter (deciles by x value, on the kept-after-2sigma sample), mirroring
    the IPL/T20 Blast example charts the user shared.

Note on day/night classification: match_day_night for IPL 2021 and 2024 used
placeholder 05:30 IST start times under the start_time_sunset method, which
mis-flags every match as 'day'. We override those with ESPN description markers
from web_start_time_sources for any match where the recorded method was
start_time_sunset.
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
                -- For seasons where match_day_night used the broken
                -- start_time_sunset method (placeholder 05:30 IST times),
                -- fall back to ESPN description markers from web sources.
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
        wm.average_relative_humidity_2m_pct AS humidity_pct,
        mr.rainfall_ema_hl_7d_mm_per_day AS rain_ema_7d,
        mr.rainfall_ema_hl_30d_mm_per_day AS rain_ema_30d
    FROM classified c
    LEFT JOIN innings inn
        ON inn.match_id = c.match_id AND inn.innings_num = 1
        AND COALESCE(inn.is_super_over, 0) = 0
    LEFT JOIN match_weather_match_summary wm ON wm.match_id = c.match_id
    LEFT JOIN match_rainfall mr ON mr.match_id = c.match_id
    """
    with sqlite3.connect(DB) as con:
        df = pd.read_sql(sql, con)
    df = df[df["day_night"].isin(["day", "night"])].copy()
    df = df.dropna(subset=["first_innings_rr"]).copy()
    df = df[df["first_innings_balls"] >= 60].copy()
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
    out = pd.DataFrame({
        "x_mean": g[x_col].mean(),
        "y_mean": g["first_innings_rr"].mean(),
        "n": g.size(),
    }).reset_index(drop=True)
    return out


def plot_var(df: pd.DataFrame, x_col: str, x_label: str, title: str, fname: str) -> dict:
    """Two-row figure: top = per-match scatter; bottom = bucket scatter. Day | Night columns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1]})
    summary: dict = {}
    for col, dn, color in [(0, "day", DAY_COLOR), (1, "night", NIGHT_COLOR)]:
        sub = df[df["day_night"] == dn]
        kept, removed, iters = iterative_2sd(sub, x_col, "first_innings_rr")
        reg_match = regress(kept[x_col], kept["first_innings_rr"])

        # Top: per-match scatter
        ax_top = axes[0, col]
        ax_top.scatter(kept[x_col], kept["first_innings_rr"], s=70, color=color, alpha=0.8,
                       edgecolor="white", linewidth=0.6, label=f"kept (n={len(kept)})")
        if len(removed):
            ax_top.scatter(removed[x_col], removed["first_innings_rr"], s=45, c="lightgray",
                           alpha=0.55, marker="x", linewidth=0.9, label=f"removed (n={len(removed)})")
        if not np.isnan(reg_match["slope"]) and len(kept) >= 3:
            xs = np.linspace(kept[x_col].min(), kept[x_col].max(), 50)
            ax_top.plot(xs, reg_match["intercept"] + reg_match["slope"] * xs, color=FIT_COLOR, lw=2, label="OLS fit")
        if len(kept) >= 3:
            xmin, xmax = kept[x_col].min(), kept[x_col].max()
            pad = max(0.03 * (xmax - xmin), 1e-3)
            ax_top.set_xlim(xmin - pad, xmax + pad)
        ax_top.set_title(
            f"{dn.title()} — per-match (kept={len(kept)}/{len(sub.dropna(subset=[x_col,'first_innings_rr']))})\n"
            f"r={reg_match['r']:.3f}, p={reg_match['p']:.3f}, 2σ iters={iters}",
            fontsize=11,
        )
        ax_top.set_xlabel(x_label)
        if col == 0:
            ax_top.set_ylabel("1st innings RR (runs/over)")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="best", fontsize=9)

        # Bottom: bucket scatter on kept sample
        ax_bot = axes[1, col]
        n_buckets = 10 if dn == "night" else 6
        bk = make_buckets(kept, x_col, n_buckets)
        if not bk.empty and len(bk) >= 3:
            reg_bk = regress(bk["x_mean"], bk["y_mean"])
            sizes = 60 + 20 * bk["n"].astype(float)
            ax_bot.scatter(bk["x_mean"], bk["y_mean"], s=sizes, color=color, alpha=0.85,
                           edgecolor="white", linewidth=0.7)
            for i, row in bk.iterrows():
                ax_bot.annotate(f"{int(row['n'])}", (row["x_mean"], row["y_mean"]),
                                xytext=(5, 4), textcoords="offset points", fontsize=8, color="#374151")
            xs = np.linspace(bk["x_mean"].min(), bk["x_mean"].max(), 50)
            ax_bot.plot(xs, reg_bk["intercept"] + reg_bk["slope"] * xs, color=FIT_COLOR, lw=2)
            ax_bot.set_title(
                f"{dn.title()} — {len(bk)}-bucket means (kept sample)\n"
                f"r={reg_bk['r']:.3f}, p={reg_bk['p']:.3f}",
                fontsize=11,
            )
            summary[dn] = {
                "n_initial": int(len(sub.dropna(subset=[x_col, 'first_innings_rr']))),
                "n_kept": int(len(kept)),
                "n_removed": int(len(removed)),
                "iterations": int(iters),
                "match_r": float(reg_match["r"]) if not np.isnan(reg_match["r"]) else None,
                "match_p": float(reg_match["p"]) if not np.isnan(reg_match["p"]) else None,
                "match_slope": float(reg_match["slope"]) if not np.isnan(reg_match["slope"]) else None,
                "buckets": int(len(bk)),
                "bucket_r": float(reg_bk["r"]) if not np.isnan(reg_bk["r"]) else None,
                "bucket_p": float(reg_bk["p"]) if not np.isnan(reg_bk["p"]) else None,
                "bucket_slope": float(reg_bk["slope"]) if not np.isnan(reg_bk["slope"]) else None,
            }
        else:
            ax_bot.text(0.5, 0.5, "Insufficient kept points\nfor bucket analysis",
                        ha="center", va="center", transform=ax_bot.transAxes, color="#6b7280")
            summary[dn] = {
                "n_initial": int(len(sub.dropna(subset=[x_col, 'first_innings_rr']))),
                "n_kept": int(len(kept)),
                "n_removed": int(len(removed)),
                "iterations": int(iters),
                "match_r": float(reg_match["r"]) if not np.isnan(reg_match["r"]) else None,
                "match_p": float(reg_match["p"]) if not np.isnan(reg_match["p"]) else None,
                "match_slope": float(reg_match["slope"]) if not np.isnan(reg_match["slope"]) else None,
                "buckets": 0,
                "bucket_r": None,
                "bucket_p": None,
                "bucket_slope": None,
            }
        ax_bot.set_xlabel(x_label + " (bucket mean)")
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
    df.to_csv(OUT / "ipl_day_night_dataset.csv", index=False)
    print(f"Loaded {len(df)} matches with first-innings RR")
    print(df.groupby("day_night").size().to_string(), "\n")

    plans = [
        ("temp_c", "Match-window mean temperature (°C)",
         "IPL 2021–2026: 1st innings RR vs Temperature (Day vs Night)", "ipl_temp_dn.png"),
        ("humidity_pct", "Match-window mean relative humidity (%)",
         "IPL 2021–2026: 1st innings RR vs Humidity (Day vs Night)", "ipl_humidity_dn.png"),
        ("rain_ema_7d", "7-day rainfall EMA (mm/day, prior day)",
         "IPL 2021–2026: 1st innings RR vs 7-day Rainfall EMA (Day vs Night)", "ipl_rain7d_dn.png"),
        ("rain_ema_30d", "30-day rainfall EMA (mm/day, prior day)",
         "IPL 2021–2026: 1st innings RR vs 30-day Rainfall EMA (Day vs Night)", "ipl_rain30d_dn.png"),
    ]
    full_summary: dict = {}
    for x_col, x_label, title, fname in plans:
        full_summary[x_col] = plot_var(df, x_col, x_label, title, fname)
        print(f"{x_col}:")
        for dn, s in full_summary[x_col].items():
            mr = s["match_r"]
            br = s["bucket_r"]
            print(f"  {dn:5s} kept={s['n_kept']:3d}/{s['n_initial']:3d} (rm {s['n_removed']:2d}, iters {s['iterations']:2d})  "
                  f"match r={mr if mr is None else round(mr,3):>6}  "
                  f"buckets={s['buckets']:>2}  "
                  f"bucket r={br if br is None else round(br,3):>6}")

    rows = []
    for v, by_dn in full_summary.items():
        for dn, s in by_dn.items():
            rows.append({"variable": v, "day_night": dn, **s})
    pd.DataFrame(rows).to_csv(OUT / "ipl_correlation_summary.csv", index=False)
    print(f"\nSaved figures + summary to {OUT}")


if __name__ == "__main__":
    main()
