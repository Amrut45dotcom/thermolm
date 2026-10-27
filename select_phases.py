
import argparse
import sys

import pandas as pd

NEEDED = ["phase", "eligible", "train_positive_compositions", "valid_positive_rows",
          "test_positive_rows", "overall_positive_rows", "overall_positive_fraction"]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", default="phase_support_inspection.csv")
    p.add_argument("--per-tier", default="4,3,3", help="how many phases from common,medium,rare")
    p.add_argument("--common-min", type=float, default=0.15)
    p.add_argument("--rare-max", type=float, default=0.05)
    p.add_argument("--min-eval-pos-rows", type=int, default=200,
                   help="min positive rows in BOTH valid and test (inspect script used 50)")
    p.add_argument("--min-train-pos-rows", type=int, default=1000,
                   help="min approx. positive rows in train, so a regression on NP is possible")
    p.add_argument("--include", default="", help="comma-separated phases to force in")
    a = p.parse_args()

    d = pd.read_csv(a.csv)
    miss = [c for c in NEEDED if c not in d.columns]
    if miss:
        sys.exit(f"{a.csv} is missing columns {miss}. Re-run inspect_phase_support.py.")

    n_common, n_medium, n_rare = (int(x) for x in a.per_tier.split(","))
    d["approx_train_pos_rows"] = (d["overall_positive_rows"] * 0.7).round().astype(int)
    ok = (
        d["eligible"].astype(bool)
        & (d["valid_positive_rows"] >= a.min_eval_pos_rows)
        & (d["test_positive_rows"] >= a.min_eval_pos_rows)
        & (d["approx_train_pos_rows"] >= a.min_train_pos_rows)
    )
    d["usable"] = ok
    d["tier"] = pd.cut(d["overall_positive_fraction"],
                       bins=[-1, a.rare_max, a.common_min, 2], labels=["rare", "medium", "common"],
                       right=False)

    chosen = []
    for tier, n in (("common", n_common), ("medium", n_medium), ("rare", n_rare)):
        pool = d[ok & (d["tier"] == tier)].sort_values("train_positive_compositions", ascending=False)
        chosen += pool.head(n)["phase"].tolist()
        if len(pool) < n:
            print(f"NOTE: only {len(pool)} usable phases in tier '{tier}', wanted {n}")
    forced = [s.strip() for s in a.include.split(",") if s.strip()]
    unknown = [f for f in forced if f not in set(d["phase"])]
    if unknown:
        sys.exit(f"--include phases not in the csv: {unknown}")
    for f in forced:
        if f not in chosen:
            chosen.append(f)
    d["chosen"] = d["phase"].isin(chosen)

    cols = ["phase", "tier", "usable", "chosen", "overall_positive_fraction",
            "train_positive_compositions", "approx_train_pos_rows", "valid_positive_rows", "test_positive_rows"]
    show = d[d["usable"] | d["chosen"]].sort_values(["tier", "train_positive_compositions"],
                                                    ascending=[False, False])
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nChosen phases:")
    print(",".join(chosen))
    print("\nNext:")
    print(f"python run_tabfm_phases.py --data <csv> --phases {','.join(chosen)}")


if __name__ == "__main__":
    main()
