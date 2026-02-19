"""
Step 2: Submit Slurm backtest jobs for each saved alpha file.

Usage:
    python 2_backtest.py --split train
    python 2_backtest.py --split test --lambda 0.015873
    python 2_backtest.py --split train --dry-run   # print scripts without submitting
    python 2_backtest.py --split train --clean      # wipe old weights before submitting
"""
import argparse
import os
import shutil

from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig
from config import SPLITS, LAMBDA_GRID, GAMMA, CONSTRAINTS, BYU_EMAIL, PROJECT_ROOT, signal_name, alphas_path, weights_dir, split_dir


def main():
    parser = argparse.ArgumentParser(description="Submit Slurm backtests")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--lambda", dest="lamb", type=float, default=None,
                        help="Single lambda value (default: use full grid)")
    parser.add_argument("--dry-run", action="store_true", help="Print scripts without submitting")
    parser.add_argument("--clean", action="store_true", help="Wipe old weights before submitting")
    args = parser.parse_args()

    lambdas = [args.lamb] if args.lamb else LAMBDA_GRID

    slurm_config = SlurmConfig(
        n_cpus=8,
        mem="32G",
        time="03:00:00",
        mail_type="BEGIN,END,FAIL",
        max_concurrent_jobs=30,
    )

    print(f"Split: {args.split}")
    print(f"Submitting {len(lambdas)} backtest(s){'  [DRY RUN]' if args.dry_run else ''}...\n")

    for lamb in lambdas:
        hl = int(0.693 / lamb)
        data_path = alphas_path(args.split, lamb)

        if not os.path.exists(data_path):
            print(f"  ✗ λ={lamb:.6f} (HL={hl}d): {data_path} not found, skipping")
            continue

        # Clean old weights if requested, then create output directory
        out_dir = weights_dir(args.split, lamb)
        if args.clean and os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            print(f"    (cleaned {out_dir})")
        os.makedirs(out_dir, exist_ok=True)

        config = BacktestConfig(
            signal_name=signal_name(lamb),
            data_path=data_path,
            gamma=GAMMA,
            project_root=split_dir(args.split),
            byu_email=BYU_EMAIL,
            constraints=CONSTRAINTS,
            slurm=slurm_config,
        )

        runner = BacktestRunner(config)
        runner.submit(dry_run=args.dry_run)
        print(f"  ✓ λ={lamb:.6f} (HL={hl}d) submitted")

    print("\nDone. Wait for Slurm jobs to finish before running step 3.")


if __name__ == "__main__":
    main()