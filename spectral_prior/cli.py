"""Build, train and export spectral priors independently of PPO."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument(
        "--input", required=True, help="Directory of LAFAN1 motions retargeted with GMR and exported as NPZ"
    )
    build.add_argument("--output", default=str(root / "outputs/reference"))
    build.add_argument("--selections")
    train = sub.add_parser("train")
    train.add_argument("--library", default=str(root / "outputs/reference/library.npz"))
    train.add_argument("--output", default=str(root / "outputs/prior.pt"))
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--beta-kl", type=float, default=1e-4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=4)
    export = sub.add_parser("export")
    export.add_argument("--prior", default=str(root / "outputs/prior.pt"))
    export.add_argument("--output", default=str(root / "outputs/reference_sweep.npz"))
    evaluate = sub.add_parser("evaluate", help="Evaluate paper prior metrics (equations 15–17)")
    evaluate.add_argument("--prior", default=str(root / "outputs/prior.pt"))
    evaluate.add_argument("--library", default=str(root / "outputs/reference/library.npz"))
    evaluate.add_argument("--output", default=str(root / "outputs/prior_evaluation.json"))
    args = parser.parse_args(argv)
    if args.command == "build":
        from spectral_prior.data import build_library

        print(json.dumps(build_library(args.input, args.output, selections=args.selections), indent=2))
    elif args.command == "train":
        import torch

        from spectral_prior.training import train_prior

        if args.threads <= 0:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
        train_prior(
            args.library, args.output, args.steps, args.batch_size, args.lr, args.beta_kl, args.device, args.seed
        )
    elif args.command == "evaluate":
        import torch

        from spectral_prior.metrics import evaluate_prior, write_report

        torch.set_num_threads(4)
        report = evaluate_prior(args.prior, args.library)
        write_report(report, args.output)
        print(json.dumps(report, indent=2))
    else:
        from spectral_prior.export import export_reference

        export_reference(args.prior, args.output)


if __name__ == "__main__":
    main()
