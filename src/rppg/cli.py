"""One public CLI for all three datasets, two stages, audits and experiments.

No hard-coded local paths. Trained weights are generated under outputs/ only.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
from .config import load_settings
from .splits import prepare_split, save_split, audit_split
from .preprocessing.manifest import read_manifest, preprocess_dataset
from .training import train_stage1, train_stage2, evaluate
from .experiments.ablations import ABLATIONS, ablation_config
from .evaluation.statistics import subject_bootstrap, summarize_seeds


def parser():
    p = argparse.ArgumentParser(prog="rppg", description="Two-stage subject-disjoint rPPG experiments")
    cmd = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="configs/pure.yaml, cohface.yaml or ubfc.yaml")
    common.add_argument("--manifest", help="Path to processed manifest.csv")
    common.add_argument("--split", help="Path to saved subject split JSON")
    common.add_argument("--output", default="outputs")
    process = cmd.add_parser("preprocess", parents=[common])
    process.add_argument("--raw-root", required=True)
    process.add_argument("--predictor", required=True)
    process.add_argument("--limit-sessions", type=int)
    split = cmd.add_parser("split", parents=[common])
    split.add_argument("--frozen", help="Archived subject-level membership file")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--write", required=True)
    for name in ("stage1", "stage2", "pipeline", "cv", "ablate", "seeds"):
        tool = cmd.add_parser(name, parents=[common])
        tool.add_argument("--encoder", help="Stage-1 weights for Stage-2 training")
        tool.add_argument("--epochs", type=int, help="Override maximum number of epochs")
        if name == "pipeline":
            tool.add_argument("--test", action="store_true", help="Evaluate only the held-out test")
        if name == "cv":
            tool.add_argument("--stage", type=int, choices=(1, 2), default=2)
        if name == "ablate":
            tool.add_argument("--variants", nargs="+", default=["no_ssl", "no_rope", "no_fft"],
                              choices=list(ABLATIONS))
        if name == "seeds":
            tool.add_argument("--values", type=int, nargs="+", default=[42, 123, 2026])
    evaluation = cmd.add_parser("evaluate", parents=[common])
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--partition", choices=["val", "test"], default="test")
    transfer = cmd.add_parser("cross-dataset", parents=[common])
    transfer.add_argument("--checkpoint", required=True, help="Source dataset model weights")
    transfer.add_argument("--source-dataset", required=True, choices=["pure", "cohface", "ubfc"])
    robustness = cmd.add_parser("occlusion", parents=[common])
    robustness.add_argument("--checkpoint", required=True)
    robustness.add_argument("--fractions", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
    ssl = cmd.add_parser("ssl-plots", parents=[common])
    ssl.add_argument("--encoder", required=True)
    ssl.add_argument("--limit", type=int, default=256)
    motion = cmd.add_parser("motion", parents=[common])
    motion.add_argument("--checkpoint", required=True)
    sweep = cmd.add_parser("fft-sweep", parents=[common])
    sweep.add_argument("--encoder", required=True)
    sweep.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.5, 3.0])
    sweep.add_argument("--epochs", type=int)
    waveforms = cmd.add_parser("waveforms", parents=[common])
    waveforms.add_argument("--checkpoint", required=True)
    waveforms.add_argument("--examples", type=int, default=2)
    bootstrap = cmd.add_parser("bootstrap")
    bootstrap.add_argument("--session-csv", required=True)
    bootstrap.add_argument("--repetitions", type=int, default=10000)
    bootstrap.add_argument("--seed", type=int, default=42)
    bootstrap.add_argument("--output", required=True)
    profiling = cmd.add_parser("profile", parents=[common])
    profiling.add_argument("--checkpoint", help="Optional trained full-model weights")
    profiling.add_argument("--iterations", type=int, default=50)
    return p


def _paths(args, cfg):
    if not args.manifest or not args.split:
        raise ValueError("--manifest and --split are required for training/evaluation")
    rows = read_manifest(args.manifest)
    split = json.loads(Path(args.split).read_text())
    if split.get("dataset", cfg.dataset) != cfg.dataset:
        raise ValueError("Split dataset does not match config")
    if any(r["dataset"] != cfg.dataset for r in rows):
        raise ValueError("Manifest contains records from another dataset")
    audit_split(split)
    ids = {r["subject"] for r in rows}
    observed = set(split["train"] + split["val"] + split["test"])
    if observed != ids:
        raise ValueError("Split must cover exactly the manifest subject identities")
    return rows, split


def run(args):
    if args.command == "bootstrap":
        import csv
        with open(args.session_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        rows = [{**r, "pred_hr": float(r["pred_hr"]), "gt_hr": float(r["gt_hr"])} for r in rows]
        result = subject_bootstrap(rows, args.repetitions, args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    cfg = load_settings(args.config)
    if args.command == "preprocess":
        cfg = replace(cfg, raw_root=args.raw_root, predictor_path=args.predictor)
        print(preprocess_dataset(cfg, limit_sessions=args.limit_sessions))
        return
    if args.command == "split":
        if not args.manifest:
            raise ValueError("--manifest required")
        result = prepare_split(args.manifest, cfg.dataset, args.seed, args.frozen)
        save_split(result, args.write)
        print(json.dumps({"subjects": {key: len(result[key]) for key in ("train", "val", "test")},
                          "clips": result["counts"], "provenance": result["provenance"]}, indent=2))
        return
    if args.command == "profile":
        import torch
        from .stage2.model import RPPGModel
        from .experiments.profiling import profile_model
        model = RPPGModel(memory_blocks=cfg.memory_blocks, long_range_blocks=cfg.long_range_blocks)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location="cpu"))
        print(json.dumps(profile_model(model, iterations=args.iterations), indent=2))
        return
    rows, split = _paths(args, cfg)
    root = Path(args.output)
    if args.command == "waveforms":
        from .evaluation.waveforms import generate_waveform_figures
        stats = generate_waveform_figures(cfg, rows, split, args.checkpoint,
                                           root / cfg.dataset / "waveforms", examples=args.examples)
        print(json.dumps(stats, indent=2))
    elif args.command == "ssl-plots":
        from .experiments.ssl_figures import generate_ssl_figures
        stats = generate_ssl_figures(rows, split, args.encoder,
                                      destination=root / cfg.dataset / "ssl_plots", sample_limit=args.limit)
        print(json.dumps(stats, indent=2))
    elif args.command == "motion":
        if cfg.dataset != "pure":
            raise ValueError("The six-condition motion protocol is available only for PURE")
        from .experiments.motion import pure_motion_table
        _, sessions = evaluate(cfg, rows, split, args.checkpoint, partition="test")
        report = pure_motion_table(sessions)
        destination = root / cfg.dataset / "motion.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    elif args.command == "fft-sweep":
        for weight in args.weights:
            if weight < 0:
                raise ValueError("FFT loss weight must be nonnegative")
            variant = replace(cfg, fft_weight=weight, use_fft=(weight > 0))
            if args.epochs:
                variant = replace(variant, stage2_epochs=args.epochs)
            work = root / "fft_sweep" / f"weight_{weight:g}"
            weights = train_stage2(variant, rows, split, pretrained_encoder=args.encoder, out=work)
            stats, _ = evaluate(variant, rows, split, weights, partition="test",
                                out=work / "heldout")
            print(json.dumps({"fft_weight": weight, **stats}))
    elif args.command == "stage1":
        if args.epochs:
            cfg = replace(cfg, stage1_epochs=args.epochs)
        print(train_stage1(cfg, rows, split, out=root))
    elif args.command == "stage2":
        if args.epochs:
            cfg = replace(cfg, stage2_epochs=args.epochs)
        print(train_stage2(cfg, rows, split, pretrained_encoder=args.encoder, out=root))
    elif args.command == "pipeline":
        if args.epochs:
            cfg = replace(cfg, stage1_epochs=args.epochs, stage2_epochs=args.epochs)
        encoder = train_stage1(cfg, rows, split, out=root)
        weights = train_stage2(cfg, rows, split, pretrained_encoder=encoder, out=root)
        print(f"Trained Stage 2: {weights}")
        if args.test:
            result, _ = evaluate(cfg, rows, split, weights,
                                 out=root / cfg.dataset / "evaluation" / "heldout", partition="test")
            print(json.dumps(result, indent=2))
    elif args.command == "cv":
        if "folds" not in split:
            raise ValueError("Five-fold CV requires split JSON generated by `rppg split`")
        for item in split["folds"]:
            fold = {"train": item["train"], "val": item["val"], "test": split["test"]}
            # Each CV fold must use its own SSL encoder; never reuse a pretrainer
            # that was fit on that fold's validation subjects.
            fold_root = root / "cv" / cfg.dataset / f"fold_{item['fold']}"
            if args.epochs:
                cfg = replace(cfg, stage1_epochs=args.epochs, stage2_epochs=args.epochs)
            if args.stage == 1:
                train_stage1(cfg, rows, fold, out=fold_root)
            else:
                encoder = train_stage1(cfg, rows, fold, out=fold_root)
                weights = train_stage2(cfg, rows, fold, pretrained_encoder=encoder, out=fold_root)
                result, _ = evaluate(cfg, rows, fold, weights,
                                     partition="val", out=fold_root / "validation")
                print(json.dumps({"fold": item["fold"], **result}))
    elif args.command == "evaluate":
        result, _ = evaluate(cfg, rows, split, args.checkpoint, partition=args.partition,
                             out=root / cfg.dataset / "evaluation" / args.partition)
        print(json.dumps(result, indent=2))
    elif args.command == "cross-dataset":
        if args.source_dataset == cfg.dataset:
            raise ValueError("Cross-dataset evaluation requires a different target dataset")
        result, _ = evaluate(cfg, rows, split, args.checkpoint,
                             out=root / "cross_dataset" / f"{args.source_dataset}_to_{cfg.dataset}")
        result["source_dataset"] = args.source_dataset
        result["zero_shot"] = True
        print(json.dumps(result, indent=2))
    elif args.command == "occlusion":
        from .experiments.robustness import apply_occlusion
        for fraction in args.fractions:
            result, _ = evaluate(cfg, rows, split, args.checkpoint,
                out=root / cfg.dataset / "occlusion" / str(fraction),
                video_transform=lambda video, level=fraction: apply_occlusion(video, level))
            print(json.dumps({"occlusion": fraction, **result}))
    elif args.command == "ablate":
        for name in args.variants:
            variant = ablation_config(cfg, name)
            if args.epochs:
                variant = replace(variant, stage2_epochs=args.epochs)
            variant_root = root / "ablation" / name
            # An ablation can reuse the same pretrained encoder except no-SSL.
            if variant.use_ssl and not args.encoder:
                raise ValueError("--encoder is required for SSL-initialized ablations")
            model_path = train_stage2(variant, rows, split, pretrained_encoder=args.encoder,
                                      out=variant_root)
            result, _ = evaluate(variant, rows, split, model_path,
                                 out=variant_root / "heldout", partition="test")
            print(json.dumps({"ablation": name, **result}))
    elif args.command == "seeds":
        if not args.encoder:
            raise ValueError("--encoder fixed Stage-1 checkpoint required")
        records = []
        for seed in args.values:
            variant = replace(cfg, seed=seed)
            if args.epochs:
                variant = replace(variant, stage2_epochs=args.epochs)
            work = root / "seed_study"
            weights = train_stage2(variant, rows, split, pretrained_encoder=args.encoder,
                                   out=work)
            result, _ = evaluate(variant, rows, split, weights, partition="test",
                                 out=work / cfg.dataset / f"eval_{seed}")
            records.append(result)
        report = {"seeds": args.values, "results": records, "mean_and_sd": summarize_seeds(records)}
        destination = root / "seed_study" / cfg.dataset / "summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def main():
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
