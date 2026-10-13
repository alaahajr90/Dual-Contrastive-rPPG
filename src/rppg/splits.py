"""Frozen subject-level partitions with train-only cross validation and strict audits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold


def split_subjects(subjects, *, dataset: str, seed: int = 42, frozen_path=None):
    subjects = sorted(set(str(s) for s in subjects))
    if not subjects:
        raise ValueError("No subjects found; check the preprocessing manifest")
    if frozen_path:
        frozen = json.loads(Path(frozen_path).read_text())
        available = set(subjects)
        if frozen.get("dataset") != dataset:
            raise ValueError("Frozen split is for a different dataset")
        train = list(map(str, frozen["train"]))
        val = list(map(str, frozen["val"]))
        test = list(map(str, frozen["test"]))
        if set(train + val + test) != available:
            raise ValueError("Frozen split subjects do not exactly match processed subjects")
        audit_split({"train": train, "val": val, "test": test})
        return {"train": train, "val": val, "test": test, "provenance": "archived"}
    n_test = int(round(0.4 * len(subjects)))
    if len(subjects) < 5 or n_test <= 0:
        raise ValueError("Insufficient subjects for five-fold development CV")
    if dataset == "pure":
        test = ["02", "05", "09", "10"]
        if not set(test).issubset(subjects) or len(subjects) != 10:
            raise ValueError("PURE expected all ten original subjects 01-10")
        dev = sorted(set(subjects) - set(test))
        provenance = "manuscript_pure_heldout"
    else:
        rng = np.random.default_rng(seed)
        shuffled = list(rng.permutation(subjects))
        test, dev = sorted(shuffled[:n_test]), sorted(shuffled[n_test:])
        provenance = "generated_seed_not_original_archived"
    # Single deterministic fold is the final validation choice. All five folds
    # can be trained separately through the CV command.
    val = make_folds(dev, 5, seed)[0]["val"]
    train = sorted(set(dev) - set(val))
    splits = {"train": train, "val": val, "test": test, "provenance": provenance}
    audit_split(splits)
    return splits


def make_folds(dev_subjects, n_folds=5, seed=42):
    subs = np.array(sorted(set(dev_subjects)))
    if len(subs) < n_folds:
        raise ValueError("Number of development subjects is smaller than number of folds")
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [{"fold": i, "train": subs[tr].tolist(), "val": subs[va].tolist()}
            for i, (tr, va) in enumerate(splitter.split(subs))]


def audit_split(splits):
    sets = {k: set(splits[k]) for k in ("train", "val", "test")}
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("LEAKAGE: a subject is present in more than one partition")
    if len(set.union(*sets.values())) != sum(len(s) for s in sets.values()):
        raise ValueError("Duplicate subject assignment")
    return {k: len(v) for k, v in sets.items()}


def prepare_split(manifest, dataset, seed=42, frozen_path=None):
    from .preprocessing.manifest import read_manifest
    clips = read_manifest(manifest)
    ids = sorted({r["subject"] for r in clips})
    splits = split_subjects(ids, dataset=dataset, seed=seed, frozen_path=frozen_path)
    dev = sorted(set(splits["train"]) | set(splits["val"]))
    splits["folds"] = make_folds(dev, 5, seed)
    splits["dataset"] = dataset
    splits["seed"] = seed
    splits["counts"] = {key: sum(r["subject"] in set(splits[key]) for r in clips)
                        for key in ("train", "val", "test")}
    if frozen_path:
        frozen = json.loads(Path(frozen_path).read_text())
        for part in ("train", "val", "test"):
            expected = frozen.get(f"{part}_clips")
            if expected is not None and splits["counts"][part] != expected:
                raise ValueError(
                    f"Frozen archive has {expected} {part} clips but preprocessing generated "
                    f"{splits['counts'][part]}. Reconcile extraction and data completeness "
                    "before running a manuscript-reproduction experiment."
                )
    return splits


def save_split(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
