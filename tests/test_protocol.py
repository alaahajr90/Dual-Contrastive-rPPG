import json
from pathlib import Path
import pytest
from rppg.splits import split_subjects, audit_split, make_folds


def test_pure_reproduces_four_heldout_subjects():
    ids = [f"{i:02d}" for i in range(1, 11)]
    split = split_subjects(ids, dataset="pure")
    assert split["test"] == ["02", "05", "09", "10"]
    assert len(split["train"] + split["val"]) == 6
    assert len(split["test"]) == 4
    folds = make_folds(split["train"] + split["val"])
    assert len(folds) == 5
    assert set().union(*(set(f["val"]) for f in folds)) == set(ids) - set(split["test"])


def test_cohface_24_16_and_folds():
    ids = [str(i) for i in range(1, 41)]
    split = split_subjects(ids, dataset="cohface")
    assert len(split["test"]) == 16
    assert len(split["train"]) + len(split["val"]) == 24
    assert audit_split(split) == {"train": 19, "val": 5, "test": 16}


def test_ubfc_archived_membership():
    path = Path(__file__).parents[1] / "reproducibility" / "splits" / "ubfc_archived_final.json"
    info = json.loads(path.read_text())
    assert info["train_clips"] == 4611
    assert info["val_clips"] == 1179
    assert info["test_clips"] == 3739
    assert sum(len(info[k]) for k in ("train", "val", "test")) == 42
    assert len(info["test"]) == 17
    split = split_subjects(info["train"] + info["val"] + info["test"],
                           dataset="ubfc", frozen_path=path)
    assert audit_split(split) == {"train": 20, "val": 5, "test": 17}


def test_overlap_fails():
    with pytest.raises(ValueError, match="LEAKAGE"):
        audit_split({"train": ["person1"], "val": ["person1"], "test": ["person2"]})
