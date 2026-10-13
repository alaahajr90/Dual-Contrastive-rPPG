"""Perform lightweight publication-preflight checks without touching private research data."""
from pathlib import Path
import json
import compileall

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility" / "reference_results.json"
SPLIT = ROOT / "reproducibility" / "splits" / "ubfc_archived_final.json"


def main():
    assert compileall.compile_dir(ROOT / "src", quiet=1)
    original = json.loads(RESULTS.read_text())
    members = json.loads(SPLIT.read_text())
    assert original["source_manuscript_protocol"]["test_fraction"] == 0.4
    assert sum(members[k + "_clips"] for k in ("train", "val", "test")) == 9529
    assert len({s for key in ("train", "val", "test") for s in members[key]}) == 42
    assert not any(ROOT.rglob("*.pth"))
    print("Repository preflight passed: source compiles, split and reference documents are consistent.")


if __name__ == "__main__":
    main()
