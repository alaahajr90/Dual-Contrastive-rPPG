# Research Code Release Checklist (Results in Engineering submission companion)

This is a recommended transparent research-code package, **not** an assertion that the journal requires a particular GitHub folder layout. Verify the journal's current *Guide for Authors* and research-data policy before submission.

- [x] Provide a descriptive README with the manuscript title and method scope.
- [x] Separate PURE, COHFACE and UBFC readers from shared training logic.
- [x] Centralize hyperparameters and flags; avoid machine-specific absolute paths.
- [x] Document 60/40 subject-level split, five-fold CV, saved UBFC membership and PURE held-out IDs.
- [x] Make missing COHFACE subject membership explicit instead of inventing original IDs.
- [x] Provide commands for the two-stage pipeline and standalone data, model, analysis steps.
- [x] Include leakage assertions and tests runnable without private videos.
- [x] Provide source-level ablations, multi-seed analysis, subject bootstrap and zero-shot transfer.
- [x] Exclude acquired videos, ground-truth physiological signals and trained checkpoint files.
- [x] Separate manuscript reference numbers from the output of a newly executed experiment.
- [x] Document model and reported-parameter discrepancy without concealing it.
- [ ] Obtain the exact PURE and COHFACE archived training implementations and frozen COHFACE IDs.
- [ ] Verify head dimensions, 4/6 attention discrepancy and the source model with 546,532 parameters.
- [ ] Verify all results with original dataset acquisition and final hyperparameter versions.
- [ ] Record full CPU/GPU/CUDA/PyTorch/library versions from the reproduction environment.
- [ ] Verify all public dataset access/redistribution conditions and credit dataset creators.
- [ ] Have the authors choose and approve a source-code LICENSE before public release.
- [ ] Create a stable Git tag and optionally an archived release DOI (e.g. Zenodo).
- [ ] Check the journal's current code/data availability expectations and submit the correct repository URL.

## Suggested manuscript code-availability statement

"The source code and experiment configurations for the two-stage rPPG pipeline are available in a public GitHub repository at [repository URL to be supplied]. The repository documents data preprocessing, subject-level partitions, contrastive pretraining, supervised fine-tuning and evaluation procedures. The publicly distributed package excludes original facial video data and model checkpoints. Access to the source datasets is subject to their respective dataset providers' conditions."

**Modify this statement if the repository is not yet public.** Do not claim full numerical reproducibility while the exceptions in `AUDIT.md` remain unresolved.

## Policy references

- Elsevier research data: https://www.elsevier.com/researcher/author/tools-and-resources/research-data
- Elsevier research data FAQ: https://www.elsevier.com/about/policies-and-standards/research-data/faq
- Read the current journal-specific *Results in Engineering* author guide before making journal-specific compliance claims.
