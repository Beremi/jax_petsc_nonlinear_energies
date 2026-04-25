# Paper Submission TODO

This checklist tracks what remains before submitting the manuscript as a
scientific-computing/software-methods journal paper. The manuscript claim is
deliberately scoped: `fenics_nonlinear_energies` is presented as a maintained
JAX+PETSc workflow with repository-backed benchmark evidence, not as a new
constitutive theory or a replacement for incremental Mohr-Coulomb plasticity.

## Current State

- The paper uses the generic `article` class and is aimed at a scientific
  computing / software-methods journal.
- The front matter is populated for Michal B{\'e}re{\v s} with the VSB and
  Institute of Geonics affiliations visible in the related Sysala papers.
- The active manuscript cites the relevant Sysala geomechanics line:
  return mapping, SSR variational principles, 3D continuation, and geotechnical
  convex optimization.
- The Plasticity3D text treats the maintained model as an endpoint surrogate and
  keeps source-faithfulness separate from incremental-history equivalence.
- Generated figures and tables are expected to remain the source of truth for
  numerical statements.

## Completed Evidence Already In The Draft

- Plasticity3D same-case source-faithfulness layer:
  work relative difference `3.877e-05`, displacement relative L2 `3.517e-03`,
  and deviatoric-strain relative L2 `8.720e-03`.
- Plasticity3D fixed-lambda source-operator layer:
  highest-successful-lambda proxy agrees, but field thresholds fail; the draft
  therefore does not claim constitutive equivalence.
- Plasticity3D derivative-route ablation:
  element AD, constitutive AD, and colored SFD converge to the same locked
  state; constitutive AD is the preferred cost-quality route.
- Hyperelasticity companion baseline against JAX-FEM:
  same mesh, material law, and load schedule; the fairness gate passes.
- Plasticity3D historical distributed timing:
  promoted `P4(L1_2)` local-constitutiveAD + local-PMG campaign reports
  `18.053x` end-to-end speedup at 32 MPI ranks relative to one rank.
- Topology workflow evidence:
  the parallel design-mechanics run remains operational at `768x384` and
  32 ranks.

## Blocking Before Submission

1. Choose the exact journal and apply its final formatting requirements.
   The current draft intentionally remains journal-neutral.
2. Confirm final author metadata:
   ORCID, corresponding-author email, funding statement, acknowledgements,
   conflict-of-interest statement, and data/code availability wording.
3. Decide whether the repository needs an archived release, Zenodo DOI, or
   separate artifact DOI for the submitted version.
4. Run a final citation-locator audit:
   every externally sourced scientific statement should map to
   `paper/literature/claim_audit.md`.
5. Rebuild the PDF from a clean checkout or clean paper build directory and
   inspect the resulting log for undefined references, undefined citations, and
   serious overfull boxes.

## Manuscript Polish Remaining

- Re-read the abstract, introduction, discussion, and conclusion for any sentence
  that sounds broader than the repository evidence.
- Keep framework comparisons factual:
  no broad speed rankings between JAX+PETSc, JAX-FEM, FEniCS, DOLFINx, cashocs,
  AutoPDEx, FEniTop, or external-operator workflows.
- Keep Plasticity3D phrasing consistent:
  use "endpoint surrogate", "source-faithfulness", and "fixed-lambda
  source-operator comparison"; avoid "validated incremental plasticity solver"
  or equivalent language.
- Check all figure and table captions for claim scope, especially where
  historical timing evidence is adjacent to glued-bottom `lambda = 1.55`
  benchmark evidence.
- Add a concise reproducibility paragraph or note if required by the target
  journal.

## Verification Commands

Run these before the submission snapshot:

```bash
./.venv/bin/python paper/scripts/generate_literature_sources.py
./.venv/bin/python paper/scripts/validate_paper_assets.py
(cd paper && latexmk -pdf main.tex)
rg -n "TODO|FIXME|Manuscript draft|placeholder|constitutively equivalent|validated incremental" paper/main.tex paper/sections paper/literature
rg -n "Warning|Citation|undefined|Overfull" paper/build/main.log paper/build/main.blg
```

If any script that generates tables, figures, or literature sources changes,
also run the focused tests for that script. A full experiment rerun is not part
of the submission-prep checklist unless the manuscript begins to depend on new
numerical evidence.

## Optional Future Work

- Add a true incremental-history Mohr-Coulomb reference comparison on a small
  Plasticity3D case. This would support stronger mechanics claims, but it is not
  required for the current scoped software-and-methods submission.
- Add more fairness-first external baselines on tightly matched problem
  contracts.
- Archive a minimal reproducer bundle for the main paper figures and tables.
