# Paper Submission Checklist

This checklist tracks what remains before submitting the manuscript as a
scientific-computing/software-methods journal paper. The manuscript claim is
deliberately scoped: `fenics_nonlinear_energies` is presented as a maintained
JAX+PETSc workflow with repository-backed benchmark evidence, not as a new
constitutive theory or a replacement for incremental Mohr-Coulomb plasticity.

## Repository-Backed State

- The paper stays on the generic `article` class and remains journal-neutral.
- The front matter is populated for Michal B{\'e}re{\v{s}} with the VSB and
  Institute of Geonics affiliations visible in the related Sysala papers.
- The active manuscript cites the relevant Sysala geomechanics line:
  return mapping, SSR variational principles, 3D continuation, and geotechnical
  convex optimization.
- The Plasticity3D text treats the maintained model as an endpoint surrogate and
  keeps source-faithfulness separate from incremental-history equivalence.
- Generated figures and tables remain the source of truth for numerical
  statements.
- The appendix now includes a neutral code, data, and reproducibility
  availability note tied to the GitHub repository, `paper/scripts`, generated
  table/figure assets, and local `artifacts/raw_results` summaries.
- The citation-locator audit currently covers every cited scientific source in
  `paper/literature/claim_audit.md`.

## Completed Evidence Already In The Draft

- Plasticity3D same-case source-faithfulness layer:
  work relative difference `3.877e-05`, displacement relative L2 `3.517e-03`,
  and deviatoric-strain relative L2 `8.720e-03`.
- Plasticity3D fixed-lambda source-operator layer:
  highest-successful-lambda proxy agrees, but field thresholds fail; the draft
  therefore does not claim constitutive equivalence.
- Plasticity3D derivative-route ablation:
  element AD, constitutive AD, and colored SFD converge to the same locked
  state; constitutive AD has the lowest median wall time on that locked case.
- Hyperelasticity companion baseline against JAX-FEM:
  same mesh, material law, and load schedule; the fairness gate passes.
- Plasticity3D historical distributed timing:
  promoted `P4(L1_2)` local-constitutiveAD + local-PMG campaign reports
  `18.053x` end-to-end speedup at 32 MPI ranks relative to one rank.
- Topology workflow evidence:
  the parallel design-mechanics run remains operational at `768x384` and
  32 ranks.

## External Decisions Still Blocking Submission

1. Choose the exact target journal and apply its template, length, figure,
   reference-style, and declaration requirements.
2. Confirm final author metadata: ORCID, corresponding-author email, postal
   affiliation formatting, and any author-note requirements.
3. Supply final funding, acknowledgements, and conflict-of-interest /
   competing-interest declarations. The repository does not contain enough
   information to state these safely.
4. Choose and record the project license. No repository license file is present
   in this checkout.
5. Decide whether to create an archived source release, Zenodo record, or
   separate artifact DOI for the submitted version, and add the resulting DOI
   to the manuscript metadata if required by the journal.

## Final Pre-Submission Checks

- Rebuild the PDF from a clean checkout or clean paper build directory.
- Inspect the resulting log for undefined references, undefined citations, and
  serious overfull boxes.
- Re-run the citation/source checks after any bibliography or related-work
  edits.
- Re-run focused tests only if a script that generates tables, figures, or
  literature sources changes.

## Verification Commands

Run these before the submission snapshot:

```bash
./.venv/bin/python paper/scripts/generate_literature_sources.py
./.venv/bin/python paper/scripts/validate_paper_assets.py
(cd paper && latexmk -pdf main.tex)
rg -n "TODO|FIXME|Manuscript draft|placeholder|constitutively equivalent|validated incremental" paper/main.tex paper/sections paper/literature
rg -n "Warning|Citation|undefined|Overfull" paper/build/main.log paper/build/main.blg
git diff --check
```

A full experiment rerun is not part of the submission-prep checklist unless the
manuscript begins to depend on new numerical evidence.

## Optional Future Work

- Add a true incremental-history Mohr-Coulomb reference comparison on a small
  Plasticity3D case. This would support stronger mechanics claims, but it is not
  required for the current scoped software-and-methods submission.
- Add more fairness-first external baselines on tightly matched problem
  contracts.
- Archive a minimal reproducer bundle for the main paper figures and tables.
