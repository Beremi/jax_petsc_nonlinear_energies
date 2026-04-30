# Paper Submission Checklist

## Publishability Verdict

Not yet publishable. The manuscript is scientifically close and now better
aligned with its evidence: major claims are scoped to repository-backed
workflow, SOTA positioning has been refreshed, citations build cleanly, and the
generated figure/table pipeline passes focused checks. Submission is still
blocked by missing target-journal metadata, license/archive decisions, and
shareable provenance gaps in several manuscript-critical artifacts. Validation
is adequate for the narrowed claims, but the paper must not be submitted until
the external submission metadata and archive-neutral provenance are fixed.

## Review Pass Scope

This pass inspected the manuscript sources, bibliography, literature manifest,
claim audit, generated tables and figures, paper generation scripts, LaTeX
build products, and supporting artifacts needed by the paper's main claims.
It also rechecked date-sensitive SOTA and citation metadata through live
arXiv, DOI, and official project pages on 2026-04-30. No long MPI campaigns
were rerun, no checked-in meshes or raw inputs were edited, and no new
scientific claims were introduced without supporting evidence.

## Changes Made In This Pass

- Updated `paper/references.bib`, `paper/literature/manifest.json`,
  `paper/literature/sources.md`, and `paper/literature/claim_audit.md` for
  JetSCI (`cattaneo2026jetsci`), Xue 2026 (`xue2026implicit`), current PETSc
  official citation metadata, richer Yashchuk arXiv metadata, and explicit
  Davis/Ginzburg/Sysala2017 evidence limits.
- Updated `paper/sections/introduction.tex` and
  `paper/sections/related_work.tex` to cover JetSCI and Xue 2026 without
  overstating novelty or source support.
- Reworded validation and performance claims in
  `paper/sections/abstract.tex`, `paper/sections/validation.tex`,
  `paper/sections/conclusion.tex`, `paper/sections/benchmarks.tex`, and
  `paper/sections/implementation.tex` so Plasticity3D gates, Plasticity2D
  fixed-work diagnostics, alternative Krylov/preconditioner diagnostics, and
  JAX-FEM comparison wording match the available evidence.
- Updated `paper/scripts/generate_paper_tables.py` and regenerated generated
  tables, including the SOTA comparison, the Plasticity3D validation summary
  with endpoint deviatoric-strain as a diagnostic row, the family highlights
  load-factor context, and the renamed fixed-source Plasticity3D table.
- Updated `paper/scripts/generate_paper_figures.py` and regenerated generated
  figures with paper notation such as `L_2` rather than implementation aliases
  such as `L1_2`.
- Updated `paper/scripts/validate_paper_assets.py` so required figures and
  generated tables are derived from the TeX sources and checked against the
  figure manifest.
- Updated `paper/scripts/generate_literature_sources.py` so the default command
  uses cached local full texts unless a required download is missing; the new
  `--refresh-downloads` flag forces a network refresh.
- Rebuilt the paper PDF through the paper generation pipeline.

## Blocking Issues Before Submission

- Issue: Target journal, template, and submission declarations are unresolved.
  Why it blocks publishability: the paper still uses a generic `article` class
  and cannot be submitted without journal-specific formatting, reference style,
  author metadata, funding, acknowledgements, and competing-interest
  declarations.
  Evidence path or citation: `paper/main.tex` front matter and
  `paper/sections/appendix.tex` availability note.
  Exact next action: choose the target journal, apply its template, and fill in
  ORCID, corresponding-author, funding, acknowledgements, data/software
  availability, and COI fields.
- Issue: Repository license and archival release/DOI are not decided.
  Why it blocks publishability: a software-methods submission needs an
  unambiguous license and a citable, durable version of the source/artifact
  snapshot.
  Evidence path or citation: no `LICENSE*` or `COPYING*` file is present at
  repository depth two; `paper/sections/appendix.tex` currently says an archive
  DOI should be supplied if required.
  Exact next action: add the chosen repository license, create the submission
  release or artifact archive, mint or record its DOI if applicable, and update
  the manuscript availability statement.
- Issue: Several manuscript-critical artifacts are not yet archive-neutral.
  Why it blocks publishability: current provenance contains local `tmp` source
  trees, absolute `/home/michal/...` commands, and local state paths that a
  reviewer cannot reproduce from a clean submission archive.
  Evidence path or citation:
  `artifacts/raw_results/plasticity3d_validation/validation_manifest.json`,
  `artifacts/raw_results/plasticity3d_derivative_ablation/comparison_summary.json`,
  `artifacts/raw_results/jax_fem_hyperelastic_baseline/comparison_summary.json`,
  and `artifacts/raw_results/jax_fem_hyperelastic_baseline/parity/*.json`.
  Exact next action: create a submission artifact bundle under
  `artifacts/reproduction/<submission-id>/` with relative paths, source
  snapshot hashes, command manifests, environment metadata, and corrected
  JAX-FEM comparison contract metadata; update paper scripts/manifests to point
  to that bundle.

## Major Revisions Needed

- Complete the journal-specific front matter and declarations once the target
  venue is chosen.
- Normalize the reproducibility archive so all paper-critical validation and
  comparison artifacts are shareable without private `tmp` checkouts or
  machine-local absolute paths.
- Decide whether locally cached full texts under `paper/literature/fulltext/`
  should remain an ignored private audit cache or become part of a controlled
  review artifact; do not imply public availability for restricted sources.
- Obtain accessible full text for Davis, Ginzburg, and Sysala2017 if the paper
  needs claims beyond the currently conservative metadata/context use.
- Keep Plasticity3D claims scoped to same-case source-faithfulness,
  fixed-lambda source-operator diagnostics, and endpoint-surrogate behavior
  unless a true incremental-history validation campaign is added.

## Minor Polish Items

- Visually inspect the rendered SOTA table after the JetSCI and Xue rows were
  added; the build passes, but the table is dense.
- Review regenerated figure labels in the final PDF for crowding after the
  notation cleanup.
- Revisit the availability statement after the archive/license decision so it
  reads like final submission metadata rather than a repository-local note.

## Claim And Citation Audit Summary

All active citation keys used by the manuscript now resolve during the LaTeX
build. Local full text or generated evidence was used where available; arXiv,
DOI, official documentation, or publisher metadata was used where local full
text was unavailable. JetSCI and Xue 2026 were added as current, materially
relevant SOTA sources. PETSc and Yashchuk metadata were refreshed. Davis,
Ginzburg, and Sysala2017 remain explicitly limited because full text was not
available in this pass.

Unsupported or overbroad paper claims found during the audit were corrected in
the manuscript rather than left as open tasks: Plasticity3D now separates gated
quantities from diagnostics, Plasticity2D L6/L7 is described as fixed-work
diagnostic evidence, the former deflated-GMRES wording is softened to
alternative Krylov/preconditioner diagnostics, and source-family Plasticity3D
language stays endpoint-surrogate only. The remaining claim/evidence risk is
not wording but reproducibility provenance for artifacts that still point to
local paths.

## SOTA Check Outcome

Related Work and the generated SOTA table needed changes and were updated.
JetSCI (`arXiv:2604.22087`, submitted 2026-04-23) is the newest directly
overlapping hybrid JAX+PETSc source found in this pass, and Xue 2026
(`Comput. Phys. Commun. 323:110102`, DOI `10.1016/j.cpc.2026.110102`) is a
peer-reviewed finite-element differentiable-physics source covering
second-order implicit differentiation. These are now reflected in the
manuscript, bibliography, literature manifest, sources table, claim audit, and
generated SOTA table.

Additional sources such as MFEM/dFEM, newer Firedrake differentiable-programming
bridges, and FormOpt were not added because they are useful context rather than
required support for the current scoped contribution.

## Validation And Reproducibility Checks

- `./.venv/bin/python paper/scripts/generate_literature_sources.py`: initially
  failed on a transient JOSS `503 Service Unavailable` while refreshing an
  already cached full text; after the cache-first generator fix, the exact
  command passed and generated `paper/literature/sources.md` with 24 public
  entries, 9 non-public local entries, and 3 unavailable entries.
- `./.venv/bin/python paper/scripts/generate_paper_tables.py`: passed.
- `./.venv/bin/python paper/scripts/generate_paper_figures.py`: passed and
  rewrote the figure manifest.
- `./.venv/bin/python paper/scripts/validate_paper_assets.py`: passed with 27
  figures and 18 generated tables checked.
- `make -C paper pdf`: passed after the script/table/figure changes and
  produced a 29-page PDF. A later rerun after the final prose cleanup was
  terminated after `generate_paper_figures.py` stalled for more than three
  minutes inside a `luatex --luaonly ... kpsewhich.lua` subprocess; the
  existing regenerated assets still passed validation.
- `(cd paper && latexmk -pdf main.tex)`: passed after the generated table rename
  and again after the final prose cleanup, refreshing `paper/build/main.pdf`.
- `./.venv/bin/python -m pytest tests/test_docs_publication.py`: passed
  13 tests.
- `./.venv/bin/python -m pytest tests/test_final_report_figure_generators.py`:
  passed 3 tests.
- `rg -n "TODO|FIXME|placeholder|constitutively equivalent|validated incremental|P4\\(L1|local_constitutiveAD|sourcefixed" paper/main.tex paper/sections paper/tables/generated paper/literature`:
  no matches.
- `rg -n "Warning|Citation|undefined|Overfull" paper/build/main.log paper/build/main.blg`:
  no matches in the final build logs.
- `git diff --check`: passed.

Exact remaining blockers are submission metadata, license/archive DOI, and
archive-neutral provenance. The only command instability observed in the final
rerun was the intermittent TeX/font lookup stall in the Makefile figure target;
direct asset validation and direct LaTeX rebuild are clean.

## Optional Future Work

- Add a true incremental-history Mohr-Coulomb Plasticity3D validation campaign.
  This would support stronger mechanics claims but is not required for the
  current endpoint-surrogate workflow paper.
- Add MFEM/dFEM, newer Firedrake differentiable-programming bridge, and FormOpt
  discussion if the paper expands from the current scoped positioning.
- Add more fairness-first external baselines on tightly matched problem
  contracts.
