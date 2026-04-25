# Paper Submission TODO

## Current Position

This project looks publishable as a strong software-and-methods paper in scientific computing / computational mechanics. It is less convincing if framed as a broad mechanics breakthrough or as a theory-first numerical analysis paper. The best current thesis is:

`A solver-centric JAX + PETSc workflow for distributed nonlinear finite-element energy minimization, with multiple derivative routes and a reproducible benchmark suite across several difficult problem families.`

The goal of the next revision should be to make that claim precise, well-supported, and easy for a reviewer to accept.

## Main Weak Points To Fix

1. The novelty boundary is still vulnerable.
   Reviewers can still ask what is fundamentally new relative to `JAX-FEM`, `AutoPDEx`, `cashocs`, `FEniTop`, `dolfin-adjoint` / `pyadjoint`, and newer `FEniCSx`-based workflows.

2. The paper is broad, but some stories are not yet deep enough.
   The benchmark spread is good, but the evidence per benchmark family is still uneven. The paper needs one or two decisive main-text results, not only broad coverage.

3. Some comparisons are not fully symmetric.
   Different load factors, historical campaigns, end-to-end timings, and not-always-identical solver settings create easy reviewer objections.

4. The plasticity story is interesting, but also the most fragile.
   The paper correctly describes the repository formulations as surrogate scalarizations, but it still needs one compact validation step that shows what these surrogates preserve relative to a standard incremental reference.

5. The results are currently more descriptive than decisive.
   The paper needs at least one flagship ablation table and one clean cross-framework comparison on a shared benchmark.

6. The venue positioning is not fixed enough yet.
   Right now the draft could be read as software, scientific computing, optimization, or mechanics. That flexibility is useful, but it weakens the submission strategy if the paper text is not tailored to one main venue family.

## Step-By-Step Improvement Plan

### Phase 1: Lock the thesis and target venue

1. Freeze the main claim in one sentence.
   Why: every section should support the same claim, and right now the draft still allows broader readings than the evidence fully supports.
   Output: one sentence inserted consistently into the abstract, end of the introduction, and conclusion.
   Files: `paper/sections/abstract.tex`, `paper/sections/introduction.tex`, `paper/sections/conclusion.tex`

2. Choose one flagship benchmark family and one secondary support story.
   Recommended default:
   `Plasticity3D` as the flagship solver-and-derivative story.
   Topology optimization as the secondary workflow / optimization story.
   Why: these are the most distinctive parts of the repository.
   Output: a short note at the top of `results.tex` and a re-ordered figure / table flow.
   Files: `paper/sections/results.tex`, `paper/sections/discussion.tex`, `paper/sections/appendix.tex`

3. Decide the primary venue track before further rewriting.
   Default venue track:
   scientific computing / computational methods.
   Alternate venue track:
   computational mechanics, only if the mechanics comparisons are strengthened.
   Why: the abstract, title, and contribution paragraph should be venue-shaped.
   Output: one chosen venue family and one backup family written at the top of this TODO when the decision is made.

4. Rewrite the title and abstract only after steps 1-3 are frozen.
   Why: doing this earlier causes churn.
   Output: title options tailored to the selected venue family.
   Files: `paper/main.tex`, `paper/sections/abstract.tex`

### Phase 2: Strengthen the core evidence

1. Add one flagship ablation table on a single benchmark with fully matched conditions.
   Recommended benchmark: `Plasticity3D`.
   Compare exactly:
   element AD, constitutive AD, colored SFD.
   Hold fixed:
   mesh family, rank count, nonlinear tolerances, line search / trust-region settings, linear solver settings, load factor, stopping rules, and hardware.
   Report at minimum:
   total wall time, assembly time, nonlinear iterations, linear iterations, final residual norm, final objective / energy, and memory if available.
   Why: this is the single highest-ROI addition for reviewer confidence.
   Output: one main-text table and one appendix table.
   Files: `paper/sections/results.tex`, `paper/sections/appendix.tex`, `paper/tables/generated/`, `paper/scripts/generate_paper_tables.py`

2. Add one external baseline on a benchmark that can be reproduced fairly.
   Good candidates:
   `JAX-FEM` on a shared small nonlinear mechanics problem.
   `AutoPDEx` on a JAX-native PDE problem.
   `FEniTop` on a topology problem.
   Rule: do not force a weak baseline. Only include a direct comparison if the benchmark, mesh, objective, and stopping rules can be matched closely enough to survive reviewer scrutiny.
   Why: one good external baseline is more valuable than many loose comparisons.
   Output: one compact comparison table and one short fairness paragraph.
   Files: `paper/sections/related_work.tex`, `paper/sections/results.tex`, `paper/sections/discussion.tex`

3. Add a plasticity surrogate validation subsection.
   Minimum version:
   take a smaller plasticity case and compare the repository surrogate objective path against a standard incremental elastoplastic reference on observable quantities such as load-displacement trend, localization pattern, and final state statistics.
   Why: this closes the most likely scientific-rigor objection.
   Output: one figure or table plus a short explanatory paragraph.
   Files: `paper/sections/benchmarks.tex`, `paper/sections/results.tex`, `paper/sections/discussion.tex`

### 2.3A Full Validation Plan For The Plasticity3D Surrogate

Objective:
validate the current Plasticity3D scalar surrogate in the strongest publishable way without overstating what is already proven. The target is now a three-part ladder:

1. `Layer 1A Exact source-faithfulness`:
   show that the maintained `JAX + PETSc` implementation reproduces the same-case source-style Plasticity3D workflow under matched conditions.
2. `Layer 1B Published source-family triangulation`:
   connect the maintained implementation to the published 3D slope-stability source family, including the recent `CAS 2025` paper and the related MATLAB/PETSc `COMSOL`-backed benchmark lineage.
3. `Layer 2 Incremental-reference validation` (future/strongest version):
   show that the surrogate remains scientifically credible when compared against a standard rate-independent incremental Mohr-Coulomb elastoplastic reference with history updates.

Defaults for this validation package:

- anchor benchmark: `Plasticity3D`
- ambition level: `full strongest`
- canonical validation case: coarse glued-bottom `P2(L1)` on the imported heterogeneous 3D slope
- common cross-layer reporting schedule:
  `lambda = 1.0, 1.2, 1.4, 1.5, 1.55, 1.6`
- authoritative `Layer 1A` direct-branch schedule from the already completed Octave/JAX comparison:
  `[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]`

Why this is required:
the current repository benchmark is an endpoint scalar surrogate, not a path-consistent incremental elastoplastic history solve. That is already stated honestly in the paper, but reviewer-facing rigor requires a separate validation layer before the manuscript can make broader constitutive claims. The relevant literature also supports this split:

- non-associated slope plasticity can be numerically difficult in strength-reduction analysis, and Davis-style modifications are used precisely to stabilize and interpret the problem
- standard incremental elastoplastic validation is tied to the constitutive integration algorithm and its tangent, not only to endpoint energy agreement
- recent constitutive-software papers expect explicit numerical verification, not only visual similarity
- the new `CAS 2025` and `SIOPT 2025` papers strengthen the source-lineage and optimization-family grounding, but they do not erase the difference between source-faithfulness and true incremental constitutive validation

Step 1: lock the exact validation contract before running anything.

Use the same geometry, raw materials, glued-bottom boundary rule, gravity direction, and Davis-B reduction across all validation layers. Do not allow silent changes in stopping rules, boundary handling, or reduction formulas between comparisons. The validation package should explicitly write these inputs into one small run manifest so that every comparison row can be audited later.

Step 2: formalize `Layer 1A` as a near-complete source-faithfulness package instead of treating it as missing work.

The repository already has same-case source-comparison evidence and it should be elevated rather than rebuilt:

- docs evidence:
  `docs/problems/Plasticity3D.md`
- runner scaffold:
  `experiments/runners/run_plasticity3d_p4_l1_lambda1p5_source_compare.py`
- test scaffold:
  `tests/test_plasticity3d_source_compare_runner.py`

Lock the default `Layer 1A` evidence package to the existing Octave/JAX direct-branch comparison:

- case: glued-bottom `P2(L1)`
- same imported mesh
- same material map
- same gravity axis
- same Davis-B reduction
- same stop policy
- accepted direct-branch schedule:
  `[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]`

Treat these existing final differences as the baseline source-faithfulness evidence that already exists and must be packaged cleanly into the paper:

- work relative difference:
  `3.877021e-05`
- displacement-field relative `L2` difference:
  `3.517247e-03`
- deviatoric-strain relative `L2` difference:
  `8.720006e-03`

The remaining tasks for `Layer 1A` are packaging, stabilization, and manuscript insertion, not proving fidelity from scratch. The paper-grade artifact set should include:

- final `energy / work` values
- `omega`
- `u_max`
- nonlinear history
- final displacement-field relative `L2`
- final deviatoric-strain-slice relative `L2`
- one visual overlay of the deformed boundary
- three visual overlays for the `x / y / z` strain slices

Completion rule for `Layer 1A`:
this layer is not finished until the already existing comparison is promoted from docs annex status into a paper-usable artifact set:

- one stored JSON summary
- one stable runner
- one stable asset generator
- one table or figure that can be referenced from the paper

Step 3: add `Layer 1B` as published source-family triangulation, clearly labeled as context rather than exact case equivalence.

Use the following published-source bridge:

- paper:
  `Sysala et al. 2025, Computers & Structures`
- repository context:
  `tmp/source_compare/slope_stability_octave_ref`
- maintained source-family benchmark:
  `tmp/source_compare/slope_stability_petsc4py/benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/archive/report.md`

This layer is valuable because it ties the maintained repository to a recent published 3D slope-stability line that:

- uses Davis-based `SSR` continuation on Mohr-Coulomb plasticity
- studies advanced nonlinear / Krylov solver behavior in 3D
- includes mesh-adapted 3D slope workflows
- explicitly states that numerical results are validated against literature or `COMSOL Multiphysics`

Use the maintained PETSc/MATLAB `COMSOL`-backed report as related-case quantitative context. The current stored comparison already reports:

- relative `lambda` history error:
  `2.144e-05`
- relative `omega` history error:
  `2.887e-04`
- relative `Umax` history error:
  `3.190e-03`

However, this layer must be labeled carefully:

- classify it as `Context / published source-family evidence`
- do not present it as the same thing as the glued-bottom imported `Plasticity3D` benchmark
- do not treat seepage+`COMSOL` agreement as a substitute for same-case constitutive validation

The manuscript value of `Layer 1B` is to show that the maintained code is not isolated: it already sits on top of a published, validated, open-source source family with both literature-backed and `COMSOL`-backed 3D evidence.

Step 4: keep the future strongest `Layer 2` as the true incremental-reference comparison.

The current manuscript revision uses a fixed-lambda source-operator comparison
as Layer 2 evidence. That comparison is useful for source-family behavior, but
it is not a path-consistent incremental-history solve and should not be
described as one.

This is the scientifically important new work. Add one new small reference problem that uses a standard rate-independent incremental Mohr-Coulomb formulation with:

- internal variables
- an implicit constitutive update
- a constitutive algorithm / tangent appropriate for incremental elastoplasticity

This reference must not reuse the repository's zero-history placeholder logic. It must be an incremental history solve, not another endpoint scalar surrogate.

Keep this layer bounded:

- use the same canonical coarse `P2(L1)` case
- keep it local and small
- do not try to scale it to the full flagship `P4(L1_2)` campaign

To keep the constitutive comparison apples-to-apples, the reference problem must use:

- the same raw material parameters
- the same geometry
- the same glued-bottom rule
- the same gravity direction
- the same Davis-B reduced parameters at each `lambda`

The reference may use its own algorithmic tangent or return-mapping machinery, but it must remain within the same Mohr-Coulomb strength-reduction family. A good target is an implicit return-mapping / semismooth-Newton style reference, because that aligns with the standard incremental viewpoint described by Simo-Taylor and by the Mohr-Coulomb return-mapping literature.

The key reviewer-facing rule remains:

- `CAS 2025` plus the source-family `COMSOL` benchmark materially strengthen the validation story
- they do **not** justify dropping `Layer 2` while the paper still wants strong constitutive claims
- only if the manuscript claim is intentionally narrowed to source-faithfulness plus optimization-family relevance could `Layer 2` be deferred

Step 5: define one shared observable set and refuse to compare implementation-specific internals.

For Layer 2, compare only observables that both formulations can provide cleanly:

- critical `lambda` or `FoS`
- `u_max(lambda)`
- one boundary-displacement profile
- final displacement field on common sample points
- final deviatoric-strain slices on common `x / y / z` planes

Do not require equality of:

- plastic multipliers
- branch labels
- internal implementation diagnostics

Those are implementation-specific and would create false negatives.

Before computing field differences, resample both outputs to the same comparison set:

- one common boundary sample path for displacement
- one common interior sample grid for displacement-field norms
- one common sampling rule for the `x / y / z` deviatoric-strain slices

Step 6: lock quantitative acceptance criteria before any manuscript wording is updated.

Hard acceptance targets:

- `FoS` or critical `lambda` relative difference at most `3%`
- `u_max(lambda)` curve relative `L2` error at most `5%`
- endpoint displacement-field relative `L2` error at most `10%` on the common sample grid

Visual corroboration rule:
the dominant high-strain localization region must appear in the same physical part of the slope on the `x / y / z` comparison slices.

Manuscript rule:
if Layer 2 disagrees materially with the surrogate, the paper must explicitly present the surrogate as a repository-specific approximation rather than as a constitutively equivalent replacement.

Step 7: implement the minimum support tooling needed for the validation package.

Add:

- one new runner for the coarse incremental-reference campaign
- one comparison script that ingests maintained-surrogate output and incremental-reference output
- common-grid or common-slice resampling inside that script
- one JSON summary writer for shared metrics
- one paper-ready figure generator or short appendix-table generator entry

The comparison script should write:

- the run manifest
- the shared metrics
- the field-difference metrics
- paths to the generated overlays and slice comparisons

Step 8: add tests so the validation package is stable enough for paper use.

Minimum tests:

- one smoke test for the new comparison-summary schema
- one smoke test for the asset generator on synthetic or tiny prepared data
- one regression-style check that shared-sampling logic produces the expected output shapes

If the incremental reference relies on external or legacy code, test the normalization and comparison layer even if the full reference run is too expensive for CI.

Step 9: integrate the validation result into the manuscript only after all required layers exist.

Paper integration tasks:

- add one benchmark paragraph in `benchmarks.tex` explaining that the Plasticity3D surrogate is validated through exact source-faithfulness, published source-family triangulation, and a separate fixed-lambda source-operator check
- add one short validation paragraph plus one figure or table in `results.tex`
- add one limitations sentence in `discussion.tex` explaining that source-faithfulness, source-family triangulation, source-operator agreement, and future incremental-reference agreement are distinct claims
- add the supporting sources and exact claim locators to `paper/literature/claim_audit.md`

Step 10: define `2.3` as incomplete until the full package exists.

`2.3` is done only when:

- `Layer 1A`, `Layer 1B`, and `Layer 2` all exist in their intended roles
- the paper-facing layers are reproducible from documented commands
- the relevant comparisons have stored summary artifacts
- the paper contains one explicit validation result instead of only a prose promise

Scientific grounding for these directions:

- `Tschuchnigg, Schweiger, Sloan 2015`:
  Davis-style modifications are used to handle non-associated slope plasticity and to obtain accurate `FoS` estimates in difficult cases.
  Source:
  <https://www.researchgate.net/publication/281931410_Slope_stability_analysis_by_means_of_finite_element_limit_analysis_and_finite_element_strength_reduction_techniques_Part_I_Numerical_studies_considering_non-associated_plasticity>

- `Oberhollenzer, Tschuchnigg, Schweiger 2018`:
  `SRFEA` with non-associated plasticity can be unstable, and modified Davis procedures can make `SRFEA` and `FELA` give very similar `FoS` values.
  Source:
  <https://www.sciencedirect.com/science/article/pii/S1674775518302129>

- `Simo and Taylor 1985`:
  standard incremental elastoplastic validation should be tied to the constitutive integration algorithm and its consistent tangent, not just to endpoint energies.
  Source:
  <https://www.sciencedirect.com/science/article/pii/0045782585900702>

- `Sysala, Cermak, Ligursky 2015/2017`:
  Mohr-Coulomb slope stability can be treated with implicit return-mapping, consistent tangents, and 2D/3D benchmark implementations.
  Source:
  <https://arxiv.org/abs/1508.07435>

- `Sysala et al. 2021`:
  optimization-based shear-strength-reduction variants exist and Davis-type modifications can be interpreted within that family, which supports treating the repository surrogate as scientifically meaningful but still approximate.
  Source:
  <https://tugraz.elsevierpure.com/en/publications/optimization-and-variational-principles-for-the-shear-strength-re-2>

- `Sysala et al. 2025 (CAS)`:
  use this paper for Davis-based `SSR` continuation, nonlinear / Krylov solution strategy, mesh-adapted 3D slope workflows, and published `COMSOL`-backed validation context. It is also directly tied to the open-source slope-stability code family mirrored in `tmp/source_compare`.
  Source:
  <https://www.sciencedirect.com/science/article/pii/S0045794925002007>

- `Sysala et al. 2025 (SIOPT)`:
  use this paper for the convex / optimization interpretation of geotechnical stability analysis and the continuation viewpoint on the resulting parametric problem formulations. Treat it as formulation support, not as constitutive ground truth.
  Source:
  <https://arxiv.org/abs/2312.12170>

- `Latyshev, Bleyer, Maurini, Hale 2025`:
  a new constitutive reference path should be verified with explicit numerical checks and correctness testing, not only by visual agreement.
  Source:
  <https://jtcam.episciences.org/16616>

4. Separate apples-to-apples results from non-symmetric evidence.
   Create two explicit result classes:
   `Matched comparisons` and `Context / historical comparisons`.
   Why: this prevents reviewers from reading every result as equally direct.
   Output: subsection headers and one explicit note at the start of each result family.
   Files: `paper/sections/results.tex`, `paper/sections/discussion.tex`

### Phase 3: Make the comparisons reviewer-proof

1. Add a `Fairness and limitations` subsection near the end of Results or at the start of Discussion.
   It should explicitly list:
   what is matched,
   what is only approximately comparable,
   what comes from historical runs,
   what is repository-specific.
   Why: saying this first reduces the chance that a reviewer frames it as concealment.
   Files: `paper/sections/discussion.tex`

2. Re-check every sentence that claims speed, robustness, scalability, or accuracy.
   Search terms to audit:
   `robust`, `competitive`, `credible`, `substantial`, `strong scaling`, `exact`, `argmin`.
   Why: these are classic reviewer trigger words.
   Output: either a direct supporting result / citation, or a softer wording.
   Files: all manuscript sections, especially `paper/sections/results.tex` and `paper/sections/discussion.tex`

3. Make the SOTA table maximally factual.
   Allowed columns:
   modeling layer,
   differentiation route,
   sparse / distributed solver path,
   second-order information,
   benchmark family coverage.
   Avoid:
   vague `yes / partial / limited` judgments unless each entry is defined explicitly.
   Files: `paper/sections/introduction.tex`, `paper/sections/related_work.tex`

4. Re-check the claim audit after every major rewrite.
   Rule:
   every externally sourced scientific statement in the main text must map to an exact page or section in `claim_audit.md`.
   Files: `paper/literature/claim_audit.md`

### Phase 4: Tighten the manuscript structure

1. Keep only the strongest narrative arc in the main text.
   Recommended order:
   architecture and derivative strategy,
   flagship benchmark evidence,
   one broader workflow benchmark family,
   limitations and positioning.
   Why: too many equal-weight stories make the paper feel diffuse.

2. Move supporting but non-critical detail to the appendix.
   Good appendix candidates:
   extended parameter sweeps,
   secondary mesh tables,
   extra timing breakdowns,
   sensitivity checks,
   less central benchmark visuals.
   Files: `paper/sections/appendix.tex`

3. Make repository-specific modeling choices explicit everywhere.
   This is especially important for:
   plasticity surrogate functionals,
   topology regularization and move-penalty choices,
   any benchmark-specific load path or continuation schedule.
   Why: the paper should never look like it is attributing repository choices to classical references.
   Files: `paper/sections/benchmarks.tex`, `paper/sections/methodology.tex`

4. End with a modest and precise conclusion.
   The conclusion should emphasize:
   maintained workflow,
   reproducibility,
   distributed sparse solve capability,
   multiple derivative routes,
   hard benchmark coverage.
   It should not claim universal superiority or a new general theory.
   Files: `paper/sections/conclusion.tex`

### Phase 5: Reproducibility and artifact polish

1. Make the compute environment explicit.
   Include:
   commit hash,
   Python / JAX / PETSc versions,
   machine type,
   CPU count,
   MPI layout,
   main run commands.
   Why: software / methods venues care about this a lot.
   Output: one reproducibility paragraph in the paper and one artifact note in the repository.
   Files: `paper/sections/methodology.tex`, repository root or `paper/`

2. Ensure every main-text figure and table can be regenerated from a documented command.
   Good target:
   one command per paper artifact family.
   Files: `paper/Makefile`, `paper/scripts/generate_paper_tables.py`, figure-generation scripts

3. Re-run the literature workflow before submission.
   Update:
   `paper/literature/manifest.json`
   `paper/literature/sources.md`
   `paper/literature/claim_audit.md`
   Why: the paper now depends heavily on citation rigor.

4. Prepare a short reviewer-facing repository note.
   Include:
   where the benchmarks live,
   how to regenerate paper tables,
   where the literature audit is stored,
   what is intentionally omitted because of runtime cost.
   Why: this reduces friction during review.

### Phase 6: Final pre-submission audit

1. Run a language pass focused only on overclaiming.
   Search again for:
   `novel`, `state of the art`, `exact`, `robust`, `competitive`, `strong scaling`, `substantial`.

2. Run a comparison pass focused only on fairness.
   For every baseline:
   ask whether the problem definition, mesh, solver settings, stopping rules, and hardware are really comparable.

3. Run a citation pass focused only on locator precision.
   Replace broad cites with pinpoint cites where a definition, algorithm, or benchmark detail depends on a specific place in the source.

4. Rebuild the paper from scratch.
   Target:
   `latexmk -pdf main.tex`
   Confirm:
   no undefined citations,
   no bibliography errors,
   no broken table inputs,
   no accidental figure drift.

## Execution Type By TODO Item

Legend:

- `Text only`: manuscript, citation, or positioning work; no new code is inherently required.
- `New experiments only`: likely needs fresh runs or external reproduction work, but not necessarily new repository implementation.
- `Support code + tests`: likely needs script / pipeline / report updates and at least smoke-level verification.
- `Possible new scientific implementation`: may require a new benchmark path, reference formulation, or comparison machinery, not just paper edits.

### Phase 1 classification

| Item | Primary Type | Why |
| --- | --- | --- |
| 1.1 Freeze the main claim | Text only | This is contribution framing across the abstract, introduction, and conclusion. |
| 1.2 Choose flagship and support stories | Text only | This is narrative prioritization and section ordering. |
| 1.3 Decide the primary venue track | Text only | This is submission strategy, not implementation work. |
| 1.4 Rewrite title and abstract | Text only | Purely manuscript-facing. |

### Phase 2 classification

| Item | Primary Type | Why |
| --- | --- | --- |
| 2.1 Add a flagship matched ablation table | Support code + tests | This likely needs fresh controlled runs plus updates to the table / asset pipeline and a small verification pass for the new generated outputs. |
| 2.2 Add one fair external baseline | New experiments only | In the best case this is a careful reproduction / comparison exercise rather than new repo implementation, although lightweight comparison scripts may still help. |
| 2.3 Add plasticity surrogate validation | Possible new scientific implementation | This is the one item most likely to require a new reference-comparison path, new diagnostics, or a new reduced validation setup inside the repo. |
| 2.4 Split matched vs historical evidence explicitly | Text only | This is primarily result framing and labeling. |

### Phase 3 classification

| Item | Primary Type | Why |
| --- | --- | --- |
| 3.1 Add a fairness and limitations subsection | Text only | This is explanatory framing for the evidence already in the paper. |
| 3.2 Audit claims about speed / robustness / scaling / exactness | Text only | Mostly wording cleanup, though some claims should only remain if supported by the new evidence from Phase 2. |
| 3.3 Make the SOTA table maximally factual | Text only | This is a literature-and-presentation task. |
| 3.4 Re-check the claim audit after major rewrites | Text only | This is a citation rigor maintenance task. |

### Phase 4 classification

| Item | Primary Type | Why |
| --- | --- | --- |
| 4.1 Keep only the strongest narrative arc | Text only | This is manuscript restructuring. |
| 4.2 Move secondary detail to the appendix | Text only | This is paper organization rather than code work. |
| 4.3 Make repository-specific modeling choices explicit | Text only | This is wording and attribution discipline, not implementation. |
| 4.4 End with a modest and precise conclusion | Text only | Purely editorial. |

### Phase 5 classification

| Item | Primary Type | Why |
| --- | --- | --- |
| 5.1 Make the compute environment explicit | Text only | The information mostly already exists; this is mainly a paper / artifact note unless you choose to automate environment capture. |
| 5.2 Ensure every main-text artifact has a documented regeneration command | Support code + tests | This likely needs `Makefile`, generator-script, and documentation updates, plus a quick check that the documented commands still work. |
| 5.3 Re-run the literature workflow before submission | Text only | This is maintenance on an existing workflow, not inherently new implementation. |
| 5.4 Prepare a reviewer-facing repository note | Text only | Documentation only. |

### Phase 6 classification

| Item | Primary Type | Why |
| --- | --- | --- |
| 6.1 Run an overclaiming pass | Text only | Language cleanup. |
| 6.2 Run a fairness pass on every baseline | Text only | Evidence interpretation and wording. |
| 6.3 Run a citation locator pass | Text only | Citation precision only. |
| 6.4 Rebuild the paper from scratch | Text only | This is a verification step on the existing paper build. |

## What Actually Requires New Code-Like Work

If the question is strictly “what probably needs implementation, scripts, or tests,” the short list is:

1. `2.1` matched ablation table:
   likely new controlled runs, updates to `paper/scripts/generate_paper_tables.py`, and maybe small smoke tests for new generated table inputs.

2. `2.2` external baseline:
   probably new experiment / comparison scaffolding, but not necessarily new solver code in this repository.

3. `2.3` plasticity surrogate validation:
   the most likely item to need genuinely new scientific support code or a new validation implementation.

4. `5.2` regeneration-command hardening:
   likely small but real engineering work in `Makefile`, paper scripts, and maybe existing tests around report generation.

## Bottom Line

Most of this TODO is still manuscript-facing.

The likely split is:

- `18` items are mainly text / citation / organization work.
- `1` item is mostly new experiments without obvious new core implementation: `2.2`.
- `2` items likely need support code and verification work: `2.1`, `5.2`.
- `1` item may need genuinely new scientific implementation support: `2.3`.

So the paper can progress a long way without major new core solver development, but the highest-value acceptance boosters are not purely editorial.

## High-ROI Changes If Time Is Short

If there is only time for a small number of changes, do these first:

1. Add the single flagship ablation table on `Plasticity3D`.
2. Add one external baseline on a truly shared benchmark.
3. Add a short `Fairness and limitations` subsection.
4. Validate the plasticity surrogate on one smaller reference case.
5. Rewrite the abstract, introduction, and conclusion around one narrow thesis.

These five items will improve acceptance odds much more than adding more benchmark breadth.

## Journal Shortlist

Ranking snapshot used below:
`2024` JCR metrics from Clarivate category exports, dataset updated `2025-06-18`, checked on `2026-04-23`.

Filter used:
include only journals with a math-related WoS category used for the filter (`Mathematics, Applied` or `Mathematics, Interdisciplinary Applications`) and `Q1` in both `2024 JIF` and `AIS` within that category.

Notes:
- `MJL` = official Clarivate Master Journal List page.
- `JCR` = the public category export used to verify `JIF`, `AIS`, and quartiles.
- `SCImago` = secondary ranking / classification site.
- Rankings change every year, so re-check all metrics again immediately before submission.

| Journal | Fit For This Paper | WoS Category Used For Filter | 2024 JIF | JIF Q | AIS | AIS Q | Classifications | Check Links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Journal of Scientific Computing | Best current overall fit for a rigorous scientific-computing / methods paper with broad benchmark coverage | Mathematics, Applied | 3.3 | Q1 | 1.252 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics; Numerical Analysis; Software; Computational Theory and Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0885-7474&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=23490&tip=sid) |
| ACM Transactions on Mathematical Software | Excellent if the revision foregrounds software architecture, reproducibility, and benchmark infrastructure | Mathematics, Applied | 3.2 | Q1 | 1.494 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Software | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0098-3500&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=18120&tip=sid) |
| SIAM Journal on Scientific Computing | Strong option if the paper sharpens the nonlinear solver, sparse linear algebra, and parallel performance story | Mathematics, Applied | 2.6 | Q1 | 1.669 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=1064-8275&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=26425&tip=sid) |
| Computational Mechanics | Best mechanics-leaning target if `Plasticity3D` and hyperelastic / mechanics validation become the main story | Mathematics, Interdisciplinary Applications | 3.8 | Q1 | 1.033 | Q1 | WoS: Mathematics, Interdisciplinary Applications<br>SCImago: Applied Mathematics; Computational Mathematics; Computational Mechanics; Mechanical Engineering | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0178-7675&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21996&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?q=28457&tip=sid) |
| Applied Mathematical Modelling | Good if the paper is positioned as an applied nonlinear modeling and optimization workflow paper rather than a software artifact paper | Mathematics, Interdisciplinary Applications | 5.1 | Q1 | 0.925 | Q1 | WoS: Mathematics, Interdisciplinary Applications<br>SCImago: Applied Mathematics; Modeling and Simulation | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0307-904X&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21996&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=28065&tip=sid) |
| Computer Methods in Applied Mechanics and Engineering | Stretch target; realistic only after stronger mechanics depth, cleaner external baselines, and a more decisive flagship result | Mathematics, Interdisciplinary Applications | 7.3 | Q1 | 1.801 | Q1 | WoS: Mathematics, Interdisciplinary Applications<br>SCImago: Computational Mechanics; Computer Science Applications; Mechanical Engineering; Mechanics of Materials | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0045-7825&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21996&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=18158&tip=sid) |
| Advances in Computational Mathematics | Plausible if the paper is tightened around computational methodology and keeps engineering benchmarking secondary | Mathematics, Applied | 2.1 | Q1 | 0.874 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=1019-7168&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?q=28041&tip=sid) |
| Computational Optimization and Applications | Optimization-tilted fallback if the revision pushes topology, globalization, and second-order optimization much harder | Mathematics, Applied | 2.0 | Q1 | 1.145 | Q1 | WoS: Mathematics, Applied<br>SCImago: Applied Mathematics; Computational Mathematics; Control and Optimization | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=0926-6003&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=28459&tip=sid) |
| Numerical Linear Algebra with Applications | Only a good match if the paper shifts more clearly toward sparse nonlinear linearization, Krylov methods, and solver methodology | Mathematics, Applied | 2.1 | Q1 | 1.013 | Q1 | WoS: Mathematics, Applied<br>SCImago: Algebra and Number Theory; Applied Mathematics | [MJL](https://mjl.clarivate.com/search-results?hide_exact_match_fl=true&issn=1070-5325&utm_campaign=journal-profile-share-this-journal&utm_medium=share-by-link&utm_source=mjl) · [JCR](https://kniznica.umb.sk/app/cmsFile.php?ID=21995&disposition=i) · [SCImago](https://www.scimagojr.com/journalsearch.php?clean=0&q=25712&tip=sid) |

## Recommended Submission Order

For the paper in its current natural shape, a reasonable submission order is:

1. `Journal of Scientific Computing`
2. `ACM Transactions on Mathematical Software`
3. `SIAM Journal on Scientific Computing`
4. `Computational Mechanics`
5. `Applied Mathematical Modelling`
6. `Advances in Computational Mathematics`
7. `Computational Optimization and Applications`
8. `Computer Methods in Applied Mechanics and Engineering`
9. `Numerical Linear Algebra with Applications`

## How To Reorder The Venue List After Revision

Promote `SIAM Journal on Scientific Computing` to the top if:
the revision adds a much stronger solver / scaling / sparse linear algebra story.

Promote `Computational Mechanics` higher if:
the paper becomes clearly mechanics-led and the plasticity validation is strengthened.

Promote `Computer Methods in Applied Mechanics and Engineering` only if:
the paper gains one or two very strong apples-to-apples comparisons and a more decisive flagship mechanics result.

Promote `ACM Transactions on Mathematical Software` to the top if:
the revision leans harder into software design, reproducibility, artifact quality, and reusable implementation infrastructure.

## Minimal Concrete Plan Before Submission

If the goal is to maximize acceptance probability without turning this into a new project, the best concrete path is:

1. Freeze the thesis as a software-and-methods paper.
2. Make `Plasticity3D` the flagship result.
3. Add one tightly controlled derivative-route ablation table.
4. Add one fair external baseline.
5. Add one short plasticity surrogate validation.
6. Add a fairness / limitations subsection.
7. Re-tune the title, abstract, and conclusion to the selected venue.
8. Re-check all rankings immediately before submission.
