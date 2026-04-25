# Plasticity3D Results

## Current Maintained Comparison

The current recommended maintained 3D result is now the converged
`P4(L1_2), lambda = 1.0` study with:

- mesh / space: `hetero_ssr_L1_2`, `P4`
- assembly: `local_constitutiveAD`
- solver: `local_pmg`
- nonlinear stop: `grad_norm < 1e-2`
- linear solve: `fgmres + mg` with `ksp_rtol = 1e-1`

This page now keeps two layers of evidence side by side:

- the new converged maintained scaling story for `lambda = 1.0`
- the older `lambda = 1.5` fixed-work and capped-Newton material, preserved
  below as historical backend and diagnostic context

## Recommended Maintained Result

The promoted maintained stack is `local_constitutiveAD + local_pmg` on
`P4(L1_2)` at `lambda = 1.0`, with full strong-scaling coverage on
`1/2/4/8/16/32` ranks. Every rank converged to the same nonlinear state:

- final gradient norm `5.842228e-03`
- energy `-3.1065003302388e+06`
- `omega = 6.2167993410608e+06`
- `u_max = 7.975388291134452e-01`

Recommended settings:

| knob | value |
| --- | --- |
| model | heterogeneous 3D Mohr-Coulomb plasticity, `lambda = 1.0` |
| mesh / space | `hetero_ssr_L1_2`, `P4` |
| assembly | `local_constitutiveAD` |
| hierarchy | `same_mesh_p4_p2_p1` |
| nonlinear method | Newton with `armijo` line search |
| stop | `grad_norm < 1e-2` or `maxit = 50` |
| initial guess | elastic solve on the same PMG stack |
| linear method | `fgmres + mg` |
| `ksp_rtol / ksp_max_it` | `1e-1 / 100` |
| fine / bridge smoothers | `chebyshev + jacobi`, `5` steps |
| coarse solve | `cg + hypre` |
| problem build mode | `rank_local` |
| MG transfer build mode | `owned_rows` |
| thread caps | `OMP/JAX/BLAS = 1` thread per rank |

Raw artifacts:

- [local-only scaling report](../../artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling/REPORT.md)
- [local-only scaling summary](../../artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling/comparison_summary.json)
- [matched local-vs-source report](../../artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling/REPORT.md)
- [all-PMG comparison report](../../artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling_all_pmg/REPORT.md)

### Full Strong Scaling

![Plasticity3D recommended overall scaling](../assets/plasticity3d/plasticity3d_l1_2_lambda1_local_pmg_scaling_overall.png)

| ranks | total [s] | solve [s] | speedup | efficiency | Newton its | linear its |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5423.858 | 5175.697 | 1.000 | 1.000 | 17 | 114 |
| 2 | 2914.163 | 2776.940 | 1.861 | 0.931 | 17 | 114 |
| 4 | 1716.574 | 1634.228 | 3.160 | 0.790 | 17 | 114 |
| 8 | 1164.304 | 1107.182 | 4.658 | 0.582 | 17 | 114 |
| 16 | 608.352 | 574.736 | 8.916 | 0.557 | 17 | 114 |
| 32 | 300.435 | 283.927 | 18.053 | 0.564 | 17 | 114 |

The important read is that this is now a clean timing-only scaling study. The
nonlinear work is identical at every rank, so the changes across the table are
actual scalability changes rather than different Newton trajectories.

### Component Timing

![Plasticity3D recommended common components](../assets/plasticity3d/plasticity3d_l1_2_lambda1_local_pmg_common_components.png)

![Plasticity3D recommended component breakdown](../assets/plasticity3d/plasticity3d_l1_2_lambda1_local_pmg_component_breakdown.png)

| ranks | backend build [s] | elastic guess [s] | linear assemble [s] | linear setup [s] | linear solve [s] | Hessian callback [s] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 308.838 | 24.310 | 2971.253 | 689.372 | 1396.128 | 3146.747 |
| 2 | 198.982 | 13.047 | 1585.133 | 373.494 | 749.252 | 1683.099 |
| 4 | 146.351 | 8.117 | 853.238 | 258.854 | 472.700 | 909.161 |
| 8 | 69.851 | 6.514 | 499.598 | 194.866 | 375.188 | 536.567 |
| 16 | 37.005 | 3.440 | 238.872 | 115.638 | 198.429 | 260.218 |
| 32 | 24.212 | 2.673 | 36.104 | 70.681 | 159.769 | 43.867 |

The standout pattern is that callback-heavy work scales very well, especially
the Hessian path, while the KSP setup and solve phases bend more slowly at high
rank. Even so, the recommended stack still reaches `18.05x` end-to-end speedup
at `32` ranks on the converged problem.

### Matched Local-Vs-Source Context

![Plasticity3D recommended local vs source](../assets/plasticity3d/plasticity3d_l1_2_lambda1_local_vs_source.png)

The maintained local path is the recommended default, but it is still useful to
see how the same local PMG nonlinear stack behaves when the source assembly is
swapped in. At matched `4/8/16/32` ranks, both variants converge in the same
`17` Newton steps with the same `114` linear iterations, so the comparison is
again timing-only:

| ranks | local wall [s] | source wall [s] | wall ratio local/source | local solve [s] | source solve [s] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1716.574 | 951.034 | 1.805 | 1634.228 | 889.558 |
| 8 | 1164.304 | 696.630 | 1.671 | 1107.182 | 656.270 |
| 16 | 608.352 | 373.699 | 1.628 | 574.736 | 350.449 |
| 32 | 300.435 | 276.856 | 1.085 | 283.927 | 259.549 |

The source assembly remains faster at every matched rank, but the gap narrows
substantially by `32` ranks. We still promote `local_constitutiveAD + local_pmg`
here because it is the maintained implementation and now has the complete,
rank-consistent `1/2/4/8/16/32` converged scaling record.

### Alternative PMG Profile

![Plasticity3D recommended sourcefixed comparison](../assets/plasticity3d/plasticity3d_l1_2_lambda1_sourcefixed_compare.png)

We also checked the source-fixed-like PMG profile on the same `lambda = 1.0`
benchmark family. Converged-only view:

| ranks | `local_constitutiveAD + local_pmg` | `source + local_pmg` | `local_constitutiveAD + local_pmg_sourcefixed` | `source + local_pmg_sourcefixed` |
| ---: | --- | --- | --- | --- |
| 4 | `1716.6 s, 17 N, 114 L, grad 5.84e-3` | `951.0 s, 17 N, 114 L, grad 5.84e-3` | `2115.9 s, 24 N, 204 L, grad 6.53e-3` | `988.4 s, 21 N, 210 L, grad 2.49e-3` |
| 8 | `1164.3 s, 17 N, 114 L, grad 5.84e-3` | `696.6 s, 17 N, 114 L, grad 5.84e-3` | `1873.6 s, 33 N, 238 L, grad 2.53e-3` | `898.0 s, 27 N, 246 L, grad 9.53e-3` |
| 16 | `608.4 s, 17 N, 114 L, grad 5.84e-3` | `373.7 s, 17 N, 114 L, grad 5.84e-3` | `-` | `-` |
| 32 | `300.4 s, 17 N, 114 L, grad 5.84e-3` | `276.9 s, 17 N, 114 L, grad 5.84e-3` | `347.9 s, 29 N, 250 L, grad 2.24e-3` | `-` |

Rows omitted from the alternative profile hit `maxit = 50` before reaching
`grad_norm < 1e-2`:

- `np16 local_constitutiveAD + local_pmg_sourcefixed`
- `np16 source + local_pmg_sourcefixed`
- `np32 source + local_pmg_sourcefixed`

That is the practical reason it is not the default here. It can be competitive
on some ranks, but it does materially more Newton and Krylov work and loses
robustness on the same benchmark family.

The remaining sections below preserve the older `lambda = 1.5` fixed-work and
`maxit = 20` diagnostic campaigns as historical backend context.

## Historical Backend Promotion Context

The historical fixed-work material below records the backend that was promoted
earlier in the assembly-optimization ladder:

- fine operator assembly: `coo`
- transfer build: `coo_vectorized`
- `P4` Hessian chunk size: `4`

Against the previously published refined `P4(L1_2)` baseline, this backend
improves:

- fixed-work `32`-rank total time from `153.333 s` to `109.831 s`
  (`28.4%` faster)
- fixed-work `32`-rank first assembled linearization from `20.616 s` to
  `18.570 s` (`9.9%` faster)
- `32`-rank `maxit = 20` diagnostic total time from `1151.586 s` to
  `1059.366 s` (`8.0%` faster)

The nonlinear trajectory and final state of the `32`-rank diagnostic are
unchanged to numerical noise; the win is in backend efficiency, not in altered
physics or solver policy.

## Historical Fixed-Work Settings

Historical maintained fixed-work `P4` stack:

| knob | value |
| --- | --- |
| model | heterogeneous 3D Mohr-Coulomb plasticity, `lambda = 1.5` |
| fine space | `P4(L1_2)` |
| maintained hierarchy | `P4(L1_2) -> P2(L1_2) -> P1(L1_2) -> P1(L1)` |
| nonlinear method | Newton with `armijo` line search |
| benchmark mode | fixed-work `maxit = 1` |
| initial guess | elastic solve on the same PMG stack |
| Newton tangent | pure plastic autodiff tangent |
| linear method | `fgmres` |
| `ksp_rtol / ksp_max_it` | `1e-2 / 100` |
| fine / bridge smoothers | `chebyshev + jacobi`, `5` steps on `P4`, `P2`, and `P1(L1_2)` |
| coarse solve | `cg + hypre` on `P1(L1)` with near-nullspace |
| problem build mode | `rank_local` |
| MG level build mode | `rank_local` |
| transfer build mode | `owned_rows` |
| hot-path overlap | `overlap_p2p` |
| reorder mode | `block_xyz` |
| fine assembly backend | `coo` |
| transfer backend | `coo_vectorized` |
| `P4` Hessian chunk size | `4` |
| thread caps | `OMP/JAX/BLAS = 1` thread per rank |

## Historical Fixed-Work Scaling

The historical fixed-work scaling story preserved here is the optimized
`P4(L1_2), lambda = 1.5` campaign on `1/2/4/8/16/32` ranks. Like the 2D
fixed-work results page, this is intentionally a capped benchmark: each row
runs one Newton iteration after setup and elastic bootstrap.

![Plasticity3D overall scaling](../assets/plasticity3d/plasticity3d_p4_l1_2_scaling_overview_loglog.png)

`P4(L1_2)` fixed-work scaling summary:

| ranks | total [s] | solve [s] | speedup | total efficiency | solve efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1390.343 | 685.397 | 1.000 | 1.000 | 1.000 |
| 2 | 789.814 | 367.937 | 1.760 | 0.880 | 0.931 |
| 4 | 443.901 | 208.625 | 3.132 | 0.783 | 0.821 |
| 8 | 289.339 | 147.136 | 4.805 | 0.601 | 0.582 |
| 16 | 166.152 | 85.864 | 8.368 | 0.523 | 0.499 |
| 32 | 109.831 | 58.459 | 12.659 | 0.396 | 0.366 |

The practical reading is:

- end-to-end efficiency stays above the `70%` target through `4` ranks
- compared with the earlier published backend, efficiency is materially better
  at `4/8/16/32` ranks, especially at `32` ranks
- the curve still bends down beyond `8` ranks because overlap duplication and
  non-scaling setup pieces remain structurally present in the overlap-domain
  formulation

## Historical Fixed-Work Linear Solve Timing Split

![Plasticity3D linear scaling](../assets/plasticity3d/plasticity3d_p4_l1_2_linear_solve_breakdown.png)

| ranks | assemble [s] | KSP setup [s] | KSP solve [s] |
| ---: | ---: | ---: | ---: |
| 1 | 301.702 | 48.812 | 302.931 |
| 2 | 164.638 | 26.983 | 160.570 |
| 4 | 88.286 | 21.116 | 95.224 |
| 8 | 50.200 | 17.689 | 79.340 |
| 16 | 28.220 | 11.892 | 47.799 |
| 32 | 18.570 | 7.898 | 33.022 |

The first assembled linearization keeps scaling reasonably well, especially in
assembly. The Krylov solve remains one of the largest repeated costs on the
refined benchmark, and KSP setup still scales more weakly than the best
callback pieces.

## Weak Scaling Context

The strong-scaling table above keeps the global `P4(L1_2)` problem fixed while
raising the rank count. To complement that view, we also maintain a weak-style
comparison between:

- `P4(L1)` on `1` rank
- `P4(L1_2)` on `8` ranks

This is a natural pair because `L1_2` is one uniform tetra refinement of `L1`,
so the refined problem has `8x` more macro elements while the `8`-rank run
keeps the local work in the same ballpark. For this comparison we use a capped
`maxit = 20` Newton window on both sides, with the same maintained backend:

- fine assembly backend: `coo`
- transfer backend: `coo_vectorized`
- `P4` chunk size: `4`
- linear stack: `fgmres + PMG + hypre coarse`
- nonlinear stack: elastic initial guess, pure plastic tangent, Armijo
- thread caps: `OMP/JAX/BLAS = 1` thread per rank

Raw weak-scaling report:

- [P4(L1) vs P4(L1_2) weak-scaling report](../../artifacts/raw_results/weak_scaling_probe/p4_l1_vs_l1_2_maxit20/REPORT.md)

Outcome summary:

| quantity | `P4(L1)` on `1` rank | `P4(L1_2)` on `8` ranks |
| --- | ---: | ---: |
| nonlinear iterations | `20` | `20` |
| total time [s] | `1499.272` | `2806.067` |
| solve time [s] | `1415.373` | `2662.813` |
| final gradient norm | `7.665534e+02` | `1.447193e+03` |
| `u_max` | `0.939201` | `0.939293` |
| `omega` | `6.4123755e+06` | `6.4125229e+06` |
| linear iterations total | `377` | `504` |
| mean Armijo `alpha` | `0.606250` | `0.501660` |

![Plasticity3D weak phase comparison](../assets/plasticity3d/plasticity3d_p4_l1_vs_l1_2_maxit20_phase_compare.png)

![Plasticity3D weak efficiency](../assets/plasticity3d/plasticity3d_p4_l1_vs_l1_2_maxit20_efficiency.png)

The raw weak-scaling read is:

- total efficiency: `0.534`
- solve-phase efficiency: `0.532`

That is informative, but it is not the whole story. The refined `L1_2 @ 8`
case is also a harder nonlinear solve, not just a larger one. Both runs used
the same `20` Newton iterations, but the refined case still needed `1.337x`
more Krylov iterations and smaller average Armijo steps. So we also keep a
work-normalized view that divides repeated nonlinear costs by Newton count and
linear costs by cumulative Krylov iterations.

![Plasticity3D weak work-normalized efficiency](../assets/plasticity3d/plasticity3d_p4_l1_vs_l1_2_maxit20_work_normalized_efficiency.png)

Work-normalized highlights:

- Hessian extraction per Newton iteration efficiency: `0.822`
- Hessian HVP per Newton iteration efficiency: `0.684`
- cumulative linear solve time per Krylov iteration efficiency: `0.510`
- cumulative linear setup time per Krylov iteration efficiency: `0.498`

The fair reading is that the callback-heavy Hessian path remains reasonably
healthy under weak scaling, while the main remaining loss is still on the
Krylov and setup side. The raw `0.53` end-to-end efficiency therefore blends
true scaling loss with the fact that the refined case is genuinely harder to
drive through the same `20`-step Newton window.

## 32-Rank Newton Diagnostic

The fixed-work table above is useful for scaling, but it hides how the refined
`P4(L1_2)` solve behaves when we let Newton continue. The maintained diagnostic
slice below reruns the same case on `32` ranks with `maxit = 20`, the same
mixed hierarchy, the same PMG/Hypre stack, an elastic initial guess, and pure
plastic autodiff tangents.

Outcome summary:

| quantity | value |
| --- | ---: |
| status | `failed` |
| message | `Maximum number of iterations reached` |
| nonlinear iterations | `20` |
| final gradient norm | `1.447193e+03` |
| energy | `-3.1883394385e+06` |
| `u_max` | `9.3929316185e-01` |
| `omega` | `6.4125229180e+06` |
| linear iterations total | `504` |
| solve time | `1005.572 s` |
| end-to-end wall time | `1059.366 s` |
| worst-rank RSS max | `4.822 GiB` |
| worst-rank RSS HWM max | `4.920 GiB` |

Detailed settings:

| knob | value |
| --- | --- |
| model | heterogeneous 3D Mohr-Coulomb plasticity, `lambda = 1.5` |
| mesh / space | `hetero_ssr_L1_2`, `P4` |
| ranks | `32` |
| hierarchy | `P4(L1_2) -> P2(L1_2) -> P1(L1_2) -> P1(L1)` |
| nonlinear method | Newton + `armijo` |
| `maxit / tolg / tolg_rel` | `20 / 1e-2 / 0.0` |
| initial guess | elastic solve on the same PMG stack |
| Newton tangent | pure plastic autodiff tangent |
| linear method | `fgmres + mg` |
| `ksp_rtol / ksp_max_it` | `1e-2 / 100` |
| capped linear-step policy | accept `DIVERGED_MAX_IT` directions with true-rel cap `0.06` |
| fine / bridge smoothers | `chebyshev + jacobi`, `5` steps on `P4`, `P2`, and `P1(L1_2)` |
| coarse solve | `cg + hypre` on `P1(L1)` with near-nullspace |
| Hypre settings | nodal coarsen `6`, vector interp `3`, strong threshold `0.5`, coarsen `HMIS`, max iter `2`, tol `0.0`, relax `symmetric-SOR/Jacobi` |
| problem build mode | `rank_local` |
| MG level build mode | `rank_local` |
| transfer build mode | `owned_rows` |
| hot-path overlap | `overlap_p2p` |
| reorder mode | `block_xyz` |
| fine assembly backend | `coo` |
| transfer backend | `coo_vectorized` |
| `P4` Hessian chunk size | `4` |
| thread caps | `OMP/JAX/BLAS = 1` thread per rank |

Setup timings:

| stage | time [s] |
| --- | ---: |
| problem load | `12.156` |
| assembler create | `16.289` |
| MG hierarchy build | `3.147` |
| elastic initial guess total | `19.736` |
| elastic initial-guess solve | `3.709` |
| elastic initial-guess KSP its | `4` |
| MG transfer build time | `2.684` |

The important practical reading from this capped run is:

- memory stayed flat and healthy: worst-rank RSS stayed in the narrow
  `4.71 .. 4.82 GiB` band and worst-rank HWM topped out at `4.92 GiB`
- every Newton step was accepted, but the line search was active throughout:
  `alpha = 1.0` only `6` times, while `14` steps were damped
- the run did not break down in the linear solver; it stalled because the
  gradient plateaued in the `1.6e3 .. 1.8e3` range by the end of the `20`-step
  window
- the dominant repeated cost is still the Hessian / linearization stage:
  average `t_hess = 48.536 s` per Newton step, with average worst-rank KSP
  solve `28.893 s`
- relative to the earlier maintained diagnostic, the Newton path is the same
  but the setup path is much cheaper, especially MG hierarchy build

![Plasticity3D 32-rank convergence](../assets/plasticity3d/plasticity3d_p4_l1_2_np32_maxit20_convergence.png)

![Plasticity3D 32-rank timing](../assets/plasticity3d/plasticity3d_p4_l1_2_np32_maxit20_newton_timing.png)

![Plasticity3D 32-rank linear diagnostics](../assets/plasticity3d/plasticity3d_p4_l1_2_np32_maxit20_linear_diagnostics.png)

![Plasticity3D 32-rank memory profile](../assets/plasticity3d/plasticity3d_p4_l1_2_np32_maxit20_memory_profile.png)

Most expensive Newton steps by wall time:

| iteration | `t_iter` [s] | `t_hess` [s] | KSP max | KSP solve max [s] | `alpha` | note |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | `86.280` | `82.085` | `57` | `62.801` | `0.00195` | strongest damping and largest linear solve |
| 11 | `77.130` | `75.658` | `50` | `55.868` | `0.5` | late-stage Krylov spike |
| 16 | `75.445` | `73.515` | `45` | `53.898` | `0.25` | another damped late-stage spike |
| 1 | `55.976` | `52.913` | `29` | `31.111` | `0.01562` | first nonlinear correction after elastic bootstrap |
| 18 | `54.304` | `53.295` | `30` | `34.255` | `1.0` | representative late full-step solve |

Per-iteration Newton summary:

| it | energy | `||g||` | `alpha` | KSP max | KSP sum | lin solve max [s] | lin assemble max [s] | lin setup max [s] | `t_grad` [s] | `t_hess` [s] | `t_ls` [s] | `t_iter` [s] | RSS max [GiB] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `-3.139751e+06` | `7.593862e+03` | `0.01562` | `29` | `928` | `31.111` | `18.627` | `8.018` | `0.637` | `52.913` | `2.425` | `55.976` | `4.720` |
| `2` | `-3.151344e+06` | `1.092720e+04` | `1.00000` | `17` | `544` | `18.736` | `16.123` | `7.739` | `0.612` | `37.965` | `0.349` | `38.927` | `4.714` |
| `3` | `-3.187262e+06` | `5.744114e+04` | `1.00000` | `7` | `224` | `7.265` | `15.959` | `7.522` | `0.621` | `26.251` | `0.291` | `27.163` | `4.740` |
| `4` | `-3.187290e+06` | `5.333515e+03` | `0.00195` | `57` | `1824` | `62.801` | `16.220` | `7.797` | `0.621` | `82.085` | `3.574` | `86.280` | `4.717` |
| `5` | `-3.187337e+06` | `5.329288e+03` | `0.01562` | `24` | `768` | `29.197` | `16.108` | `8.150` | `0.620` | `48.661` | `2.874` | `52.156` | `4.774` |
| `6` | `-3.187593e+06` | `5.348373e+03` | `0.25000` | `19` | `608` | `22.088` | `16.295` | `7.588` | `0.639` | `41.518` | `1.079` | `43.237` | `4.774` |
| `7` | `-3.188057e+06` | `4.445070e+03` | `1.00000` | `21` | `672` | `24.970` | `16.928` | `9.139` | `0.638` | `45.232` | `0.432` | `46.302` | `4.774` |
| `8` | `-3.188067e+06` | `4.284615e+03` | `0.50000` | `26` | `832` | `30.640` | `15.758` | `7.592` | `0.619` | `49.516` | `0.640` | `50.776` | `4.774` |
| `9` | `-3.188108e+06` | `4.912981e+03` | `0.25000` | `18` | `576` | `20.531` | `16.392` | `7.991` | `0.618` | `40.122` | `1.084` | `41.825` | `4.774` |
| `10` | `-3.188262e+06` | `4.086129e+03` | `1.00000` | `20` | `640` | `22.298` | `16.077` | `7.770` | `0.611` | `41.606` | `0.351` | `42.569` | `4.774` |
| `11` | `-3.188264e+06` | `1.912059e+03` | `0.50000` | `50` | `1600` | `55.868` | `16.643` | `8.533` | `0.615` | `75.658` | `0.857` | `77.130` | `4.778` |
| `12` | `-3.188299e+06` | `3.673696e+03` | `0.50000` | `14` | `448` | `16.708` | `17.426` | `9.516` | `0.620` | `37.494` | `0.860` | `38.975` | `4.778` |
| `13` | `-3.188307e+06` | `2.610517e+03` | `0.50000` | `21` | `672` | `22.597` | `16.802` | `8.647` | `0.638` | `42.575` | `0.601` | `43.814` | `4.778` |
| `14` | `-3.188315e+06` | `2.653163e+03` | `0.25000` | `19` | `608` | `22.064` | `16.028` | `7.756` | `0.610` | `41.278` | `1.160` | `43.049` | `4.778` |
| `15` | `-3.188328e+06` | `2.187657e+03` | `1.00000` | `25` | `800` | `31.168` | `16.289` | `7.718` | `0.621` | `50.813` | `0.431` | `51.866` | `4.781` |
| `16` | `-3.188329e+06` | `1.595221e+03` | `0.25000` | `45` | `1440` | `53.898` | `16.329` | `8.266` | `0.635` | `73.515` | `1.295` | `75.445` | `4.781` |
| `17` | `-3.188333e+06` | `1.800749e+03` | `0.25000` | `25` | `800` | `29.684` | `16.080` | `7.278` | `0.639` | `48.730` | `1.262` | `50.631` | `4.781` |
| `18` | `-3.188336e+06` | `1.591789e+03` | `1.00000` | `30` | `960` | `34.255` | `15.997` | `7.502` | `0.614` | `53.295` | `0.395` | `54.304` | `4.781` |
| `19` | `-3.188337e+06` | `1.652568e+03` | `0.25000` | `22` | `704` | `24.307` | `16.770` | `8.580` | `0.618` | `44.175` | `1.001` | `45.794` | `4.781` |
| `20` | `-3.188339e+06` | `1.716143e+03` | `0.50000` | `15` | `480` | `17.674` | `16.264` | `8.253` | `0.606` | `37.316` | `0.855` | `38.778` | `4.822` |

## Component Breakdown

![Plasticity3D phase scaling](../assets/plasticity3d/plasticity3d_p4_l1_2_phase_scaling_grid.png)

Largest repeated or user-visible components:

| component | 1 rank [s] | 32 ranks [s] | speedup | interpretation |
| --- | ---: | ---: | ---: | --- |
| Hessian callbacks total | 537.232 | 31.110 | 17.27x | still the largest repeated nonlinear cost |
| Hessian extraction | 342.366 | 13.071 | 26.19x | one of the healthiest scaling callback pieces |
| elastic initial guess | 324.966 | 20.059 | 16.20x | scales well enough and no longer dominates |
| problem load | 312.900 | 17.844 | 17.53x | finite and scalable after the rank-local load fixes |
| first linear KSP solve | 302.931 | 33.022 | 9.17x | major repeated cost and still far from ideal |
| first linear assemble | 301.702 | 18.570 | 16.25x | clearly better than the earlier maintained backend |

Worst-scaling pieces:

| component | 1 rank [s] | 32 ranks [s] | speedup | interpretation |
| --- | ---: | ---: | ---: | --- |
| MG hierarchy build | 17.926 | 4.189 | 4.28x | much better than the old maintained backend, but still weak |
| assembler create | 44.438 | 14.605 | 3.04x | setup-side distributed metadata cost remains visible |
| Hessian HVP compute | 192.228 | 17.902 | 10.74x | repeated callback cost that lags the extraction stage |
| first linear KSP setup | 48.812 | 7.898 | 6.18x | repeated but non-ideal at high rank counts |

So the current refined 3D PMG story is:

- the catastrophic `P4(L1_2)` setup blow-up is fixed
- the promoted backend removes the earlier MG-transfer build bottleneck
- `problem_load`, transfer build, and the elastic bootstrap now scale in a
  usable way
- the remaining structural limits are the overlap-heavy fine-level assembly
  path, the first Krylov solve, and setup-side assembler metadata

## Overlap / Duplication

![Plasticity3D overlap](../assets/plasticity3d/plasticity3d_p4_l1_2_overlap_and_efficiency.png)

| ranks | local elements min | local elements max | element duplication factor | overlap DOF factor |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 147352 | 147352 | 1.000 | 1.008 |
| 2 | 75476 | 75757 | 1.026 | 1.041 |
| 4 | 38248 | 40902 | 1.074 | 1.101 |
| 8 | 18880 | 22482 | 1.152 | 1.200 |
| 16 | 9762 | 13250 | 1.318 | 1.409 |
| 32 | 5017 | 8625 | 1.646 | 1.824 |

The duplication table explains the late-rank bend in the scaling curves. The
current overlap-domain assembly remains productive through `4` ranks, but by
`16` and especially `32` the duplicated local work is already a large part of
the remaining time.

## Reproduction Commands

For timings comparable to the maintained 3D table, pin the CPU backend to one
thread per rank before running the commands below:

```bash
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
```

Run the promoted converged `lambda = 1.0` maintained sweep. The `np = 1` and
`np = 2` rows should be run with no other benchmark MPI jobs active:

```bash
./.venv/bin/python experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling.py \
  --ranks 1 2

./.venv/bin/python experiments/analysis/generate_plasticity3d_impl_scaling_assets.py \
  --summary-json artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling/comparison_summary.json \
  --out-dir artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling

./.venv/bin/python experiments/analysis/generate_plasticity3d_l1_2_lambda1_docs_assets.py
```

Run the maintained refined `1/2/4/8/16/32` optimized sweep:

```bash
P4_HESSIAN_CHUNK_SIZE=4 \
ASSEMBLY_BACKEND=coo \
ENABLE_PETSC_LOG_EVENTS=1 \
ENABLE_PETSC_LOG_VIEW=1 \
REPORT_OUTDIR=artifacts/raw_results/assembly_opt_ladder/coo_chunk4_sweep/assets \
REPORT_PATH=artifacts/raw_results/assembly_opt_ladder/coo_chunk4_sweep/REPORT.md \
bash experiments/analysis/run_p4_l1_2_uniform_tail_scaling_serial.sh \
  artifacts/raw_results/assembly_opt_ladder/coo_chunk4_sweep
```

Regenerate the optimized sweep report assets:

```bash
P4_L1_2_SCALING_ROOT=artifacts/raw_results/assembly_opt_ladder/coo_chunk4_sweep \
P4_L1_2_SCALING_OUTDIR=artifacts/raw_results/assembly_opt_ladder/coo_chunk4_sweep/assets \
P4_L1_2_SCALING_REPORT=artifacts/raw_results/assembly_opt_ladder/coo_chunk4_sweep/REPORT.md \
./.venv/bin/python experiments/analysis/generate_p4_l1_2_uniform_tail_scaling_assets.py
```

Run the detailed optimized `32`-rank capped-Newton diagnostic:

```bash
mpiexec -n 32 ./.venv/bin/python -u -m src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof \
  --mesh_name hetero_ssr_L1_2 \
  --elem_degree 4 \
  --lambda-target 1.5 \
  --profile performance \
  --ksp_type fgmres \
  --pc_type mg \
  --ksp_rtol 1e-2 \
  --ksp_max_it 100 \
  --accept_ksp_maxit_direction \
  --ksp_maxit_direction_true_rel_cap 0.06 \
  --distribution_strategy overlap_p2p \
  --problem_build_mode rank_local \
  --mg_level_build_mode rank_local \
  --mg_transfer_build_mode owned_rows \
  --element_reorder_mode block_xyz \
  --mg_strategy uniform_refined_p4_p2_p1_p1 \
  --use_near_nullspace \
  --mg_coarse_backend hypre \
  --mg_coarse_ksp_type cg \
  --mg_coarse_pc_type hypre \
  --mg_coarse_hypre_nodal_coarsen 6 \
  --mg_coarse_hypre_vec_interp_variant 3 \
  --mg_coarse_hypre_strong_threshold 0.5 \
  --mg_coarse_hypre_coarsen_type HMIS \
  --mg_coarse_hypre_max_iter 2 \
  --mg_coarse_hypre_tol 0.0 \
  --mg_coarse_hypre_relax_type_all symmetric-SOR/Jacobi \
  --mg_p1_smoother_ksp_type chebyshev \
  --mg_p1_smoother_pc_type jacobi \
  --mg_p1_smoother_steps 5 \
  --mg_p2_smoother_ksp_type chebyshev \
  --mg_p2_smoother_pc_type jacobi \
  --mg_p2_smoother_steps 5 \
  --mg_p4_smoother_ksp_type chebyshev \
  --mg_p4_smoother_pc_type jacobi \
  --mg_p4_smoother_steps 5 \
  --assembly_backend coo \
  --enable_petsc_log_events \
  --petsc_log_view_path artifacts/raw_results/assembly_opt_ladder/coo_chunk4_np32_maxit20/petsc_log_view.txt \
  --p4_hessian_chunk_size 4 \
  --line_search armijo \
  --armijo_alpha0 1.0 \
  --armijo_c1 1e-4 \
  --armijo_shrink 0.5 \
  --armijo_max_ls 40 \
  --elastic_initial_guess \
  --no-regularized_newton_tangent \
  --tolg 1e-2 \
  --tolg_rel 0.0 \
  --maxit 20 \
  --save_history \
  --debug_setup \
  --quiet \
  --out artifacts/raw_results/assembly_opt_ladder/coo_chunk4_np32_maxit20/output.json \
  --progress-out artifacts/raw_results/assembly_opt_ladder/coo_chunk4_np32_maxit20/progress.json
```

Regenerate the detailed diagnostic plots:

```bash
./.venv/bin/python experiments/analysis/generate_plasticity3d_p4_l1_2_np32_diagnostic_assets.py \
  --output-json artifacts/raw_results/assembly_opt_ladder/coo_chunk4_np32_maxit20/output.json \
  --asset-dir artifacts/raw_results/assembly_opt_ladder/coo_chunk4_np32_maxit20/assets \
  --report-path artifacts/raw_results/assembly_opt_ladder/coo_chunk4_np32_maxit20/REPORT.md
```

## Notes

- Rows marked `status=failed` on this page are historical fixed-work or
  alternative-profile rows that hit an intentional iteration cap. They do not
  indicate a solver crash.
- The `32`-rank Newton diagnostic also ends with `status=failed`, but there it
  means the run exhausted the requested `20` Newton iterations rather than
  breaking in PETSc or JAX.
- The current problem-card result for source-parity and field quality is still
  the from-scratch `P2(L1), lambda = 1.6` solve documented on
  [Plasticity3D](../problems/Plasticity3D.md).
- The current maintained parallel-scaling recommendation for the 3D refined path
  is the converged `P4(L1_2), lambda = 1.0` `local_constitutiveAD + local_pmg`
  study documented at the top of this page.
- The older `lambda = 1.5` fixed-work and capped-Newton sections remain here as
  historical backend and diagnostic context.
