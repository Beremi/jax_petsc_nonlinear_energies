# Claim Audit

This note records the source used for each externally sourced scientific claim
family in the manuscript. Repository-specific benchmark formulas, load paths,
and algorithmic surrogates are intentionally not assigned external citations;
those are presented in the paper as implementation choices of `fenics_nonlinear_energies`.

| Manuscript area | Claim family | Source | Locator used for verification |
| --- | --- | --- | --- |
| Introduction, Related Work | High-level symbolic FEM with compiled kernels and parallel execution in the FEniCS family | `logg2012fenicsbook` | Book overview and compiler/workflow chapters |
| Introduction, Related Work | DOLFINx as the next-generation FEniCS environment with data-oriented design, extensibility, and parallel support | `baratta2025dolfinx` | Zenodo preprint abstract and introduction |
| Introduction, Related Work | Automated adjoints for high-level transient finite-element programs | `farrell2013dolfinadjoint` | Abstract and Sections 1--2 |
| Introduction, Related Work | Maintained pyadjoint workflow for automated adjoints in FEniCS/Firedrake | `mitusch2019pyadjoint` | JOSS abstract |
| Introduction, Related Work | `cashocs` v2 covers adjoint-based shape optimization, optimal control, topology optimization, and MPI-aware workflows | `blauth2023cashocsv2` | Abstract and software-feature summary |
| Introduction, Related Work | Automatic differentiation background in scientific computing and machine learning | `baydin2018autodiff` | Survey abstract and introductory sections |
| Introduction, Related Work | JAX as a high-level tracing / program-transformation environment | `jax2018` | SysML paper abstract |
| Introduction, Related Work | JAX-FEM as a differentiable GPU-accelerated 3D finite-element solver for inverse design and mechanics workflows | `xue2023jaxfem` | Abstract and introduction |
| Introduction, Related Work | AutoPDEx as a JAX-based PDE solver with nonlinear minimization and implicit-differentiation support | `bode2025autopdex` | JOSS abstract |
| Introduction, Related Work | JAX-CPFEM as a differentiable crystal-plasticity FEM platform with GPU emphasis | `hu2025jaxcpfem` | Abstract and introduction |
| Introduction, Related Work | Firedrake--JAX bridge differentiates PDE solves via tangent-linear and adjoint equations instead of low-level solver tracing | `yashchuk2023bringing` | arXiv abstract |
| Introduction, Related Work | FEniCSx external operators plus algorithmic AD for general constitutive models, including Mohr--Coulomb examples | `latyshev2025externaloperators` | Abstract and constitutive-example sections |
| Introduction, Related Work | FEniTop as a parallel FEniCSx topology-optimization implementation | `jia2024fenitop` | Abstract and implementation overview |
| Related Work, Topology benchmark | Classical SIMP / educational topology optimization background | `sigmund2001topology` | Abstract and method description |
| Related Work, Topology benchmark | Topology optimization textbook background | `bendsoe2003topology` | Introductory chapters on SIMP and compliance topology optimization |
| Related Work, Topology benchmark | Compact modern 99-line compliance topology optimization workflow | `ferrari2020top99` | Abstract and method description |
| Related Work, Topology benchmark | Filter regularization in topology optimization | `bourdin2001filters` | Abstract and formulation sections |
| Related Work, Methodology | Sparse Jacobian recovery by directional probing | `curtis1974sparsejacobian` | Full note |
| Related Work, Methodology | Sparse Jacobian / Hessian recovery via graph coloring | `coleman1983sparsejacobian`, `coleman1984sparsehessian` | Full articles |
| Related Work, Methodology | PETSc as scalable vector/matrix/Krylov/nonlinear infrastructure | `balay1997petsc`, `petsc2024web` | PETSc chapter and current PETSc documentation front page |
| Benchmarks: $p$-Laplace | Weak formulation in `W_0^{1,p}` and minimizer/weak-solution equivalence for the stationary problem | `lindqvist2019plaplace` | Chapter 2, “The Dirichlet Problem and Weak Solutions” |
| Benchmarks: Ginzburg--Landau | Historical origin of the Ginzburg--Landau model class | `ginzburg1950theory` | Original article record and bibliographic metadata |
| Benchmarks: Hyperelasticity | Finite-strain kinematics, deformation gradient notation, Piola stress, and hyperelastic background | `bonet2008nonlinear` | Chapters 4--6 |
| Benchmarks: Plasticity2D / Plasticity3D | Incremental elastoplasticity, return mapping, and constitutive linearization as the correct continuum reference frame | `simo1985consistent`, `simo1998compinel` | Simo--Taylor abstract and Computational Inelasticity Chapters 1, 3, and 5 |
| Benchmarks: Plasticity2D / Plasticity3D | Davis strength-reduction background and Davis A/B/C discussion used by later slope-stability papers | `davis1968plasticity`, `tschuchnigg2015nonassociated` | Davis chapter bibliographic record; Tschuchnigg Section 3.3.2 and reference list |
| Related Work, Benchmarks: Plasticity2D / Plasticity3D | Mohr--Coulomb return mapping, nonsmooth constitutive operators, and consistent tangents as the incremental-history reference context | `sysala2017returnmapping` | Metadata/title verified only from the CAS ASEP record and reference metadata; no cached full text is present |
| Introduction, Related Work, Benchmarks: Plasticity2D / Plasticity3D | Modified shear-strength reduction and variational / optimization viewpoints for slope stability | `sysala2021optimization` | Source-verified from cached PDF: abstract, Davis-modification discussion, and OPT-MSSR formulation |
| Introduction, Related Work, Benchmarks: Plasticity3D | Published 3D slope-stability source-family context with continuation and iterative-solver evidence | `sysala2025advancedcontinuation` | Source-verified from cached PDF: abstract, method sections, and reported 3D SSR source-family experiments |
| Related Work, Benchmarks: Plasticity3D | Convex optimization problems motivated by geotechnical stability analysis | `sysala2025convexoptimization` | Source-verified from cached PDF: abstract and problem-formulation sections |
| Methodology | Armijo backtracking, line-search Newton, and trust-region globalization terminology | `nocedal2006numerical` | Chapters 2--4, especially Section 3.1 |

Sysala-family evidence level: the Sysala papers support literature context,
source-family framing, and continuum/reference-method positioning. They do not
directly validate the repository's Plasticity3D endpoint surrogate unless a
numeric artifact comparison for the same case is separately verified.
