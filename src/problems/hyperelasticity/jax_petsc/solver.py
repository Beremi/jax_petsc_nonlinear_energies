"""
HyperElasticity 3D — solver logic (DOF-partitioned JAX + PETSc).

Provides ``PROFILE_DEFAULTS`` and ``run(args)`` which runs all load steps.
CLI entry point (argparse) is in ``solve_HE_dof.py``.
"""

import gc
import time

import numpy as np
from mpi4py import MPI

from src.core.benchmark.repair import build_retry_attempts, needs_solver_repair
from src.core.benchmark.state_export import export_hyperelasticity_state_npz
from src.core.petsc.minimizers import newton
from src.core.petsc.load_step_driver import (
    attempts_from_tuples,
    build_load_step_result,
    run_load_steps,
)
from src.core.petsc.trust_ksp import ksp_cg_set_radius
from src.problems.hyperelasticity.jax_petsc.parallel_hessian_dof import (
    LocalColoringAssembler,
    ParallelDOFHessianAssembler,
)
from src.problems.hyperelasticity.jax_petsc.reordered_element_assembler import (
    HEReorderedElementAssembler,
)
from src.problems.hyperelasticity.jax_petsc.multigrid import (
    HEPmgSmootherConfig,
    build_he_pmg_hierarchy,
    choose_he_pmg_coarsest_level,
    configure_he_pmg,
)
from src.problems.hyperelasticity.support.mesh import (
    MeshHyperElasticity3D,
    build_procedural_hyperelasticity_export_params,
    load_rank_local_hyperelasticity,
    local_dirichlet_values_from_reference,
    reordered_free_to_total_dofs,
)
from src.problems.hyperelasticity.support.rotate_boundary import (
    rotate_right_face_from_reference,
)
from src.core.petsc.gamg import build_gamg_coordinates


PROFILE_DEFAULTS = {
    "reference": {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "pc_setup_on_ksp_cap": True,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
    "performance": {
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "pc_setup_on_ksp_cap": True,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
}


def _gather_full_free_original(assembler, vec) -> np.ndarray:
    owned = np.asarray(vec.array[:], dtype=np.float64)
    if hasattr(assembler, "part") and hasattr(assembler.part, "get_u_full"):
        full_reordered = assembler.part.get_u_full(owned)
        return np.asarray(assembler.part.reordered_to_original(full_reordered), dtype=np.float64)
    if hasattr(assembler, "_allgather_full_owned") and hasattr(assembler, "layout"):
        full_reordered, _ = assembler._allgather_full_owned(owned)
        if bool(getattr(assembler, "_formula_layout", False)):
            grid = assembler.params["_he_grid"]
            mode = str(assembler.params["_distributed_reorder_mode"])
            total_dofs = reordered_free_to_total_dofs(
                np.arange(int(assembler.layout.n_free), dtype=np.int64),
                grid,
                mode,
            )
            original = np.empty_like(full_reordered)
            node = total_dofs // 3
            comp = total_dofs % 3
            ix = node % int(grid.nx1)
            plane = node // int(grid.nx1)
            iy = plane % int(grid.ny1)
            iz = plane // int(grid.ny1)
            block = (iz * int(grid.ny1) + iy) * (int(grid.nx) - 1) + (ix - 1)
            original_index = 3 * block + comp
            original[original_index] = full_reordered
            return original
        full_original = np.empty_like(full_reordered)
        full_original[np.asarray(assembler.layout.perm, dtype=np.int64)] = full_reordered
        return full_original
    raise TypeError(f"Unsupported assembler type for state export: {type(assembler).__name__}")


def _export_state_if_requested(args, assembler, params, vec, step_records, comm) -> None:
    state_out = str(getattr(args, "state_out", "") or "")
    if not state_out:
        return

    full_free_original = _gather_full_free_original(assembler, vec)
    export_params = params
    if bool(params.get("_distributed_local_data", False)):
        if comm.rank != 0:
            return
        mesh_source = str(params.get("_distributed_mesh_source", "hdf5"))
        if mesh_source == "procedural":
            export_params = build_procedural_hyperelasticity_export_params(args.level)
        else:
            mesh_obj = MeshHyperElasticity3D(args.level)
            export_params, _, _ = mesh_obj.get_data()

    if step_records:
        final_angle = float(step_records[-1]["angle"])
        final_energy = float(step_records[-1]["energy"])
        completed_steps = int(len(step_records))
        full_state = rotate_right_face_from_reference(
            export_params["u_0_ref"],
            export_params["nodes2coord"],
            final_angle,
            export_params["right_nodes"],
        )
    else:
        final_energy = None
        completed_steps = 0
        full_state = np.asarray(export_params["u_0_ref"], dtype=np.float64).copy()

    full_state = np.asarray(full_state, dtype=np.float64).copy()
    full_state[np.asarray(export_params["freedofs"], dtype=np.int64)] = full_free_original

    if comm.rank == 0:
        export_hyperelasticity_state_npz(
            state_out,
            coords_ref=np.asarray(export_params["nodes2coord"], dtype=np.float64),
            x_final=full_state.reshape((-1, 3)),
            tetrahedra=np.asarray(export_params["elems_scalar"], dtype=np.int32),
            mesh_level=int(args.level),
            total_steps=int(args.total_steps),
            energy=final_energy,
            metadata={
                "solver_family": "hyperelasticity_jax_petsc_element",
                "mpi_ranks": int(comm.Get_size()),
                "completed_steps": int(completed_steps),
            },
        )


def _summarize_rank_memory(rank_summaries) -> dict[str, float | int | list[dict[str, object]]]:
    if not rank_summaries:
        return {"ranks": 0, "rank_summaries": []}

    rows: list[dict[str, object]] = []
    for rank, summary in enumerate(rank_summaries):
        row = dict(summary)
        row["rank"] = int(rank)
        rows.append(row)

    aggregate_keys = (
        "local_elements",
        "local_overlap_dofs",
        "owned_nnz",
        "layout_gib",
        "local_overlap_gib",
        "scatter_gib",
        "owned_hessian_values_gib",
        "petsc_owned_values_gib",
        "local_backend_gib",
        "tracked_total_gib",
    )
    out: dict[str, float | int | list[dict[str, object]]] = {
        "ranks": int(len(rows)),
        "rank_summaries": rows,
    }
    for key in aggregate_keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        out[f"{key}_min"] = float(min(values))
        out[f"{key}_max"] = float(max(values))
        out[f"{key}_total"] = float(sum(values))
    return out


def _resolve_linear_settings(args):
    settings = dict(PROFILE_DEFAULTS[args.profile])
    overrides = {
        "ksp_type": args.ksp_type,
        "pc_type": args.pc_type,
        "ksp_rtol": args.ksp_rtol,
        "ksp_max_it": args.ksp_max_it,
        "pc_setup_on_ksp_cap": args.pc_setup_on_ksp_cap,
        "gamg_threshold": args.gamg_threshold,
        "gamg_agg_nsmooths": args.gamg_agg_nsmooths,
        "use_near_nullspace": args.use_near_nullspace,
        "gamg_set_coordinates": args.gamg_set_coordinates,
        "reorder": args.reorder,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def _pc_options(settings):
    opts = {}
    if settings["pc_type"] == "gamg":
        opts["pc_gamg_threshold"] = float(settings["gamg_threshold"])
        opts["pc_gamg_agg_nsmooths"] = int(settings["gamg_agg_nsmooths"])
    return opts


def run(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    total_runtime_start = time.perf_counter()

    settings = _resolve_linear_settings(args)
    pc_options = _pc_options(settings)
    use_element_assembly = args.assembly_mode == "element"
    element_reorder_mode = str(
        getattr(args, "element_reorder_mode", None) or "block_xyz"
    )
    local_hessian_mode = str(
        getattr(args, "local_hessian_mode", None) or "element"
    )
    problem_build_mode = str(
        getattr(
            args,
            "problem_build_mode",
            "rank_local" if use_element_assembly else "replicated",
        )
        or ("rank_local" if use_element_assembly else "replicated")
    )
    distribution_strategy = str(
        getattr(
            args,
            "distribution_strategy",
            "overlap_p2p" if problem_build_mode == "rank_local" else "overlap_allgather",
        )
        or ("overlap_p2p" if problem_build_mode == "rank_local" else "overlap_allgather")
    )
    assembly_backend = str(
        getattr(
            args,
            "assembly_backend",
            "coo_local" if problem_build_mode == "rank_local" else "coo",
        )
        or ("coo_local" if problem_build_mode == "rank_local" else "coo")
    )
    mesh_source = str(
        getattr(
            args,
            "mesh_source",
            "procedural" if problem_build_mode == "rank_local" else "hdf5",
        )
        or ("procedural" if problem_build_mode == "rank_local" else "hdf5")
    )

    mesh_obj = None
    if problem_build_mode == "rank_local":
        if not use_element_assembly:
            raise ValueError("problem_build_mode='rank_local' is supported only for element assembly")
        if distribution_strategy != "overlap_p2p":
            raise ValueError("rank-local HyperElasticity requires distribution_strategy='overlap_p2p'")
        if assembly_backend != "coo_local":
            raise ValueError("rank-local HyperElasticity requires assembly_backend='coo_local'")
        if local_hessian_mode != "element":
            raise ValueError("rank-local HyperElasticity requires local_hessian_mode='element'")
        params, adjacency, u_init = load_rank_local_hyperelasticity(
            int(args.level),
            comm=comm,
            reorder_mode=element_reorder_mode,
            mesh_source=mesh_source,
        )
    elif problem_build_mode == "replicated":
        if mesh_source != "hdf5":
            raise ValueError(
                "problem_build_mode='replicated' currently supports only "
                "mesh_source='hdf5'"
            )
        mesh_obj = MeshHyperElasticity3D(args.level)
        params, adjacency, u_init = mesh_obj.get_data()
    else:
        raise ValueError(
            f"Unsupported HyperElasticity problem_build_mode={problem_build_mode!r}"
        )

    setup_start = time.perf_counter()
    if use_element_assembly:
        if not args.local_coloring:
            raise ValueError("--assembly_mode element requires --local_coloring")
        assembler = HEReorderedElementAssembler(
            params=params,
            comm=comm,
            adjacency=adjacency,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            pc_options=pc_options,
            reorder_mode=element_reorder_mode,
            local_hessian_mode=local_hessian_mode,
            use_abs_det=bool(args.use_abs_det),
            distribution_strategy=distribution_strategy,
            assembly_backend=assembly_backend,
        )
    else:
        assembler_cls = (
            LocalColoringAssembler if args.local_coloring else ParallelDOFHessianAssembler
        )
        assembler_kwargs = dict(
            params=params,
            comm=comm,
            adjacency=adjacency,
            coloring_trials_per_rank=args.coloring_trials,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            pc_options=pc_options,
            reorder=bool(settings["reorder"]),
            use_abs_det=bool(args.use_abs_det),
        )
        if args.local_coloring:
            assembler_kwargs["hvp_eval_mode"] = str(args.hvp_eval_mode)
        assembler = assembler_cls(**assembler_kwargs)
        assembler.A.setBlockSize(3)

    # The assembler has extracted its local/owned sparsity pattern by here.
    # Keep the problem arrays in ``params``, but release the replicated mesh
    # wrapper and global HDF5 adjacency before PETSc/JAX setup grows memory.
    del mesh_obj, adjacency
    gc.collect()

    setup_time = time.perf_counter() - setup_start
    local_assembler_setup = assembler.setup_summary() if use_element_assembly else {}
    local_assembler_memory = assembler.memory_summary() if use_element_assembly else {}
    gathered_assembler_setup = comm.gather(local_assembler_setup, root=0)
    gathered_assembler_memory = comm.gather(local_assembler_memory, root=0)
    if rank == 0:
        assembler_memory_report = _summarize_rank_memory(gathered_assembler_memory)
        assembler_setup_report = [
            {"rank": int(idx), **dict(summary)}
            for idx, summary in enumerate(gathered_assembler_setup or [])
        ]
    else:
        assembler_memory_report = {"ranks": 0, "rank_summaries": []}
        assembler_setup_report = []

    if "_distributed_u_init_owned" in params:
        x = assembler.create_vec(np.asarray(params["_distributed_u_init_owned"], dtype=np.float64))
    else:
        u_init_reordered = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
        x = assembler.create_vec(u_init_reordered)
    x_step_start = x.duplicate()

    ksp = assembler.ksp
    A = assembler.A
    pc = ksp.getPC()
    pmg_hierarchy = None
    pmg_metadata: dict[str, object] | None = None
    if use_element_assembly and str(settings["pc_type"]) == "mg":
        coarsest_level = choose_he_pmg_coarsest_level(
            finest_level=int(args.level),
            n_ranks=int(nprocs),
            requested=getattr(args, "he_pmg_coarsest_level", 1),
            min_dofs_per_rank=int(getattr(args, "he_pmg_auto_min_dofs_per_rank", 128)),
        )
        t_pmg0 = time.perf_counter()
        pmg_hierarchy = build_he_pmg_hierarchy(
            finest_level=int(args.level),
            coarsest_level=int(coarsest_level),
            reorder_mode=element_reorder_mode,
            comm=comm,
        )
        configure_he_pmg(
            ksp,
            pmg_hierarchy,
            smoother=HEPmgSmootherConfig(
                ksp_type=str(getattr(args, "he_pmg_smoother_ksp_type", "chebyshev")),
                pc_type=str(getattr(args, "he_pmg_smoother_pc_type", "jacobi")),
                steps=int(getattr(args, "he_pmg_smoother_steps", 2)),
            ),
            coarse_ksp_type=(
                None
                if getattr(args, "he_pmg_coarse_ksp_type", None) in {None, ""}
                else str(getattr(args, "he_pmg_coarse_ksp_type"))
            ),
            coarse_pc_type=str(getattr(args, "he_pmg_coarse_pc_type", "hypre")),
            coarse_redundant_number=int(
                getattr(args, "he_pmg_coarse_redundant_number", 0)
            ),
            coarse_telescope_reduction_factor=int(
                getattr(args, "he_pmg_coarse_telescope_reduction_factor", 0)
            ),
            coarse_factor_solver_type=(
                None
                if getattr(args, "he_pmg_coarse_factor_solver_type", None) in {None, ""}
                else str(getattr(args, "he_pmg_coarse_factor_solver_type"))
            ),
            coarse_hypre_nodal_coarsen=int(
                getattr(args, "he_pmg_coarse_hypre_nodal_coarsen", 6)
            ),
            coarse_hypre_vec_interp_variant=int(
                getattr(args, "he_pmg_coarse_hypre_vec_interp_variant", 3)
            ),
            coarse_hypre_strong_threshold=getattr(
                args, "he_pmg_coarse_hypre_strong_threshold", None
            ),
            coarse_hypre_coarsen_type=getattr(
                args, "he_pmg_coarse_hypre_coarsen_type", None
            ),
            coarse_hypre_max_iter=int(
                getattr(args, "he_pmg_coarse_hypre_max_iter", 2)
            ),
            coarse_hypre_tol=float(getattr(args, "he_pmg_coarse_hypre_tol", 0.0)),
            coarse_hypre_relax_type_all=getattr(
                args,
                "he_pmg_coarse_hypre_relax_type_all",
                "symmetric-SOR/Jacobi",
            ),
            galerkin=str(getattr(args, "he_pmg_galerkin", "both")),
        )
        pmg_metadata = dict(pmg_hierarchy.build_metadata)
        pmg_metadata["configure_time"] = float(time.perf_counter() - t_pmg0)
        pmg_metadata["coarsest_level_resolved"] = int(coarsest_level)
        pmg_metadata["coarsest_level_requested"] = str(
            getattr(args, "he_pmg_coarsest_level", 1)
        )
        pmg_metadata["auto_min_dofs_per_rank"] = int(
            getattr(args, "he_pmg_auto_min_dofs_per_rank", 128)
        )
        pmg_metadata["smoother"] = {
            "ksp_type": str(getattr(args, "he_pmg_smoother_ksp_type", "chebyshev")),
            "pc_type": str(getattr(args, "he_pmg_smoother_pc_type", "jacobi")),
            "steps": int(getattr(args, "he_pmg_smoother_steps", 2)),
        }
        pmg_metadata["coarse_solver"] = {
            "ksp_type": str(getattr(args, "he_pmg_coarse_ksp_type", "") or ""),
            "pc_type": str(getattr(args, "he_pmg_coarse_pc_type", "hypre")),
            "redundant_number": int(
                getattr(args, "he_pmg_coarse_redundant_number", 0)
            ),
            "telescope_reduction_factor": int(
                getattr(args, "he_pmg_coarse_telescope_reduction_factor", 0)
            ),
            "factor_solver_type": str(
                getattr(args, "he_pmg_coarse_factor_solver_type", "") or ""
            ),
        }

    gamg_coords = None
    if settings["pc_type"] == "gamg" and settings["gamg_set_coordinates"]:
        if "_distributed_owned_block_coordinates" in params:
            gamg_coords = np.asarray(
                params["_distributed_owned_block_coordinates"], dtype=np.float64
            )
        else:
            gamg_coords = build_gamg_coordinates(
                assembler.part,
                params["freedofs"],
                params["nodes2coord"],
                block_size=3,
            )

    rotation_per_iter = 4.0 * 2.0 * np.pi / float(args.total_steps)
    ls_primary = (float(args.linesearch_a), float(args.linesearch_b))
    line_search = str(getattr(args, "line_search", "golden_fixed"))
    use_trust_region = bool(getattr(args, "use_trust_region", False))
    trust_radius_init = float(getattr(args, "trust_radius_init", 1.0))
    trust_radius_min = float(getattr(args, "trust_radius_min", 1e-8))
    trust_radius_max = float(getattr(args, "trust_radius_max", 1e6))
    trust_shrink = float(getattr(args, "trust_shrink", 0.5))
    trust_expand = float(getattr(args, "trust_expand", 1.5))
    trust_eta_shrink = float(getattr(args, "trust_eta_shrink", 0.05))
    trust_eta_expand = float(getattr(args, "trust_eta_expand", 0.75))
    trust_max_reject = int(getattr(args, "trust_max_reject", 6))
    trust_subproblem_line_search = bool(
        getattr(args, "trust_subproblem_line_search", False)
    )
    step_time_limit_s = getattr(args, "step_time_limit_s", None)
    trust_ksp_subproblem = bool(
        use_trust_region and str(settings["ksp_type"]).lower() in {"stcg", "nash", "gltr"}
    )

    if rank == 0 and not args.quiet:
        print(
            f"HE 3D DOF solver | level={args.level} np={nprocs} profile={args.profile} "
            f"ksp={settings['ksp_type']} pc={settings['pc_type']} setup={setup_time:.3f}s",
            flush=True,
        )

    linear_timing_records: list[dict[str, object]] = []
    linear_iters_this_attempt: list[int] = []
    force_pc_setup_next = True
    used_ksp_rtol = float(settings["ksp_rtol"])
    used_ksp_max_it = int(settings["ksp_max_it"])

    def _assemble_and_solve(vec, rhs, sol, trust_radius=None):
        nonlocal force_pc_setup_next, gamg_coords

        if trust_radius is not None:
            ksp_cg_set_radius(ksp, float(trust_radius))

        t_asm0 = time.perf_counter()
        u_owned = np.array(vec.array[:], dtype=np.float64)
        if use_element_assembly:
            assembler.assemble_hessian(u_owned)
        else:
            assembler.assemble_hessian(u_owned, variant=2)
        asm_total_time = time.perf_counter() - t_asm0

        asm_details = {}
        if assembler.iter_timings:
            asm_details = dict(assembler.iter_timings[-1])
        asm_details["assembly_total_time"] = float(asm_total_time)

        t_setop0 = time.perf_counter()
        ksp.setOperators(A)
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords = None
        t_setop = time.perf_counter() - t_setop0

        t_tol0 = time.perf_counter()
        ksp.setTolerances(rtol=float(used_ksp_rtol), max_it=int(used_ksp_max_it))
        t_tol = time.perf_counter() - t_tol0

        t_setup0 = time.perf_counter()
        if settings["pc_setup_on_ksp_cap"]:
            if force_pc_setup_next:
                ksp.setUp()
                force_pc_setup_next = False
        else:
            ksp.setUp()
        t_setup = time.perf_counter() - t_setup0

        t_solve0 = time.perf_counter()
        ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t_solve0
        ksp_its = int(ksp.getIterationNumber())
        linear_iters_this_attempt.append(ksp_its)

        if settings["pc_setup_on_ksp_cap"] and ksp_its >= int(used_ksp_max_it):
            force_pc_setup_next = True

        if args.save_linear_timing:
            record = {
                "assemble_total_time": float(asm_total_time),
                "assemble_p2p_exchange": float(asm_details.get("p2p_exchange", 0.0)),
                "assemble_hvp_compute": float(asm_details.get("hvp_compute", 0.0)),
                "assemble_extraction": float(asm_details.get("extraction", 0.0)),
                "assemble_coo_assembly": float(asm_details.get("coo_assembly", 0.0)),
                "assemble_n_hvps": int(asm_details.get("n_hvps", 0)),
                "setop_time": float(t_setop),
                "set_tolerances_time": float(t_tol),
                "pc_setup_time": float(t_setup),
                "solve_time": float(t_solve),
                "linear_total_time": float(
                    asm_total_time + t_setop + t_tol + t_setup + t_solve
                ),
                "ksp_its": int(ksp_its),
            }
            if trust_radius is not None:
                record["trust_radius"] = float(trust_radius)
            linear_timing_records.append(record)

        return ksp_its

    def hessian_solve_fn(vec, rhs, sol):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=None)

    def trust_subproblem_solve_fn(vec, rhs, sol, trust_radius):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=float(trust_radius))

    attempt_specs = attempts_from_tuples(
        build_retry_attempts(
            retry_on_failure=bool(args.retry_on_failure),
            linesearch_interval=ls_primary,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_max_it=int(settings["ksp_max_it"]),
        )
    )

    def prepare_step(step_ctx):
        if "_distributed_local_data" in params:
            u0_step = local_dirichlet_values_from_reference(params, step_ctx.angle)
        else:
            u0_step = rotate_right_face_from_reference(
                params["u_0_ref"],
                params["nodes2coord"],
                step_ctx.angle,
                params["right_nodes"],
            )
        assembler.update_dirichlet(u0_step)
        x.copy(x_step_start)

    def build_attempts(_step_ctx):
        return attempt_specs

    def solve_attempt(step_ctx, attempt):
        nonlocal force_pc_setup_next, used_ksp_rtol, used_ksp_max_it

        x_step_start.copy(x)
        force_pc_setup_next = True
        linear_iters_this_attempt.clear()
        linear_timing_records.clear()

        used_ksp_rtol = float(attempt.linear_rtol)
        used_ksp_max_it = int(attempt.linear_max_it)

        t0 = time.perf_counter()
        result = newton(
            energy_fn=assembler.energy_fn,
            gradient_fn=assembler.gradient_fn,
            hessian_solve_fn=hessian_solve_fn,
            x=x,
            tolf=float(args.tolf),
            tolg=float(args.tolg),
            tolg_rel=float(args.tolg_rel),
            linesearch_tol=float(args.linesearch_tol),
            linesearch_interval=attempt.linesearch_interval,
            line_search=line_search,
            maxit=int(args.maxit),
            tolx_rel=float(args.tolx_rel),
            tolx_abs=float(args.tolx_abs),
            require_all_convergence=True,
            fail_on_nonfinite=True,
            verbose=(not args.quiet),
            comm=comm,
            ghost_update_fn=None,
            hessian_matvec_fn=lambda _x, vin, vout: assembler.A.mult(vin, vout),
            trust_subproblem_solve_fn=(
                trust_subproblem_solve_fn if trust_ksp_subproblem else None
            ),
            trust_subproblem_line_search=trust_subproblem_line_search,
            save_history=bool(args.save_history),
            trust_region=use_trust_region,
            trust_radius_init=trust_radius_init,
            trust_radius_min=trust_radius_min,
            trust_radius_max=trust_radius_max,
            trust_shrink=trust_shrink,
            trust_expand=trust_expand,
            trust_eta_shrink=trust_eta_shrink,
            trust_eta_expand=trust_eta_expand,
            trust_max_reject=trust_max_reject,
            step_time_limit_s=step_time_limit_s,
        )
        step_ctx.state["step_time_raw"] = time.perf_counter() - t0
        return result, float(step_ctx.state["step_time_raw"])

    def should_retry(result, _step_ctx):
        return needs_solver_repair(result)

    def build_step_record(step_ctx, result, step_time, attempt):
        step_record = {
            "step": int(step_ctx.step),
            "angle": float(step_ctx.angle),
            "time": float(round(step_time, 6)),
            "nit": int(result["nit"]),
            "linear_iters": int(sum(linear_iters_this_attempt)),
            "energy": float(result["fun"]),
            "message": str(result["message"]),
            "attempt": str(attempt.name),
            "ksp_rtol_used": float(attempt.linear_rtol),
            "ksp_max_it_used": int(attempt.linear_max_it),
            "linesearch_interval_used": [
                float(attempt.linesearch_interval[0]),
                float(attempt.linesearch_interval[1]),
            ],
        }
        if step_time_limit_s is not None:
            step_record["step_time_limit_s"] = float(step_time_limit_s)
            step_record["kill_switch_exceeded"] = bool(
                step_time > float(step_time_limit_s)
            )
        if args.save_history:
            step_record["history"] = result.get("history", [])
        if args.save_linear_timing:
            step_record["linear_timing"] = list(linear_timing_records)
        return step_record

    def on_retry(step_ctx, attempt, _attempt_idx, _total_attempts):
        if rank == 0 and not args.quiet:
            print(
                f"Step {step_ctx.step}: retrying with repair settings "
                f"(rtol={float(attempt.linear_rtol):.3e}, "
                f"ksp_max_it={int(attempt.linear_max_it)}, "
                f"ls=[{attempt.linesearch_interval[0]:.3g},"
                f"{attempt.linesearch_interval[1]:.3g}])",
                flush=True,
            )

    def on_step_complete(step_record, _step_ctx):
        if rank == 0 and not args.quiet:
            print(
                f"step={step_record['step']:3d} angle={step_record['angle']:.6f} "
                f"time={step_record['time']:.3f}s nit={step_record['nit']:3d} "
                f"ksp={step_record['linear_iters']:5d} "
                f"energy={step_record['energy']:.6e} "
                f"[{step_record['message']}]",
                flush=True,
            )

    def should_stop(step_record, _result, step_ctx):
        if args.stop_on_fail and "converged" not in step_record["message"].lower():
            if rank == 0 and not args.quiet:
                print(
                    f"Stopping at step {step_ctx.step} due to failure message.",
                    flush=True,
                )
            return True
        if step_time_limit_s is not None and bool(step_record.get("kill_switch_exceeded")):
            if rank == 0 and not args.quiet:
                print(
                    f"Stopping at step {step_ctx.step} because step time "
                    f"{float(step_ctx.state['step_time_raw']):.3f}s exceeded limit "
                    f"{float(step_time_limit_s):.3f}s",
                    flush=True,
                )
            return True
        return False

    step_records = []

    try:
        step_records = run_load_steps(
            start_step=int(args.start_step),
            num_steps=int(args.steps),
            rotation_per_step=float(rotation_per_iter),
            prepare_step=prepare_step,
            build_attempts=build_attempts,
            solve_attempt=solve_attempt,
            should_retry=should_retry,
            build_step_record=build_step_record,
            should_stop=should_stop,
            on_retry=on_retry,
            on_step_complete=on_step_complete,
        )
    finally:
        try:
            _export_state_if_requested(args, assembler, params, x, step_records, comm)
        finally:
            x_step_start.destroy()
            x.destroy()
            assembler.cleanup()
            if pmg_hierarchy is not None:
                pmg_hierarchy.cleanup()

    return build_load_step_result(
        mesh_level=int(args.level),
        total_dofs=int(params.get("_distributed_total_dofs", len(params.get("u_0", [])))),
        setup_time=setup_time,
        total_runtime_start=total_runtime_start,
        steps=step_records,
        extra={
            "free_dofs": int(assembler.part.n_free),
            "metadata": {
                "profile": args.profile,
                "nprocs": nprocs,
                "nproc_threads": max(1, int(args.nproc)),
                "linear_solver": {
                    "ksp_type": str(settings["ksp_type"]),
                    "pc_type": str(settings["pc_type"]),
                    "ksp_rtol": float(settings["ksp_rtol"]),
                    "ksp_max_it": int(settings["ksp_max_it"]),
                    "pc_setup_on_ksp_cap": bool(settings["pc_setup_on_ksp_cap"]),
                    "gamg_threshold": float(settings["gamg_threshold"]),
                    "gamg_agg_nsmooths": int(settings["gamg_agg_nsmooths"]),
                    "gamg_set_coordinates": bool(settings["gamg_set_coordinates"]),
                    "use_near_nullspace": bool(settings["use_near_nullspace"]),
                    "matrix_block_size": 3,
                    "reorder": bool(settings["reorder"]),
                    "hvp_eval_mode": str(
                        getattr(assembler, "_hvp_eval_mode", "batched")
                    ),
                    "assembly_mode": str(args.assembly_mode),
                    "element_reorder_mode": (
                        element_reorder_mode if use_element_assembly else None
                    ),
                    "local_hessian_mode": (
                        local_hessian_mode if use_element_assembly else None
                    ),
                    "distribution_strategy": str(
                        getattr(assembler, "distribution_strategy", "reduced_free_dofs")
                    ),
                    "assembly_backend": str(getattr(assembler, "assembly_backend", "")),
                    "assembly_backend_requested": str(
                        getattr(assembler, "assembly_backend_requested", "")
                    ),
                    "problem_build_mode": str(problem_build_mode),
                    "mesh_source": str(mesh_source),
                    "rank_local_formula_layout": bool(
                        getattr(assembler, "_formula_layout", False)
                    ),
                    "pmg_hierarchy": pmg_metadata,
                    "assembler_setup_by_rank": assembler_setup_report,
                    "assembler_memory_by_rank": assembler_memory_report,
                    "assembler": assembler.__class__.__name__,
                    "trust_subproblem_solver": (
                        "petsc_ksp" if trust_ksp_subproblem else "reduced_2d"
                    ),
                    "trust_subproblem_line_search": bool(
                        trust_subproblem_line_search
                    ),
                },
                "newton": {
                    "tolf": float(args.tolf),
                    "tolg": float(args.tolg),
                    "tolg_rel": float(args.tolg_rel),
                    "tolx_rel": float(args.tolx_rel),
                    "tolx_abs": float(args.tolx_abs),
                    "maxit": int(args.maxit),
                    "require_all_convergence": True,
                    "fail_on_nonfinite": True,
                    "linesearch_interval": [
                        float(args.linesearch_a),
                        float(args.linesearch_b),
                    ],
                    "linesearch_tol": float(args.linesearch_tol),
                    "line_search": str(line_search),
                    "trust_region": bool(use_trust_region),
                    "trust_radius_init": float(trust_radius_init),
                    "trust_radius_min": float(trust_radius_min),
                    "trust_radius_max": float(trust_radius_max),
                    "trust_subproblem_line_search": bool(
                        trust_subproblem_line_search
                    ),
                    "step_time_limit_s": (
                        None if step_time_limit_s is None else float(step_time_limit_s)
                    ),
                },
                "load_stepping": {
                    "start_step": int(args.start_step),
                    "steps": int(args.steps),
                    "total_steps": int(args.total_steps),
                    "rotation_per_iter": float(rotation_per_iter),
                    "retry_on_failure": bool(args.retry_on_failure),
                },
            },
        },
    )
