"""Support helpers for the 3D heterogeneous slope-stability benchmark."""

from src.problems.slope_stability_3d.support.materials import (
    MaterialSpec,
    heterogenous_materials_qp,
)
from src.problems.slope_stability_3d.support.mesh import (
    DEFAULT_MESH_NAME,
    SlopeStability3DCaseData,
    base_mesh_name_for_name,
    build_case_data_from_raw_mesh,
    build_near_nullspace_modes_3d,
    build_same_mesh_lagrange_case_data,
    clear_same_mesh_case_hdf5_caches,
    ensure_same_mesh_case_hdf5,
    load_case_hdf5,
    load_case_hdf5_fields,
    load_same_mesh_case_hdf5_light,
    load_same_mesh_case_hdf5_rank_local_light,
    load_same_mesh_case_hdf5_rank_local,
    mesh_name_from_raw_filename,
    raw_mesh_path_for_name,
    same_mesh_case_hdf5_path,
    same_mesh_case_name,
    supported_mesh_names,
    write_case_hdf5,
)
from src.problems.slope_stability_3d.support.reduction import (
    davis_reduction_qp,
    davis_b_reduction_qp,
)

__all__ = [
    "DEFAULT_MESH_NAME",
    "MaterialSpec",
    "SlopeStability3DCaseData",
    "base_mesh_name_for_name",
    "build_case_data_from_raw_mesh",
    "build_near_nullspace_modes_3d",
    "build_same_mesh_lagrange_case_data",
    "clear_same_mesh_case_hdf5_caches",
    "davis_b_reduction_qp",
    "davis_reduction_qp",
    "ensure_same_mesh_case_hdf5",
    "heterogenous_materials_qp",
    "load_case_hdf5",
    "load_case_hdf5_fields",
    "load_same_mesh_case_hdf5_light",
    "load_same_mesh_case_hdf5_rank_local_light",
    "load_same_mesh_case_hdf5_rank_local",
    "mesh_name_from_raw_filename",
    "raw_mesh_path_for_name",
    "same_mesh_case_hdf5_path",
    "same_mesh_case_name",
    "supported_mesh_names",
    "write_case_hdf5",
]
