import os
import torch
import numpy as np
import open3d as o3d
from typing import Union
from pytorch3d.structures import Meshes

from diff_curvature.Method.Mesh.curvature import (
    get_gaussian_curvature_vertices_packed,
    get_gaussian_curvature_vertices_from_face_packed,
    get_gaussian_curvature_faces_packed,
    get_mean_curvature_vertices_packed,
    get_mean_curvature_faces_packed,
    get_total_curvature_vertices_packed,
    get_total_curvature_faces_packed,
)
from diff_curvature.Method.render import renderVertexCurvatures


class MeshCurvature(object):
    def __init__(
        self, mesh_file_path: Union[str, None] = None, device: str = "cpu"
    ) -> None:
        self.mesh = None

        if mesh_file_path is not None:
            self.loadMesh(mesh_file_path, device)
        return

    def reset(self) -> bool:
        self.mesh = None
        return True

    def isValid(self) -> bool:
        if self.mesh is None:
            return False

        return True

    def loadMesh(self, mesh_file_path: str, device: str = "cpu") -> bool:
        self.reset()

        if not os.path.exists(mesh_file_path):
            print("[ERROR][MeshCurvature::loadMesh]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        mesh_o3d = o3d.io.read_triangle_mesh(mesh_file_path)

        vertices = torch.from_numpy(np.asarray(mesh_o3d.vertices).astype(np.float32))
        faces = torch.from_numpy(np.asarray(mesh_o3d.triangles).astype(np.int64))

        self.mesh = Meshes(verts=[vertices], faces=[faces]).to(device)
        return True

    def toGaussV(self) -> torch.Tensor:
        assert self.isValid()
        return get_gaussian_curvature_vertices_packed(self.mesh).flatten()

    def toGaussFV(self) -> torch.Tensor:
        assert self.isValid()
        return get_gaussian_curvature_vertices_from_face_packed(self.mesh).flatten()

    def toGaussF(self) -> torch.Tensor:
        assert self.isValid()
        return get_gaussian_curvature_faces_packed(self.mesh).flatten()

    def toMeanV(self) -> torch.Tensor:
        assert self.isValid()
        return get_mean_curvature_vertices_packed(self.mesh).flatten()

    def toMeanF(self) -> torch.Tensor:
        assert self.isValid()
        return get_mean_curvature_faces_packed(self.mesh).flatten()

    def toTotalV(self) -> torch.Tensor:
        assert self.isValid()
        return get_total_curvature_vertices_packed(self.mesh).flatten()

    def toTotalF(self) -> torch.Tensor:
        assert self.isValid()
        return get_total_curvature_faces_packed(self.mesh).flatten()

    def render(self, curvatures: torch.Tensor) -> bool:
        if not self.isValid():
            print("[ERROR][MeshCurvature::render]")
            print("\t isValid failed!")
            return False

        if curvatures.shape[0] != self.mesh._V:
            print("[ERROR][MeshCurvature::render]")
            print("\t only support vertex curvatures now!")
            return False

        renderVertexCurvatures(
            self.mesh.verts_packed().cpu().numpy(),
            self.mesh.faces_packed().cpu().numpy(),
            curvatures.cpu().numpy(),
        )
        return True
