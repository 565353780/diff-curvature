from pytorch3d.utils import ico_sphere, torus
from pytorch3d.ops import sample_points_from_meshes

from diff_curvature.Method.mesh_geometry import normalize_mesh
from diff_curvature.Module.diff_global_pc import DiffGlobalPC


def demo():
    mesh_temp = ico_sphere(4, device="cuda:0")
    # mesh_temp = torus(r=1, R=2, sides=40, rings=20, device='cuda:0')

    mesh_temp = normalize_mesh(mesh_temp)
    sample_points = sample_points_from_meshes(mesh_temp, 100000, return_normals=False)
    sample_points = sample_points.to("cuda:0")

    pcl_geometry = DiffGlobalPC(sample_points, k=50)
    print(pcl_geometry.differentiable_euler_number(local_W=128, n_temp=2000))
