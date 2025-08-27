import torch
import numpy as np
import open3d as o3d
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
from typing import Union


def _translation(x, y, z, device):
    return torch.tensor(
        [[1.0, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], device=device
    )  # 4,4


def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4, 4], device=device)
    p[0, 0] = 2 * n / (r - l)
    p[0, 2] = (r + l) / (r - l)
    p[1, 1] = 2 * n / (t - b) * (-1 if flip_y else 1)
    p[1, 2] = (t + b) / (t - b)
    p[2, 2] = -(f + n) / (f - n)
    p[2, 3] = -(2 * f * n) / (f - n)
    p[3, 2] = -1
    return p  # 4,4


def make_star_cameras(
    az_count,
    pol_count,
    distance: float = 10.0,
    r=None,
    n=None,
    f=None,
    image_size=[512, 512],
    device="cuda",
):
    if r is None:
        r = 1 / distance
    if n is None:
        n = 1
    if f is None:
        f = 50
    A = az_count
    P = pol_count
    C = A * P

    phi = torch.arange(0, A) * (2 * torch.pi / A)
    phi_rot = torch.eye(3, device=device)[None, None].expand(A, 1, 3, 3).clone()
    phi_rot[:, 0, 2, 2] = phi.cos()
    phi_rot[:, 0, 2, 0] = -phi.sin()
    phi_rot[:, 0, 0, 2] = phi.sin()
    phi_rot[:, 0, 0, 0] = phi.cos()

    theta = torch.arange(1, P + 1) * (torch.pi / (P + 1)) - torch.pi / 2
    theta_rot = torch.eye(3, device=device)[None, None].expand(1, P, 3, 3).clone()
    theta_rot[0, :, 1, 1] = theta.cos()
    theta_rot[0, :, 1, 2] = -theta.sin()
    theta_rot[0, :, 2, 1] = theta.sin()
    theta_rot[0, :, 2, 2] = theta.cos()

    mv = torch.empty((C, 4, 4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(C, 3, 3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r, device, n=n, f=f)


def _warmup(glctx):
    # windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device="cuda", **kwargs)

    pos = tensor(
        [[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]],
        dtype=torch.float32,
    )
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])


def toTriangleSoup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    all_vertices = []
    all_triangles = []

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        v0, v1, v2 = vertices[tri]

        base_idx = len(all_vertices)
        all_vertices.extend([v0, v1, v2])
        all_triangles.append([base_idx, base_idx + 1, base_idx + 2])

    all_vertices = np.array(all_vertices)
    all_triangles = np.array(all_triangles)

    triangle_soup = o3d.geometry.TriangleMesh()
    triangle_soup.vertices = o3d.utility.Vector3dVector(all_vertices)
    triangle_soup.triangles = o3d.utility.Vector3iVector(all_triangles)
    return triangle_soup


def paintTriangleSoup(
    triangle_soup: o3d.geometry.TriangleMesh,
    triangle_colors: Union[np.ndarray, list],
) -> bool:
    triangles = np.asarray(triangle_soup.triangles)

    vertex_colors = np.zeros_like(np.asarray(triangle_soup.vertices))
    for i, triangle in enumerate(triangles):
        vertex_colors[triangle] = triangle_colors[i]

    triangle_soup.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return True


def renderTriangleCurvatures(
    vertices: np.ndarray, triangles: np.ndarray, vertex_curvatures: np.ndarray
) -> bool:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    print("曲率统计信息:")
    print(f"  最小值: {np.min(vertex_curvatures):.6f}")
    print(f"  最大值: {np.max(vertex_curvatures):.6f}")
    print(f"  平均值: {np.mean(vertex_curvatures):.6f}")
    print(f"  中位数: {np.median(vertex_curvatures):.6f}")
    print(f"  标准差: {np.std(vertex_curvatures):.6f}")

    min_curv = np.min(vertex_curvatures)
    max_curv = np.max(vertex_curvatures)

    if np.isclose(max_curv, min_curv):
        print("[WARN][render::renderVertexCurvatures]")
        print("\t all curvatures are close to 0!")
        normalized_curvatures = np.ones_like(vertex_curvatures) * 0.5
    else:
        normalized_curvatures = (vertex_curvatures - min_curv) / (max_curv - min_curv)

    print("normalized 曲率统计信息:")
    print(f"  最小值: {np.min(normalized_curvatures):.6f}")
    print(f"  最大值: {np.max(normalized_curvatures):.6f}")
    print(f"  平均值: {np.mean(normalized_curvatures):.6f}")
    print(f"  中位数: {np.median(normalized_curvatures):.6f}")
    print(f"  标准差: {np.std(normalized_curvatures):.6f}")

    triangle_soup = toTriangleSoup(mesh)

    cmap = plt.get_cmap("jet")
    triangle_colors = np.zeros([triangles.shape[0], 3])
    for i, curv in enumerate(normalized_curvatures):
        triangle_colors[i] = cmap(curv)[:3]

    paintTriangleSoup(triangle_soup, triangle_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(triangle_soup)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 5.0

    vis.update_renderer()
    vis.poll_events()
    vis.update_renderer()

    vis.run()
    vis.destroy_window()
    return True
