import os
import torch
import trimesh
import warnings
from pytorch3d.io import load_objs_as_meshes

from diff_curvature.Method.io import save_image
from diff_curvature.Method.Mesh.geometry import normalize_mesh
from diff_curvature.Method.render import make_star_cameras
from diff_curvature.Module.Renderer.alpha import AlphaRenderer
from diff_curvature.Module.Renderer.multi_channel import MultiChannelRenderer


def demo(
    mesh_path: str,
    result_dir: str = "./output/",
    num_viewpoints: int = 6,
    image_size: int = 256,
    device: str = "cuda:0",
):
    warnings.filterwarnings("ignore")

    ## python render.py --mesh_path data_example/Kar.obj --result_dir results/ --num_viewpoints 6 --image_size 256 --device cuda:0
    """
    Ground truth mesh
    """

    DEVICE = torch.device(device)

    mesh_name = os.path.basename(mesh_path).split("/")[-1]
    mesh_name = mesh_name.split(".")[0]

    image_save_path = os.path.join(result_dir, mesh_name)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)

    mesh_tem = load_objs_as_meshes([mesh_path], device=DEVICE)
    mesh_tem = normalize_mesh(mesh_tem, 0.88)

    verts, faces = mesh_tem.verts_list()[0], mesh_tem.faces_list()[0]

    print("===== Ground truth mesh =====")
    print("Id: ", mesh_name)
    print("Number of vertices: ", verts.shape[0])
    print("Number of faces: ", faces.shape[0])
    print("=============================")

    # save gt mesh;
    mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
    mesh.export(os.path.join(image_save_path, "gt_mesh.obj"))

    """
    Multi-channel renderer
    """

    mv, proj = make_star_cameras(
        num_viewpoints, num_viewpoints, distance=2.0, r=0.6, n=1.0, f=3.0
    )
    proj = proj.unsqueeze(0).expand(mv.shape[0], -1, -1)
    renderer = AlphaRenderer(mv, proj, [image_size, image_size])

    gt_manager = MultiChannelRenderer(verts, faces, DEVICE)
    gt_manager.render(renderer)

    gt_diffuse_map = gt_manager.diffuse_images()
    gt_depth_map = gt_manager.depth_images()
    gt_shil_map = gt_manager.shillouette_images()
    gt_normals_map = gt_manager.normal_images()
    gt_gs_curv_map = gt_manager.gs_curvatures_images()
    gt_mean_curv_map = gt_manager.mean_curvature_images()
    gt_curv_rgb_map = gt_manager.curvatures_rgb_images()

    for i in range(len(gt_diffuse_map)):
        save_image(
            gt_diffuse_map[i], os.path.join(image_save_path, "diffuse_{}.png".format(i))
        )
        save_image(
            gt_depth_map[i], os.path.join(image_save_path, "depth_{}.png".format(i))
        )
        save_image(
            gt_shil_map[i], os.path.join(image_save_path, "shil_{}.png".format(i))
        )
        save_image(
            gt_normals_map[i], os.path.join(image_save_path, "normals_{}.png".format(i))
        )
        save_image(
            gt_gs_curv_map[i], os.path.join(image_save_path, "gs_curv_{}.png".format(i))
        )
        save_image(
            gt_mean_curv_map[i],
            os.path.join(image_save_path, "mean_curv_{}.png".format(i)),
        )
        save_image(
            gt_curv_rgb_map[i],
            os.path.join(image_save_path, "curv_rgb_{}.png".format(i)),
        )

    print("Saved images to: ", image_save_path)
    print("======== Done! ========")
