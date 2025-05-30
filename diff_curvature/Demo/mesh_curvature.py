from diff_curvature.Module.mesh_curvature import MeshCurvature


def demo():
    mesh_file_path = "/home/chli/chLi/Dataset/vae-eval/mesh/000.obj"

    mesh_curvature = MeshCurvature(mesh_file_path)

    if not mesh_curvature.isValid():
        print("load mesh failed!")
        return False

    gauss_vc = mesh_curvature.toGaussV()
    gauss_fvc = mesh_curvature.toGaussFV()
    gauss_fc = mesh_curvature.toGaussF()

    mean_vc = mesh_curvature.toMeanV()
    mean_fc = mesh_curvature.toMeanF()

    total_vc = mesh_curvature.toTotalV()
    total_fc = mesh_curvature.toTotalF()

    print("==== Gauss ====")
    print("V:", gauss_vc.min(), gauss_vc.mean(), gauss_vc.max())
    print("FV:", gauss_fvc.min(), gauss_fvc.mean(), gauss_fvc.max())
    print("F:", gauss_fc.min(), gauss_fc.mean(), gauss_fc.max())

    print("==== Mean ====")
    print("V:", mean_vc.min(), mean_vc.mean(), mean_vc.max())
    print("F:", mean_fc.min(), mean_fc.mean(), mean_fc.max())

    print("==== Total ====")
    print("V:", total_vc.min(), total_vc.mean(), total_vc.max())
    print("F:", total_fc.min(), total_fc.mean(), total_fc.max())

    return True
