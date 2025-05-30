from diff_curvature.Demo.render import demo as demo_render

if __name__ == "__main__":
    mesh_file_path = "/home/chli/chLi/Dataset/Objaverse_82K/mesh/000-000/0000ecca9a234cae994be239f6fec552.obj"
    result_dir = "./output/"
    num_viewpoints = 6
    image_size = 256
    device = "cuda:0"

    demo_render(mesh_file_path, result_dir, num_viewpoints, image_size, device)
