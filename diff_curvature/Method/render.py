import torch
import nvdiffrast.torch as dr


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
