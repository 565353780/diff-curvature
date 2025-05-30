import torch
import torch.nn as nn


class SolidAngle(nn.Module):
    def __init__(self):
        """
        Compute the solid angle of a batch of triangles
        Input: batch_of_three_vectors: [B, 3, 3]
        Output: solid_angle: [B,]
        """
        super(SolidAngle, self).__init__()

    def forward(self, batch_of_three_vectors):
        assert batch_of_three_vectors.shape[-1] == 3
        assert batch_of_three_vectors.shape[-2] == 3

        a_vert = batch_of_three_vectors[..., 0, :]  # [B, 3]
        b_vert = batch_of_three_vectors[..., 1, :]  # [B, 3]
        c_vert = batch_of_three_vectors[..., 2, :]  # [B, 3]

        face_det = (a_vert * b_vert.cross(c_vert)).sum(dim=-1)  # [B,]

        abc = batch_of_three_vectors.norm(dim=-1).prod(dim=-1)  # [B,3]-->[B,]

        ab = (a_vert * b_vert).sum(-1)  # [B,]
        bc = (b_vert * c_vert).sum(-1)  # [B,]
        ac = (a_vert * c_vert).sum(-1)  # [B,]

        solid_angle = 2 * torch.arctan2(
            face_det,
            (
                abc
                + bc * a_vert.norm(dim=-1)
                + ac * b_vert.norm(dim=-1)
                + ab * c_vert.norm(dim=-1)
            ),
        )  # []

        return solid_angle
