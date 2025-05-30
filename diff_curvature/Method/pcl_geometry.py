import torch
from pytorch3d.ops import knn_points, knn_gather


def disambiguate(normals, pointscloud, K=15):
    normals_field = 1.0 * normals  # flip the normals to make them point outward

    current_set = set([0])

    finished_set = set()

    knn_info = knn_points(pointscloud, pointscloud, K=K)

    while len(current_set) > 0:
        renew_index = knn_info.idx[:, list(current_set), :]

        current_normals = knn_gather(normals_field, renew_index)

        direction_consistency = (current_normals * current_normals[:, :, 0:1, :]).sum(
            -1, keepdim=True
        )

        direction_consistency = torch.where(direction_consistency > 0, 1, -1)

        renew_normals = current_normals * direction_consistency

        for i in range(renew_index.shape[0]):
            for j in range(renew_index.shape[1]):
                normals_field[i, (renew_index[i, j, :]).view(-1), :] = renew_normals[
                    i, j, :, :
                ]

        finished_set = finished_set.union(current_set)

        current_set = set(renew_index[:, :, 1:].reshape(-1).tolist())

        current_set = current_set - finished_set

    return normals_field
