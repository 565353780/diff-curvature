from pytorch3d.structures import Meshes
from pytorch3d.ops import packed_to_padded


def vert_feature_packed_padded(Surfaces: Meshes, feature):
    """
    Compute the feature of each vertices in a mesh
    Args:
        Surfaces: Meshes object
        feature: Tensor of shape (V, d) or (V,) where V is the number of vertices and d is the feature dimension
    Returns:
        vert_feature: Tensor of shape (N,1) where N is the number of vertices
        the feature of a vertices is defined as the sum of the feature of the triangles that contains this vertices divided by the dual area of this vertices
    """

    vert_first_idx = Surfaces.mesh_to_verts_packed_first_idx()
    vert_feature_padded = packed_to_padded(
        feature, vert_first_idx, max_size=Surfaces.verts_padded().shape[1]
    )
    return vert_feature_padded


def face_feature_packed_padded(Surfaces: Meshes, feature):
    """
    From the packed face feature to the padded face feature
    Args:
        Surfaces: Meshes object
        feature: Tensor of shape (F, d) or (F,) where F is the number of faces and d is the feature dimension
    Returns:
        face_feature: B x F x d
    """

    face_first_idx = Surfaces.mesh_to_faces_packed_first_idx()
    face_feature_padded = packed_to_padded(
        feature, face_first_idx, max_size=Surfaces.faces_padded().shape[1]
    )
    return face_feature_padded
