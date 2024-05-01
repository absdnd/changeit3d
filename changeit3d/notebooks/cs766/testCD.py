import numpy as np
from scipy.spatial import cKDTree

def load_point_cloud(filename):
    """Load point cloud from a .npz file."""
    data = np.load(filename)
    # Use the correct key from your .npz file
    return data['pointcloud']

def closest_point_distance(points, other_points):
    tree = cKDTree(other_points)
    dists, _ = tree.query(points, k=1)
    return np.mean(dists)

def chamfer_distance(pc1, pc2):
    d1 = closest_point_distance(pc1, pc2)
    d2 = closest_point_distance(pc2, pc1)
    return (d1 + d2) / 2


# Example usage
pc1 = load_point_cloud('/home/shared/changeit3d/changeit3d/data/shapetalk/point_clouds/scaled_to_align_rendering/bottle/ShapeNet/f4851a2835228377e101b7546e3ee8a7.npz')
pc2 = load_point_cloud('/home/shared/changeit3d/changeit3d/data/shapetalk/point_clouds/scaled_to_align_rendering/bottle/ShapeNet/2db802ef3de3d00618a36258eabc2b9c.npz')
distance = chamfer_distance(pc1, pc2)
print(f"Chamfer Distance: {distance}")
