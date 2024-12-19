import open3d as o3d
import numpy as np
import os
import argparse
import sys
import torch

def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

def build_kdtree(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    return kdtree

def compute_nn_distances(src_points, dst_points):
    """
    Compute nearest neighbor distances from each point in src_points to the closest point in dst_points.
    Returns:
        distances: numpy array of shape (len(src_points),) containing the squared distances.
    """
    # Build KD-tree for dst_points
    dst_kdtree = build_kdtree(dst_points)
    distances = np.zeros(len(src_points), dtype=np.float32)
    for i, p in enumerate(src_points):
        [_, idx, dist] = dst_kdtree.search_knn_vector_3d(p, 1)
        distances[i] = dist[0]  # dist is already squared Euclidean distance in Open3D
    return distances

def compute_chamfer_distance(xyz1, xyz2, device="cpu"):
    """
    Compute Chamfer Distance via KD-tree nearest neighbors.
    Chamfer = mean(min_dist(xyz1->xyz2)) + mean(min_dist(xyz2->xyz1))
    """
    dist_12 = compute_nn_distances(xyz1, xyz2)
    dist_21 = compute_nn_distances(xyz2, xyz1)
    chamfer = dist_12.mean() + dist_21.mean()
    return chamfer

def compute_precision_completeness(gt_points, pred_points, threshold=0.05, device="cpu"):
    """
    Compute precision and completeness using KD-tree nearest neighbors.
    Completeness: fraction of GT points within threshold of a pred point
    Precision: fraction of Pred points within threshold of a GT point
    """
    dist_gt_pred = np.sqrt(compute_nn_distances(gt_points, pred_points)) # sqrt because NN dist are squared
    dist_pred_gt = np.sqrt(compute_nn_distances(pred_points, gt_points))

    completeness = np.mean(dist_gt_pred < threshold)
    precision = np.mean(dist_pred_gt < threshold)
    return precision, completeness

def compute_voxel_iou(gt_points, pred_points, voxel_size=0.1):
    all_points = np.vstack((gt_points, pred_points))
    min_xyz = all_points.min(axis=0)

    def points_to_voxels(points, min_xyz, voxel_size):
        indices = ((points - min_xyz) / voxel_size).astype(np.int32)
        return indices

    gt_vox = points_to_voxels(gt_points, min_xyz, voxel_size)
    pred_vox = points_to_voxels(pred_points, min_xyz, voxel_size)

    gt_set = set(map(tuple, gt_vox))
    pred_set = set(map(tuple, pred_vox))

    intersection = gt_set.intersection(pred_set)
    union = gt_set.union(pred_set)

    if len(union) == 0:
        iou = 1.0 if len(intersection) == 0 else 0.0
    else:
        iou = len(intersection) / float(len(union))

    return iou

def downsample(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down_pcd.points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Reconstruction Metrics")
    parser.add_argument("--gt", type=str, required=False, default="/vast/yw4142/checkpoints/spann3r/checkpoints/output/demo/brown_cogsci_7/time.ply", help="Path to ground truth PLY")
    parser.add_argument("--pred", type=str, required=False, default="/vast/yw4142/checkpoints/spann3r/checkpoints/output/demo/brown_cogsci_7/brown_cogsci_7_conf0.001_weighted.ply", help="Path to predicted PLY")
    parser.add_argument("--out", type=str, required=False, default="/vast/yw4142/checkpoints/spann3r/checkpoints", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.05, help="Distance threshold for precision/completeness")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size for IoU computation")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.gt):
        print("Ground truth file not found:", args.gt)
        sys.exit(1)

    if not os.path.exists(args.pred):
        print("Predicted file not found:", args.pred)
        sys.exit(1)

    gt_points = load_point_cloud(args.gt)
    pred_points = load_point_cloud(args.pred)
    gt_points = downsample(gt_points, voxel_size=0.05)
    pred_points = downsample(pred_points, voxel_size=0.05)

    # Compute metrics
    chamfer = compute_chamfer_distance(gt_points, pred_points, device)
    precision, completeness = compute_precision_completeness(gt_points, pred_points, threshold=args.threshold, device=device)
    iou = compute_voxel_iou(gt_points, pred_points, voxel_size=args.voxel_size)

    print("Chamfer Distance:", chamfer)
    print(f"Precision (@{args.threshold}m):", precision)
    print(f"Completeness (@{args.threshold}m):", completeness)
    print("Voxel IoU:", iou)

    out_file = os.path.join(args.out, "evaluation_results.txt")
    with open(out_file, "w") as f:
        f.write(f"Chamfer Distance: {chamfer}\n")
        f.write(f"Precision (@{args.threshold}m): {precision}\n")
        f.write(f"Completeness (@{args.threshold}m): {completeness}\n")
        f.write(f"Voxel IoU: {iou}\n")

    print("Evaluation results saved to:", out_file)

