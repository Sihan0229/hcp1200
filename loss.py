import torch
import numpy as np
from utils.mesh import face_area


def distortion(metric_pred_surf, metric_surf):
    """
    Compute metric distortion. (area or edge)

    Inputs:
    - metric_pred_surfe: the metric of the (prediced) surphere, (1,|V|) torch.Tensor
    - metric_surf: the metric of the reference WM surface, (1,|V|) torch.Tensor
    Returns:
    - distort: the metric distortion (RMSD), torch.float
    """
    # print(f"[DEBUG] metric_pred_surf.shape: {metric_pred_surf.shape}")
    # print(f"[DEBUG] metric_surf.shape: {metric_surf.shape}")
    # beta = (metric_sphere * metric_surf).mean() /  (metric_surf**2).mean()
    # distort = ((metric_sphere - beta*metric_surf)**2).mean()
    beta = (metric_pred_surf * metric_surf).mean() /  (metric_pred_surf**2).mean()
    distort = ((beta*metric_pred_surf - metric_surf)**2).mean().sqrt()
    return distort


def edge_distortion(pred_vert, vert_surf, edge):
    """
    Compute edge distortion.

    Inputs:
    - pred_vert: the vertices of the (prediced) surface, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - edge: the edge list of the mesh, (2,|E|) torch.LongTensor
    
    Returns:
    - edge distortion, torch.float
    """
    # compute edge length
    edge_len_pred = (pred_vert[:,edge[0]] -\
                       pred_vert[:,edge[1]]).norm(dim=-1)
    edge_len_surf = (vert_surf[:,edge[0]] -\
                     vert_surf[:,edge[1]]).norm(dim=-1)

    return distortion(edge_len_pred, edge_len_surf)


def area_distortion(pred_vert, vert_surf, face):
    """
    Compute area distortion.

    Inputs:
    - pred_vert: the vertices of the (prediced) surface, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - face: the mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area distortion, torch.float
    """
    
    # compute face area
    area_pred = face_area(pred_vert, face)
    area_surf = face_area(vert_surf, face)
    return distortion(area_pred, area_surf)

# def chamfer_distance(pc1, pc2):
#     """
#     Compute Chamfer Distance between two point clouds.

#     Inputs:
#     - pc1: (N, 3) torch.Tensor, predicted points
#     - pc2: (M, 3) torch.Tensor, reference points
    
#     Returns:
#     - chamfer_dist: Chamfer distance, torch.float
#     """
#     dist_matrix = torch.cdist(pc1, pc2)  # 计算点对之间的欧式距离矩阵
#     min_dist1 = dist_matrix.min(dim=1)[0]  # pc1 到 pc2 的最近点距离
#     min_dist2 = dist_matrix.min(dim=0)[0]  # pc2 到 pc1 的最近点距离
#     return min_dist1.mean() + min_dist2.mean()


# def chamfer_distance_batched(pc1, pc2, batch_size=4096):
#     """ 
#     Compute Chamfer Distance between two point clouds in batches to save memory. 
#     """
#     N, M = pc1.shape[0], pc2.shape[0]
#     min_dist1 = torch.full((N,), float('inf'), device=pc1.device)
#     min_dist2 = torch.full((M,), float('inf'), device=pc2.device)

#     for i in range(0, N, batch_size):
#         dist_batch = torch.cdist(pc1[i:i+batch_size], pc2)  # 计算小批量的距离
#         min_dist1[i:i+batch_size] = dist_batch.min(dim=1)[0]

#     for j in range(0, M, batch_size):
#         dist_batch = torch.cdist(pc2[j:j+batch_size], pc1)  # 计算小批量的距离
#         min_dist2[j:j+batch_size] = dist_batch.min(dim=1)[0]

#     return min_dist1.mean() + min_dist2.mean()



# def hausdorff_distance(pc1, pc2):
#     """
#     Compute Hausdorff Distance between two point clouds.

#     Inputs:
#     - pc1: (N, 3) torch.Tensor, predicted points
#     - pc2: (M, 3) torch.Tensor, reference points
    
#     Returns:
#     - hausdorff_dist: Hausdorff distance, torch.float
#     """
#     dist_matrix = torch.cdist(pc1, pc2)
#     min_dist1 = dist_matrix.min(dim=1)[0]  # pc1 到 pc2 的最近点距离
#     min_dist2 = dist_matrix.min(dim=0)[0]  # pc2 到 pc1 的最近点距离
#     return torch.max(torch.max(min_dist1), torch.max(min_dist2))

# def combine_edge_distortion(vert_sphere, vert_surf, edge, alpha=1.0, beta=0.5, gamma=0.1):
#     """
#     Compute edge distortion with distortion loss, Chamfer loss, and Hausdorff loss.

#     Inputs:
#     - vert_sphere: (1, |V|, 3) torch.Tensor, predicted sphere vertices
#     - vert_surf: (1, |V|, 3) torch.Tensor, reference WM surface vertices
#     - edge: (2, |E|) torch.LongTensor, edge list of the mesh
#     - alpha, beta, gamma: weights for combining the loss terms
    
#     Returns:
#     - combined edge distortion loss, torch.float
#     """
#     # Compute edge length
#     edge_len_sphere = (vert_sphere[:, edge[0]] - vert_sphere[:, edge[1]]).norm(dim=-1)
#     edge_len_surf = (vert_surf[:, edge[0]] - vert_surf[:, edge[1]]).norm(dim=-1)

#     # Compute individual losses
#     dist_loss = distortion(edge_len_sphere, edge_len_surf)  # 原始失真损失
#     chamfer_loss = chamfer_distance_batched(vert_sphere.squeeze(0), vert_surf.squeeze(0), batch_size=4096)
#     hausdorff_loss = hausdorff_distance(vert_sphere.squeeze(0), vert_surf.squeeze(0))  # Hausdorff 距离

#     # Combine losses
#     total_loss = alpha * dist_loss + beta * chamfer_loss + gamma * hausdorff_loss
#     return total_loss

# def combine_area_distortion(vert_sphere, vert_surf, face, alpha=1.0, beta=0.5, gamma=0.1):
#     """
#     Compute area distortion with distortion loss, Chamfer loss, and Hausdorff loss.

#     Inputs:
#     - vert_sphere: (1, |V|, 3) torch.Tensor, predicted sphere vertices
#     - vert_surf: (1, |V|, 3) torch.Tensor, reference WM surface vertices
#     - face: (1, |F|, 3) torch.LongTensor, mesh faces
#     - alpha, beta, gamma: weights for combining the loss terms

#     Returns:
#     - combined area distortion loss, torch.float
#     """
#     # Compute face areas
#     area_sphere = face_area(vert_sphere, face)
#     area_surf = face_area(vert_surf, face)

#     # Compute individual losses
#     dist_loss = distortion(area_sphere, area_surf)  # 原始失真损失
#     chamfer_loss = chamfer_distance_batched(vert_sphere.squeeze(0), vert_surf.squeeze(0), batch_size=4096)
#     hausdorff_loss = hausdorff_distance(vert_sphere.squeeze(0), vert_surf.squeeze(0))  # Hausdorff 距离

#     # Combine losses
#     total_loss = alpha * dist_loss + beta * chamfer_loss + gamma * hausdorff_loss
#     return total_loss''
