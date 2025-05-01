import os
import torch
import numpy as np
import trimesh
from glob import glob
from tqdm import tqdm
import csv

### ===============================
### 1. Loss Functions
### ===============================

def chamfer_distance(verts1, verts2):
    d1 = torch.cdist(verts1, verts2).min(dim=2)[0]
    d2 = torch.cdist(verts2, verts1).min(dim=2)[0]
    return (d1.mean() + d2.mean()) / 2

def edge_length_loss(verts, faces):
    v0 = verts[:, faces[:, 0], :]
    v1 = verts[:, faces[:, 1], :]
    v2 = verts[:, faces[:, 2], :]
    e1 = torch.norm(v0 - v1, dim=2)
    e2 = torch.norm(v1 - v2, dim=2)
    e3 = torch.norm(v2 - v0, dim=2)
    edge_lens = torch.cat([e1, e2, e3], dim=1)
    mean = edge_lens.mean()
    return torch.mean((edge_lens - mean) ** 2)

def face_normals(verts, faces):
    v0 = verts[:, faces[:, 0], :]
    v1 = verts[:, faces[:, 1], :]
    v2 = verts[:, faces[:, 2], :]
    normals = torch.cross(v1 - v0, v2 - v0, dim=2)
    return torch.nn.functional.normalize(normals, dim=2)

def normal_consistency_loss(verts, faces):
    normals = face_normals(verts, faces)
    dot = (normals[:, :-1, :] * normals[:, 1:, :]).sum(dim=2)
    return 1.0 - dot.mean()

def area_of_triangles(verts, faces):
    v0 = verts[:, faces[:, 0], :]
    v1 = verts[:, faces[:, 1], :]
    v2 = verts[:, faces[:, 2], :]
    return 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=2), dim=2)

def area_distortion_loss(verts_pred, faces_pred, verts_gt, faces_gt):
    area_pred = area_of_triangles(verts_pred, faces_pred)
    area_gt = area_of_triangles(verts_gt, faces_gt)
    return torch.mean((area_pred - area_gt) ** 2 / (area_gt + 1e-8))

def edge_distortion_loss(verts_pred, faces_pred, verts_gt, faces_gt):
    def get_edges(v, f):
        v0 = v[:, f[:, 0], :]
        v1 = v[:, f[:, 1], :]
        v2 = v[:, f[:, 2], :]
        e1 = torch.norm(v0 - v1, dim=2)
        e2 = torch.norm(v1 - v2, dim=2)
        e3 = torch.norm(v2 - v0, dim=2)
        return torch.cat([e1, e2, e3], dim=1)

    e_pred = get_edges(verts_pred, faces_pred)
    e_gt = get_edges(verts_gt, faces_gt)
    return torch.mean((e_pred - e_gt) ** 2 / (e_gt + 1e-8))

### ===============================
### 2. IO Functions
### ===============================

import nibabel as nib

def load_surf_gii(filepath):
    gii = nib.load(filepath)
    vertices = torch.tensor(gii.darrays[0].data, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
    faces = torch.tensor(gii.darrays[1].data, dtype=torch.long)  # [F, 3]
    return vertices, faces

### ===============================
### 3. Comparison
### ===============================

def compare_surfaces(gt_path, pred_path):
    vert_gt, face_gt = load_surf_gii(gt_path)
    vert_pred, face_pred = load_surf_gii(pred_path)

    losses = {
        'chamfer': chamfer_distance(vert_pred, vert_gt),
        'edge_length': edge_length_loss(vert_pred, face_pred),
        'normal_consistency': normal_consistency_loss(vert_pred, face_pred),
        'area_distortion': area_distortion_loss(vert_pred, face_pred, vert_gt, face_gt),
        'edge_distortion': edge_distortion_loss(vert_pred, face_pred, vert_gt, face_gt),
    }

    return {k: v.item() for k, v in losses.items()}

### ===============================
### 4. Main Evaluation Loop
### ===============================

def main():
    base_dir = '/root/autodl-tmp/hcp_gcn/datasets_for_surface_resconstruction_testing'
    output_file = 'evaluation_results.csv'
    all_results = []

    subjects = sorted(glob(os.path.join(base_dir, 'sub-*')))
    for subj_dir in tqdm(subjects):
        subj_id = os.path.basename(subj_dir)
        gt_path = os.path.join(subj_dir, f"{subj_id}_hemi-left_wm.surf.gii")
        pred_paths = sorted(glob(os.path.join(subj_dir, f"{subj_id}_hemi-left_wm_*cd-nc-edge-50.surf.gii")))

        for pred_path in pred_paths:
            try:
                epoch = os.path.basename(pred_path).split('_')[3].split('-')[0]
                losses = compare_surfaces(gt_path, pred_path)
                all_results.append({
                    "subject": subj_id,
                    "epoch": epoch,
                    **losses
                })
            except Exception as e:
                print(f"[!] Failed to process {pred_path}: {e}")

    # 保存结果
    if all_results:
        with open(output_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)

        print(f"✅ Evaluation done. Results saved to {output_file}")
    else:
        print("⚠️ No results to write!")

if __name__ == "__main__":
    main()
