import os
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
from nilearn import surface

# 手动定义 mesh_reduce（基于 Open3D）
def mesh_reduce(mesh, n_vertices=1000):
    import open3d as o3d
    coords, faces = mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(coords)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=n_vertices * 2)
    coords = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return coords, faces

# 输入输出路径
input_path = "/root/autodl-tmp/hcp_gcn/template/dhcp_week-40_hemi-right_init.surf.gii"
output_path = "/root/autodl-tmp/hcp_gcn/template/dhcp_week-40_88365_hemi-right_init.surf.gii"
# input_path = "/root/autodl-tmp/hcp_gcn/template/dhcp_week-40_hemi-left_init.surf.gii"
# output_path = "/root/autodl-tmp/hcp_gcn/template/dhcp_week-40_88365_hemi-left_init.surf.gii"
# 加载原始表面
mesh = surface.load_surf_mesh(input_path)
coords, faces = mesh[0], mesh[1]

# 降采样,写88365实际会降成88367,+2
reduced_coords, reduced_faces = mesh_reduce((coords, faces), n_vertices=88365)

# 转换数据类型（关键步骤）
reduced_coords = reduced_coords.astype(np.float32)
reduced_faces = reduced_faces.astype(np.int32)

# 保存为 GIFTI
gii_img = GiftiImage(darrays=[
    GiftiDataArray(
        data=reduced_coords.astype(np.float32),
        intent='NIFTI_INTENT_POINTSET',
        meta={'AnatomicalStructurePrimary': 'CortexLeft'}
    ),
    GiftiDataArray(
        data=reduced_faces.astype(np.int32),
        intent='NIFTI_INTENT_TRIANGLE'
    )
])
nib.save(gii_img, output_path)

print(f"✅ 降采样完成，文件保存到：{output_path}")
