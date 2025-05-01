import nibabel as nib

try:
    gii = nib.load("/root/autodl-tmp/hcp_gcn/template/dhcp_week-40_88365_hemi-left_init.surf.gii")
    print("✅ 文件可以正常加载，顶点数：", gii.darrays[0].data.shape[0])
except Exception as e:
    print("❌ 文件损坏：", e)
