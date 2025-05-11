'''
检查surf.gii文件的顶点数量
'''
import nibabel as nib
# surf = nib.load("/root/autodl-tmp/hcp1200/template_hcp1200/lh.pial_150k.surf.gii")
surf = nib.load("/root/autodl-tmp/hcp1200_dataset/HCP1200_cut/100408/100408.R.white.native.surf.gii")
vertices = surf.darrays[0].data
print("Template vertices:", vertices.shape[0])

# 模板当中
# 不加150k的surf都是163842，和sphere一样的尺寸
# 加了150k的大小不一，已经删掉了

# HCP当中，各个都不太一样大，需要remesh