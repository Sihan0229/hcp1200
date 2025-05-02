import nibabel as nib
# surf = nib.load("/root/autodl-tmp/hcp1200/template_hcp1200/lh.pial_150k.surf.gii")
surf = nib.load("/root/autodl-tmp/hcp1200_dataset/HCP1200_cut/100408/100408.R.white.native.surf.gii")
vertices = surf.darrays[0].data
print("Template vertices:", vertices.shape[0])