python3 train_surface_reconstruction_pial.py --in_dir='/root/autodl-tmp/hcp_gcn/datasets_for_surface_resconstruction_training/' \
                       --out_dir='/root/autodl-tmp/hcp_gcn/model_trained_by_50/' \
                       --restore_suffix='_desc_T2w_restore_brain.nii.gz' \
                       --hemi='right' \
                       --surface='pial' \
                       --device='cuda:0'