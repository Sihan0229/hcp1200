python3 run_pipeline.py --in_dir='/root/autodl-tmp/hcp_gcn/datasets_dhcp_cut2/' \
                       --out_dir='/root/autodl-tmp/hcp_gcn/datasets_for_surface_resconstruction_testing/' \
                       --T2='_T2w.nii.gz' \
                       --T1='_T1w.nii.gz' \
                       --sphere_proj='fs' \
                       --device='cuda:0'