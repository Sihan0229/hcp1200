python3 train_pipeline.py --in_dir='/root/autodl-tmp/hcp_gcn/datasets_train/' \
                       --out_dir='/root/autodl-tmp/hcp_gcn/result_train_model/' \
                       --T2='_T2w.nii.gz' \
                       --T1='_T1w.nii.gz' \
                       --sphere_proj='fs' \
                       --device='cuda:0'