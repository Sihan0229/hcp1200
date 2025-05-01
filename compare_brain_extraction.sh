python3 compare_brain_extraction.py --in_dir='/root/autodl-tmp/hcp1200_dataset/HCP1200_cut_test/' \
                       --out_dir='/root/autodl-tmp/hcp1200/result_train_model_test/' \
                       --T2='T2w.nii.gz' \
                       --T1='T1w.nii.gz' \
                       --sphere_proj='fs' \
                       --device='cuda:0'