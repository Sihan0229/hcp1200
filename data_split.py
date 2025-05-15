import os

def save_two_level_structure_to_txt(root_dir, save_path):
    with open(save_path, 'w') as f:
        # 遍历第一层
        for first_level_name in sorted(os.listdir(root_dir)):
            first_level_path = os.path.join(root_dir, first_level_name)
            if os.path.isdir(first_level_path):
                f.write(f"{first_level_name}/\n")
                # 遍历第二层
                second_level_names = sorted(os.listdir(first_level_path))
                for second_level_name in second_level_names:
                    second_level_path = os.path.join(first_level_path, second_level_name)
                    if os.path.isdir(second_level_path):
                        f.write(f"  {second_level_name}/\n")

if __name__ == "__main__":
    root = "/root/autodl-tmp/hcp1200_dataset/HCP1200_split/train_valid"
    save_file = "split_train_valid.txt"
    save_two_level_structure_to_txt(root, save_file)
