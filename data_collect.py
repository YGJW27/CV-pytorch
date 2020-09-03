import os
import shutil
import glob


# # DTI data copy
# source_path = "D:/study/data/PANDA_data/DTI_T1/"
# dti_path = "D:/study/data/PANDA_data/DTI/"

# sub_dirs = [x[0] for x in os.walk(source_path)]
# sub_dirs.pop(0)

# for sub_dir in sub_dirs:
#     indiv = os.path.basename(sub_dir)
#     subsub_dirs = [x[0] for x in os.walk(sub_dir)]
#     subsub_dirs.pop(0)
#     for subsub_dir in subsub_dirs:
#         subsub_folder = os.path.basename(subsub_dir)
#         if "DTI" in subsub_folder:
#             file_names = os.listdir(subsub_dir)
#             output_dir = os.path.join(dti_path, indiv, 'DTI1')
#             for file_name in file_names:
#                 if (".nii.gz" in file_name) or ("bvals" in file_name) or ("bvecs" in file_name):
#                     if not os.path.exists(output_dir):
#                         os.makedirs(output_dir)
#                     src = os.path.join(subsub_dir, file_name)
#                     dst = os.path.join(output_dir, file_name)
#                     shutil.copyfile(src, dst)


# T1 data copy
source_path = "D:/study/data/PANDA_data/DTI_T1/"
t1_path = "D:/study/data/PANDA_data/T1/"

sub_dirs = [x[0] for x in os.walk(source_path)]
sub_dirs.pop(0)

for sub_dir in sub_dirs:
    indiv = os.path.basename(sub_dir)
    subsub_dirs = [x[0] for x in os.walk(sub_dir)]
    subsub_dirs.pop(0)
    for subsub_dir in subsub_dirs:
        subsub_folder = os.path.basename(subsub_dir)
        if "t1" in subsub_folder:
            file_names = os.listdir(subsub_dir)
            for file_name in file_names:
                if ".nii" in file_name:
                    src = os.path.join(subsub_dir, file_name)
                    dst = os.path.join(t1_path, indiv + ".nii")
                    shutil.copyfile(src, dst)