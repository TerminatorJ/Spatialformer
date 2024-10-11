import os
import re
from collections import defaultdict
import numpy as np
import math
# Initialize the output dictionary with defaultdict to automatically create sets
# output = defaultdict(lambda: {"cell_num": set(), "tt": set(), "mep": set(), "mdp": set(), "mt": set(), "pt": set(), "max_pt":[]})
# root_dir = "/home/sxr280/Spatialformer/scripts/"
# pattern = r'^find_gene_interaction_\d{4}\.out$'
# faild_command = []
# for filename in os.listdir(root_dir):
#     # Check if the file matches the desired pattern
#     if re.match(pattern, filename):
#         with open(os.path.join(root_dir, filename)) as f1:
#             for line in f1.readlines():
#                 if "Running command:" in line and "python" in line:
#                     # import pdb; pdb.set_trace()
#                     # try:
#                     str_id = line.split("/")[-2]
#                     # except:
#                     #     print(line)
#                     max_pt = line.split("--partition ")[1].split(" ")[0]
#                     command = line.split("Running command: ")[1].strip()

#                 if "The number of cells that are kept" in line:
                    
#                     cell_num = line.split(": ")[1].strip()
#                     output[str_id]["cell_num"].add(int(cell_num))
#                     output[str_id]["max_pt"].append(int(max_pt))
#                 if "Total transcripts left" in line:
#                     # import pdb; pdb.set_trace()
#                     tt = line.split(": ")[1].strip()
#                     output[str_id]["tt"].add(int(tt))
#                 if "mean number of the pairs is:" in line:
#                     mep = line.split(": ")[1].strip()
#                     output[str_id]["mep"].add(float(mep))
#                     # import pdb; pdb.set_trace()
#                     # if math.isnan(float(mep)):
#                     #     faild_command.append(command)
#                 if "median number of the pairs is:" in line:
#                     mdp = line.split(": ")[1].strip()
#                     output[str_id]["mdp"].add(float(mdp))
#                 if "Mean transcripts per cell:" in line:
#                     mt = line.split(": ")[1].strip()
#                     output[str_id]["mt"].add(float(mt))

#                 if "total partitions you need are:" in line:
#                     pt = line.split(": ")[1].strip()
#                     output[str_id]["pt"].add(int(pt))

# #getting the arguments that are faild
# # print("faild command:")
# # print(faild_command)



                
# # Function to calculate mean of a set
# def mean(values_set):
#     clean_values = {v for v in values_set if not math.isnan(v)}  # Remove NaN values
#     if len(values_set) == 0:
#         return 0
#     return sum(clean_values) / len(clean_values)

# # Calculate mean values and print


# str_ids = ["Xenium_V1_hLung_cancer_section_outs",
# "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs",
# "Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs",
# "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",
# "Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs",
# "Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs",
# "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs",
# "Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs",
# "Xenium_V1_human_Pancreas_FFPE_outs",
# "Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs",
# "Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs",
# "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs",
# "Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs",
# "Xenium_V1_hLiver_nondiseased_section_FFPE_outs",
# "Xenium_V1_hLiver_cancer_section_FFPE_outs",
# "Xenium_V1_hHeart_nondiseased_section_FFPE_outs",
# "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs",
# "Xenium_V1_hSkin_Melanoma_Base_FFPE_outs",
# "Xenium_V1_hColon_Non_diseased_Base_FFPE_outs",
# "Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs",
# "Xenium_V1_hColon_Cancer_Base_FFPE_outs",
# "Xenium_V1_hColon_Cancer_Add_on_FFPE_outs",
# "Xenium_V1_hLung_cancer_section_outs",
# "Xenium_V1_hLymphNode_nondiseased_section_outs",
# "Xenium_V1_humanLung_Cancer_FFPE_outs",
# "Xenium_Prime_Human_Ovary_FF_outs",
# "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
# "Xenium_Prime_Cervical_Cancer_FFPE_outs",
# "Xenium_Prime_Human_Skin_FFPE_outs",
# "Xenium_Prime_Human_Prostate_FFPE_outs",
# "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
# "Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs",
# "Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs",
# "Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs",
# "Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs",
# "Xenium_V1_Human_Brain_GBM_FFPE_outs",
# "Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs",
# "Xenium_V1_hBoneMarrow_nondiseased_section_outs",
# "Xenium_V1_hBone_nondiseased_section_outs",
# "Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs",
# "Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs",
# "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs",
# "Xenium_V1_FFPE_Human_Breast_ILC_With_Addon_outs",
# "Xenium_V1_FFPE_Human_Breast_IDC_outs",
# "Xenium_V1_FFPE_Human_Breast_ILC_outs",
# "Xenium_V1_hPancreas_nondiseased_section_outs",
# "Xenium_V1_hKidney_nondiseased_section_outs",
# "Xenium_V1_hKidney_cancer_section_outs"]

# output_df = {"Str_ID": [],"Cell_left":[],"Mean pairs":[],"Median pairs":[],"Mean transcripts":[],"Total transcripts":[], "progress":[], "partition":[], "Partition_left":[]}

# print("Mean Values:")
# # for str_id, data in output.items():
# for str_id in str_ids:
#     data = output[str_id]
#     print(data)
#     if len(data["cell_num"]) > 0:
#         mean_values = {key: mean(values) for key, values in data.items()}
#         # counter = {key: values for key, values in data.items() if key in ["mep","pt"]}
#         cleaned_data = {key: {v for v in values if not math.isnan(v)} for key, values in data.items()}
#         print(f"{str_id}: {mean_values}")  
#         print(len(cleaned_data["pt"]))
#         # cleaned_data["mep"]
#         print(data)
#         pt = cleaned_data["pt"].pop()
#         keep_index = np.where(~np.isnan(list(data["mep"])))[0]
#         print("keep_index:", keep_index)
#         keep_pt = len(np.unique(np.array(data["max_pt"])[keep_index]))
#         # max_pt = np.max(list(cleaned_data["max_pt"])[:len(cleaned_data["mep"])])
#         pt_left = [i+1 for i in range(pt) if i+1 not in np.unique(np.array(data["max_pt"])[keep_index])]
#         print("pt_left:",pt_left)
#         mep = len(cleaned_data["mep"])
#         # print(f"{str_id}: has {pt} partitions, and complete {mep}, progress {mep/pt*100}%")  
#         output_df["Str_ID"].append(str_id)  
#         output_df["Cell_left"].append(mean_values["cell_num"])  
#         output_df["Mean pairs"].append(mean_values["mep"])  
#         output_df["Median pairs"].append(mean_values["mdp"])
#         output_df["Mean transcripts"].append(mean_values["mt"])
#         output_df["Total transcripts"].append(mean_values["tt"])
#         output_df["progress"].append(f"{keep_pt/pt*100}%")
#         output_df["partition"].append(pt)
#         output_df["Partition_left"].append(str(pt_left))
#     else:
#         output_df["Str_ID"].append(str_id)  
#         output_df["Cell_left"].append("NAN")  
#         output_df["Mean pairs"].append("NAN")  
#         output_df["Median pairs"].append("NAN")
#         output_df["Mean transcripts"].append("NAN")
#         output_df["Total transcripts"].append("NAN")
#         output_df["progress"].append("NAN")
#         output_df["partition"].append(pt)
#         output_df["Partition_left"].append("NAN")
# import pandas as pd
# df = pd.DataFrame(output_df)
# df.to_csv("/home/sxr280/Spatialformer/scripts/find_gene_interaction_stadf.csv")


processed_dir = "/tmp/erda/Spatialformer/downloaded_data/processed/"
import os

def list_small_files(directory, max_size_mb=1):
    # Convert max size from megabytes to bytes
    max_size_bytes = max_size_mb * 1024 * 1024

    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Get the size of the file
            file_size = os.path.getsize(file_path)
            # Check if the file size is less than the specified maximum size
            if file_size < max_size_bytes:
                print(f"{file}: {file_size / 1024:.2f} KB")
                os.remove(file_path)
# Specify the directory you want to search

list_small_files(processed_dir)
