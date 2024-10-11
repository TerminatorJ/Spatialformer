import pandas as pd
import os
datainfo = pd.read_table("/home/sxr280/Spatialformer/data/data_info.csv")

batch_size = 1
chunk = 20000
all_bash_str = ""
file_names = [
    "Xenium_Prime_Human_Ovary_FF_outs",
    "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
    "Xenium_Prime_Cervical_Cancer_FFPE_outs",
    "Xenium_Prime_Human_Skin_FFPE_outs",
    "Xenium_Prime_Human_Prostate_FFPE_outs",
    "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs"
]


partitions = [8,14,26,2,3,3,29,10,5,61,38,2,2,12,8,2,5,6,12,14,28,28,8,17,7,7,7,7,7,7,7,11,8,9,18,41,7,1,1,42,42,29,18,28,18,6,6,6]


def listdelete_small_files(directory, max_size_mb=1):
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
                print(file)
                # os.remove(file_path)

def get_generated(directory):
    # Traverse the directory
    str_id_p = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            str_id = file.split("_gene")[0]
            partition = file.split("_")[-1].split(".h5")[0]
            match_str = str_id + "_" + partition
            str_id_p.append(match_str)
    return str_id_p

def removefile(directory):
    for root, dirs, files in os.walk(directory):
        
        for file in files:
            file_path = os.path.join(root, file)
            if "sh" in file:
                # import pdb; pdb.set_trace()
                os.remove(file_path)


# Specify the directory you want to search
# import pdb; pdb.set_trace()
processed_dir = "/tmp/erda/Spatialformer/downloaded_data/processed/"
listdelete_small_files(processed_dir)
str_id_p = get_generated(processed_dir)
print(str_id_p)

#remove the previous bash file
removefile("/home/sxr280/Spatialformer/scripts/find_gene_interaction")

counter = 0
batch = 0
for i,j in enumerate(datainfo["Str_ID"]):
    cell_num = datainfo.iloc[i]["Cell"]
    cell_num = int(cell_num.replace(",", ""))
    str_name = datainfo.iloc[i]["Str_ID"]
    
    #filter out the parquet file
    if str_name not in file_names:
        partition = partitions[i]
        for p in range(1, partition+1):
            match_p = str_name + "_" + str(p)
            if match_p in str_id_p: 
               
                pass
            else:
                counter += 1
                if counter > batch_size:
                    batch += 1
                    with open(f"/home/sxr280/Spatialformer/scripts/find_gene_interaction/find_gene_interaction_{batch}.sh","w") as file:
                        file.write(all_bash_str)
                        all_bash_str = ""
                        bash_str = f"python find_gene_interaction.py --transcript_file /tmp/erda/Spatialformer/downloaded_data/raw/{str_name}/transcripts.csv.gz --number_cell {str(cell_num)} --partition {p} --dataname {j};\n"
                        all_bash_str += bash_str
                        counter = 0
                else:

                    bash_str = f"python find_gene_interaction.py --transcript_file /tmp/erda/Spatialformer/downloaded_data/raw/{str_name}/transcripts.csv.gz --number_cell {str(cell_num)} --partition {p} --dataname {j};\n"
                    all_bash_str += bash_str
            


        
   

