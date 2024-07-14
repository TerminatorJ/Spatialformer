import os 
import sys 
import h5py
from pathlib import Path
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
sys.path.append("p_path")
data_path = os.path.join(p_path, "david_data")

# import pdb; pdb.set_trace()
names = [file.split("__")[2] for file in os.listdir(data_path) if file.startswith("relabel")]
data_names = [file for file in os.listdir(data_path) if file.startswith("relabel")]

#getting the number of the partitions
name_file = {}
for file in os.listdir(data_path):
    if (".h5" in file) and (file.split("_")[0] in names) and ("merge" not in file):
        this_name = file.split("_")[0]
        name_file.setdefault(this_name, []).append(file)


# import pdb; pdb.set_trace()
shell_str = ""
h5loader_str = ""

for i, name in enumerate(names): 
    cond = "Healthy" if "HD" in name else "Disease"
    shell_str += f"python build_h5ad.py --partitions {len(name_file[name])} --data_name {data_names[i]} --matrix_name {name}_gene_interaction --condition {cond} --tissues Lung --species Human --assay Xenium" + "\n"
    h5loader_str += f"python h5toloader.py --data_path /scratch/project_465001027/spatialformer/david_data/{data_names[i]}/processed/{data_names[i]}.h5ad" + "\n"
# import pdb; pdb.set_trace()
with open("./run_build_h5ad_shell.sh", "w") as f:
    f.write(shell_str)
    
with open("./run_h5toloader_shell.sh", "w") as f1:
    f1.write(h5loader_str)
    



