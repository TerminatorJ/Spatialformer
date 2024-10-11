# %% [markdown]
# This jupyter notebook is used to process the unmatch david annotation data, in order to   
# match the cell id and cell annotation when constructing the cell embeddings 

#
import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import scipy.io
import argparse
import torch





def get_real_id(obj, exp_mtx, vec, changed_id, sample_name):
    # Convert numpy array to PyTorch tensor and move to GPU
    # import pdb; pdb.set_trace()
    vec = vec.squeeze()
    # Find the matching column
    # matches = (exp_mtx == vec.view(-1, 1)).all(dim=0).nonzero(as_tuple=True)[0]

    # Compute the difference and find the column with minimum sum of differences
    diff = torch.abs(exp_mtx - vec.view(-1, 1))
    match_scores = diff.sum(dim=0)
    min_score, matches = match_scores.min(dim=0)
    
    # Check if the match is exact (i.e., all differences are zero)
    if min_score.item() != 0:
        print(changed_id, "cannot be matched")
        return {}

    real_cell_id = obj.obs.index[matches.item()]
    cell_type = ann[ann["cell_id"] == changed_id]["broad_CT5"].values[0]

    return {f"{sample_name}_{real_cell_id}": cell_type}

# %%

def get_ann(h5_path = "/scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUHD116A__20230308__003730/outs/cell_feature_matrix.h5"):
    sample_name = h5_path.split("/")[-3].split("__")[-3]
    current_df = exp_df.filter(like = sample_name)#df with changed name
    current_ts = torch.tensor(current_df.values, dtype=torch.int).cuda()
    sample_obj = sc.read_10x_h5(h5_path)
    exp_ts = torch.tensor(sample_obj.X.T.toarray(), dtype=torch.int).cuda()#raw matrix

    ann_dict = {}


    for i in tqdm(range(0, current_df.shape[1])):
            # batch = (sample_obj, exp_ts, current_ts[:, i], list(current_df.columns)[i], sample_name)
        cur_dict = get_real_id(sample_obj, exp_ts, current_ts[:, i], list(current_df.columns)[i], sample_name)
        ann_dict.update(cur_dict)

    return ann_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='getting the annotation for each sample')
    parser.add_argument('--h5_path', type=str, default="None", help='The name of the raw h5 dataset')
    args = parser.parse_args()

    #loading all exp vector
    exp_path = "/scratch/project_465001027/spatialformer/david_data/expression_data.mtx"
    # Load the matrix from the .mtx file
    matrix = scipy.io.mmread(exp_path)
    # Check the type of the loaded matrix
    print(f"Loaded matrix type: {type(matrix)}")
    # Convert to a dense NumPy array if needed
    dense_matrix = matrix.toarray()

    #loading the colnames and rownames
    changed_ids = pd.read_csv("/scratch/project_465001027/spatialformer/david_data/expression_data_colnames.csv")
    changed_ids = list(changed_ids["x"])
    gene_names = pd.read_csv("/scratch/project_465001027/spatialformer/david_data/expression_data_rownames.csv")
    gene_names = list(gene_names["x"])

    exp_df = pd.DataFrame(dense_matrix, columns = changed_ids, index = gene_names)

    #loading the nich annotation file that comes from GEO object
    ann_path = "/scratch/project_465001027/spatialformer/david_data/nich_annotation_df.csv"
    ann = pd.read_csv(ann_path)
    # import pdb;pdb.set_trace()

    all_ann_dict = {}

    ann_dict = get_ann(h5_path = args.h5_path)
    sample_name = args.h5_path.split("/")[-3].split("__")[-3]
    all_ann_dict.update({sample_name : ann_dict})


    # %%
    #save the annotation info as json file
    import json
    file_path = f'{sample_name}_annotation.json'

    # Write the dictionary to a JSON file
    with open("/scratch/project_465001027/spatialformer/david_data/" + file_path, 'w') as file:
        json.dump(all_ann_dict, file, indent=3) 





#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD106__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD110__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__VUILD115__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__THD0011__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD117LF__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD117MF__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__TILD175__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD78LF__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD78MF__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD91LF__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003400__VUILD91MF__20230313__191400/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUHD069__20230308__003731/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUHD095__20230308__003731/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUHD113__20230308__003731/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD48MF__20230308__003731/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD104LF__20230308__003731/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003789__VUILD105MF__20230308__003731/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUHD116A__20230308__003730/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUHD116B__20230308__003731/outs/cell_feature_matrix.h5 
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96LF__20230308__003730/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD96MF__20230308__003730/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD102LF__20230308__003731/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD102MF__20230308__003730/outs/cell_feature_matrix.h5
#python /scratch/project_465001027/spatialformer/scripts/get_cell_annotation.py --h5_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003817__VUILD107MF__20230308__003731/outs/cell_feature_matrix.h5




