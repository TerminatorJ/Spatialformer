import simfish
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def sim_single_cell(cell_exp, ct, ct_id, gene_list, cell_id):
    '''
    Simulates and plots gene expression pattern localization for a single cell.

    Args:
    - cellmor (any): Indicates morphology or template used for simulation.
    - plot (bool): Whether to plot the simulated localization pattern.
    - cellid (any): Identifier for the cell being simulated.
    - ct (int): An identifier representing the cell type category for classification.
    - cellseed (int): A seed to ensure reproducibility for spot number generation.
    - gene_vocab (list): List of gene identifiers.
    - gene_patterns (list): List of patterns corresponding to each gene.

    Returns:
    - pandas.DataFrame: DataFrame of coordinates and features related to RNA spots simulated in the cell.
    '''
    # import pdb; pdb.set_trace()
#     moridx = cellmor["id"].values[0]
    cellmors = pd.read_csv("/scratch/project_465001027/Spatialformer/downstream/subcellular_localization_prediction/templates/index.csv", sep = ";")
    # cellmors[ct_id]
    ct_mor = cellmors.iloc[ct_id: ct_id+1,]
    pattern_dir = "/scratch/project_465001027/Spatialformer/downstream/subcellular_localization_prediction/templates"
    gene_coords = []
    patterns = ["random", "intranuclear", "extranuclear", "perinuclear", "pericellular"]
    for i, exp in tqdm(enumerate(cell_exp)):
        pattern = np.random.choice(patterns)   
        gene_name = gene_list[i]
        # try:
            # import pdb; pdb.set_trace()
        if exp > 0:
            instance_coord = simfish.simulate_localization_pattern(pattern_dir, n_spots=int(exp), index_template=ct_mor, pattern=pattern, proportion_pattern=1)
            rna_coord = instance_coord["rna_coord"]
            coord_df = pd.DataFrame({
            "z": rna_coord[:, 0], 
            "y": rna_coord[:, 1], 
            "x": rna_coord[:, 2], 
            "feature_name": rna_coord.shape[0] * [gene_name], 
            "cell_id": rna_coord.shape[0] * [cell_id],
            "cell_type": rna_coord.shape[0] * [ct],
            "pattern": rna_coord.shape[0] * [pattern]
            })
            gene_coords.append(coord_df)
        else:
            # gene_coords.append()
        # except:
            continue
    gene_df = pd.concat(gene_coords, ignore_index=True)
    
    return gene_df
