#For the 17M spatial corpus datasets, we need to know the genes vocabulary that need to be set up in the PTM model.

#loading the ".h5" files for all the datasets.
import os
import scanpy as sc
import numpy as np
import json


def get_gene(file_names):
    genes = []
    total_cells = 0
    for file in file_names:
        try:
            # Read the data file using scanpy
            sc_data = sc.read_10x_h5(file)  # Change `read_data` to `read` for an AnnData object
            # import pdb; pdb.set_trace()
            cell_num = len(sc_data.obs)
            total_cells += cell_num
            # Extract the gene names from the 'var' DataFrame
            genes += list(sc_data.var_names) # Use var_names instead of var.unique()
            print(f"{file} has {len(sc_data.var_names)} genes")
            print(f"{file} has {cell_num} cells")
        except FileNotFoundError:
            print(f"{file} still downloading or failed to be downloaded")

    # import pdb;pdb.set_trace()
    ugene = np.unique(genes)
    return ugene, total_cells



if __name__ == "__main__":
    root_path = "/tmp/erda/Spatialformer/downloaded_data/raw"

    mouse_names = ["Xenium_V1_FFPE_TgCRND8_17_9_months_outs",
                   "Xenium_V1_FFPE_TgCRND8_2_5_months_outs",
                   "Xenium_V1_FFPE_TgCRND8_5_7_months_outs",
                   "Xenium_V1_FFPE_wildtype_13_4_months_outs",
                   "Xenium_V1_FFPE_wildtype_2_5_months_outs",
                   "Xenium_V1_FFPE_wildtype_5_7_months_outs",
                   "Xenium_V1_mouse_pup_outs",
                   "Xenium_V1_mouse_Colon_FF_outs",
                   "Xenium_V1_FF_Mouse_Brain_Coronal_outs",
                   "Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs",
                   "Xenium_Prime_Mouse_Brain_Coronal_FF_outs",
                   "Xenium_V1_mFemur_formic_acid_24hrdecal_section_outs",
                   "Xenium_V1_mFemur_EDTA_3daydecal_section_outs",
                   "Xenium_V1_mFemur_EDTA_PFA_3daydecal_section_outs",
                   "Xenium_V1_FF_Mouse_Brain_MultiSection_1_outs",
                   "Xenium_V1_FF_Mouse_Brain_MultiSection_2_outs",
                   "Xenium_V1_FF_Mouse_Brain_MultiSection_3_outs"]
    large_human_file = [
            "Xenium_Prime_Human_Ovary_FF_outs",
            "Xenium_Prime_Ovarian_Cancer_FFPE_outs",
            "Xenium_Prime_Cervical_Cancer_FFPE_outs",
            "Xenium_Prime_Human_Skin_FFPE_outs",
            "Xenium_Prime_Human_Prostate_FFPE_outs",
            "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs",
            "Xenium_V1_hBoneMarrow_nondiseased_section_outs",
            "Xenium_V1_hBone_nondiseased_section_outs"
    ]
    #only select the human Xenium datasets
    # import pdb;pdb.set_trace()
    data_names = [os.path.join(root_path,file, "cell_feature_matrix.h5") for file in os.listdir("/tmp/erda/Spatialformer/downloaded_data/raw") if (".zip" not in file) and (file not in mouse_names) and (file not in large_human_file)]

    

    genes, total_cells = get_gene(data_names)
    # import pdb; pdb.set_trace()

    #including the vocabulary from David data
    #loading the json file
    with open("/home/sxr280/Spatialformer/tokenizer/token.json", "r") as file:
        David_vocab_dict = json.load(file)

    # import pdb; pdb.set_trace()
    old_genes = list(David_vocab_dict.keys())[10:]
    all_genes = np.unique(list(genes) + old_genes)


    print(f"There are {len(all_genes)} genes exists: {all_genes}")
    print(f"There are {total_cells} cells exists")

    
    prefix = ["<pad>", "<CLS>", "<mask>"]
    organism = ["Human", "Mouse"]
    assay = ["Merfish", "Xenium", "CosMx"]
    tissue =["Lung", "Bone", "Brain", "Breast", "Cervix", "Colon", "Heart", "Liver", "Lymph Node", "Ovary", "Pancreas", "Prostate", "Skin", "Tonsil", "Kidney", "LymphNode", "BoneMarrow"]
    condition = ["Healthy", "Disease"]

    #currectly, we don't want to apply tissue meta data
    vocab = prefix + organism + assay + tissue + condition + sorted(all_genes)

    #rank the vocabulary
    vocab_dict = {gene: i for i, gene in enumerate(vocab)}

    #saving the gene vocabulary
    with open("/home/sxr280/Spatialformer/tokenizer/tokenv3.json", 'w') as file:
        json.dump(vocab_dict, file, indent=3) 
    


