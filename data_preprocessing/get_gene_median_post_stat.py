import os
import pickle
import scanpy as sc
from tqdm import tqdm


datanames=[
    "Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs",
    "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs",
    "Xenium_Prime_Breast_Cancer_FFPE_xe_outs",
    "Xenium_Prime_Cervical_Cancer_FFPE_xe_outs",
    "Xenium_Prime_Human_Skin_FFPE_xe_outs",
    "Xenium_Prime_Human_Prostate_FFPE_xe_outs",
    "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs",
    "Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
    "Xenium_V1_hKidney_cancer_section_outs",
     "Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs",
     "Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs",
     "Xenium_V1_hLiver_nondiseased_section_FFPE_outs",
     "Xenium_V1_hLung_cancer_section_outs",
     "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs",
     "Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs",
     "Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs",
     "Xenium_V1_hBone_nondiseased_section_outs",
     "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs",
     "Xenium_V1_hHeart_nondiseased_section_FFPE_outs",
     "Xenium_V1_human_Pancreas_FFPE_outs",
     "Xenium_V1_FFPE_Human_Brain_Alzheimers_With_Addon_outs",
     "Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs",
     "Xenium_V1_hBoneMarrow_nondiseased_section_outs",
     "Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs",
     "Xenium_V1_hSkin_Melanoma_Base_FFPE_outs",
     "Xenium_V1_FFPE_Human_Breast_ILC_outs",
     "Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs",
     "Xenium_V1_humanLung_Cancer_FFPE_outs",
     "Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs",
     "Xenium_V1_FFPE_Human_Breast_ILC_With_Addon_outs",
     "Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs",
     "Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE_outs",
    "Xenium_V1_hColon_Cancer_Base_FFPE_outs",
     "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs",
    "Xenium_V1_FFPE_Human_Breast_IDC_outs",
     "Xenium_V1_hLiver_cancer_section_FFPE_outs",
     "Xenium_V1_hColon_Non_diseased_Base_FFPE_outs",
     "Xenium_V1_hKidney_nondiseased_section_outs",
     "Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs",
     "Xenium_V1_FFPE_Human_Brain_Glioblastoma_With_Addon_outs",
     "Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs",
     "Xenium_V1_hLymphNode_nondiseased_section_outs",
     "Xenium_V1_hPancreas_nondiseased_section_outs",
     "Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs",
     "Xenium_V1_hColon_Cancer_Add_on_FFPE_outs",
     "Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs",
     "Xenium_V1_Human_Brain_GBM_FFPE_outs",
     "Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs",
]

base_path = "/scratch/project_465001820/Spatialformer/data"
magic_path = lambda *x: os.path.join(os.path.abspath(base_path), *x)

all_exp = {}
for dn in tqdm(datanames, desc="running all datasets"):
    try:
        exp_dict = pickle.load(open(f"/scratch/project_465001820/Spatialformer/data/Xenium_5k_gene_final_exp_{dn}.pkl", "rb"))
        for gene in exp_dict.keys():
            if gene not in all_exp.keys():
                all_exp[gene] = exp_dict[gene]
            else:
                all_exp[gene].extend(exp_dict[gene])
    except:
        pass
import numpy as np

gene_median = {}
empty_genes = []
for gene in all_exp.keys():
    non_zero_value = all_exp[gene]
    if len(non_zero_value) == 0:
        print("The empty genes are: ",gene)
        empty_genes.append(gene)
        gene_median[gene] = 1
    else:
        median = np.median(non_zero_value)
        gene_median[gene] = median
nan_gene = [k for k, v in gene_median.items() if not np.isnan(v)]
import pdb; pdb.set_trace()  
import pickle

pickle.dump(gene_median, open("/scratch/project_465001820/Spatialformer/data/gene_median.pkl", "wb"))