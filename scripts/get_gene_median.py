import multiprocessing
import scanpy as sc
from collections import defaultdict
import os
from tqdm import tqdm
# from scipy.sparse import csr_matrix
from scipy.sparse import issparse
import numpy as np
import pickle
def process_file(h5file):
    """
    Function to process a single .h5ad file and return the gene medians in a dictionary.
    """
    print(f"Running {h5file}")
    adata = sc.read_h5ad(h5file)
    
    gene_dataset_nonzero = []
    for i in range(adata.shape[1]):
        if issparse(adata.X):
            # Convert the i-th column to a dense array
            column_data = adata.X[:, i].toarray().flatten()
        else:
            column_data = adata.X[:, i].flatten()

        non_zero_data = column_data[column_data.nonzero()]
        
        # if non_zero_data.size > 0:
        #     median_value = np.median(non_zero_data)
        # else:
        #     # If there are no non-zero elements, handle appropriately (e.g., use NaN)
        #     median_value = np.nan
            
        gene_dataset_nonzero.append(non_zero_data)

    genes = list(adata.var["gene_name"])
    cellnum = adata.X.shape[0]
    
    file_gene_dict = {gene: nonzero for gene, nonzero in zip(genes, gene_dataset_nonzero)}
    file_cell_dict = {h5file: cellnum}
    
    return file_gene_dict, file_cell_dict

def get_all_genes_parallel(h5file_list):
    """
    Processes multiple .h5ad files in parallel to extract gene lists.
    """
    global_gene_dict = defaultdict(list)
    global_cell_dict = defaultdict(list)

    # Initiate a pool of workers based on the CPU count
    with multiprocessing.Pool() as pool:
        results = []
        for h5file in tqdm(h5file_list):
            result = pool.apply_async(process_file, args=(h5file,))
            results.append(result)

        for result in results:
            file_gene_dict, file_cell_dict = result.get()
            for gene, nonzero in file_gene_dict.items():
                global_gene_dict[gene].extend(nonzero)
            for h5file, cellnum in file_cell_dict.items():
                global_cell_dict[h5file].append(cellnum)

    return global_gene_dict, global_cell_dict

if __name__ == "__main__":
    root = "/tmp/erda/Spatialformer/downloaded_data/processed"
    all_files = os.listdir(root)
    h5files = [os.path.join(root, file) for file in all_files if file.endswith(".h5ad")]
    # import pdb; pdb.set_trace()
    # h5files = ["/tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hSkin_nondiseased_section_2_FFPE_outs.h5ad","/tmp/erda/Spatialformer/downloaded_data/processed/Xenium_V1_hHeart_nondiseased_section_FFPE_outs.h5ad"]
    gene_dict, cell_dict = get_all_genes_parallel(h5files)
    #compute the median values
    gene_median_dict = {gene: np.median(count_list) for gene,count_list in gene_dict.items()}

    pickle.dump(gene_dict, open("/home/sxr280/Spatialformer/data/Xenium_median_gene_exp.pkl", "wb"))
    pickle.dump(gene_median_dict, open("/home/sxr280/Spatialformer/data/Xenium_median_gene_final_exp.pkl", "wb"))
    pickle.dump(cell_dict, open("/home/sxr280/Spatialformer/data/Xenium_cell_number.pkl", "wb"))
    # For debugging: print a summary of the collected data
    print(f"Number of genes processed: {len(gene_dict)}")
    print("gene median:", gene_dict)
    print(f"Number of cells in each file processed: {cell_dict}")
