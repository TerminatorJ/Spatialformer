import os
import pickle
import scanpy as sc
import numpy as np
from collections import defaultdict
import argparse
import sys # Import sys for potential exit if dataname is invalid
from tqdm import tqdm

# Define the list of available dataset names
AVAILABLE_DATANAMES = [
    "Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs",
    "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs",
    "Xenium_Prime_Breast_Cancer_FFPE_xe_outs",
    "Xenium_Prime_Cervical_Cancer_FFPE_xe_outs",
    "Xenium_Prime_Human_Skin_FFPE_xe_outs",
    "Xenium_Prime_Human_Prostate_FFPE_xe_outs",
    "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs",
    "Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs"
]

BASE_PATH = "/scratch/project_465001820/Spatialformer/data"

def get_gene_median(adata, gene_exp):
    """
    Processes gene expression data in a vectorized way by directly accessing 
    the non-zero data from the sparse matrix.
    
    Args:
        adata (AnnData): The AnnData object (potentially backed='r').
        gene_exp (defaultdict): A defaultdict(list) to store all gene expression values.
    """
    # Ensure adata.X is in a format where .data and .indices are readily accessible.
    # CSR (Compressed Sparse Row) is best for row-based access (cells), 
    # but CSC (Compressed Sparse Column) is better for column-based access (genes).
    # We will convert to CSC to efficiently extract gene data.
    
    # 1. Convert to CSC format for fast column slicing (gene-wise access)
    # The .tocsc() call is the 'vectorization' step for this sparse operation.
    X_csc = adata.X[:].tocsc()
    
    # Get the list of gene names
    genes = adata.var["gene_name"].values
    
    # 2. Iterate over the columns (genes) of the CSC matrix.
    # This loop is necessary to append to the gene-specific lists in the dictionary,
    # but it is much faster than the previous version because it iterates over 
    # the efficiently structured sparse columns, not repeatedly converting to dense arrays.
    for i, gene in tqdm(enumerate(genes), desc="Processing all the genes"):
        
        # Get the sparse column for the current gene
        sparse_column = X_csc[:, i]
        
        # 3. Extract the non-zero values directly from the column's .data attribute
        # .data holds only the non-zero values for that column.
        non_zero_values = sparse_column.data
        # 4. Convert to list and extend
        # Check if the list conversion is necessary based on the data type, 
        # but numpy arrays are generally fine for .extend after .tolist()
        gene_exp[gene].extend(non_zero_values.tolist())

def process_data_and_calculate_median(dataname_list):
    """
    Main function to process the specified datasets, calculate the median 
    gene expression across all cells in these datasets, and save the result.

    Args:
        dataname_list (list): A list of dataset names to process.
    """
    
    magic_path = lambda *x: os.path.join(os.path.abspath(BASE_PATH), *x)
    adata_files = [magic_path("processed", i+".h5ad") for i in dataname_list]

    # Use defaultdict to store gene expression values
    gene_exp = defaultdict(list)

    # Process each dataset
    print(f"\n{'#'*70}")
    print(f"Starting processing for {len(dataname_list)} dataset(s).")
    print(f"{'#'*70}")
    
    for idx, adata_file in enumerate(adata_files):
        current_dataname = dataname_list[idx]
        print(f"\n{'='*50}")
        print(f"Processing dataset {idx+1}/{len(dataname_list)}: {current_dataname}")
        print(f"{'='*50}")
        if not os.path.exists(adata_file):
            print(f"Error: AnnData file not found at {adata_file}. Skipping.")
            continue
        
        # Load the AnnData file in read-only (backed) mode
        try:
            adata = sc.read_h5ad(adata_file, backed='r')
            print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
        except Exception as e:
            print(f"Error loading {adata_file}: {e}. Skipping.")
            continue
        
        # Get gene expression values
        get_gene_median(adata, gene_exp)
        del adata  # Free memory
        print(f"Finished processing {current_dataname}")
   
    # Save results
    output_file = magic_path(f"Xenium_5k_gene_final_exp_{current_dataname}.pkl")
    try:
        with open(output_file, "wb") as f:
            pickle.dump(gene_exp, f)
        print(f"Saving pickle file to {output_file}")  
    except Exception as e:
        print(f"Error saving pickle file to {output_file}: {e}")

# --- Argument Parsing ---
def main():
    parser = argparse.ArgumentParser(
        description="Process Xenium AnnData files to calculate the median gene expression.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--dataname", 
        type=str,
        default="all",
        help=(
            "The name of the dataset to process. \n"
            "Use 'all' to process all available datasets. \n"
            "Available names:\n  " + "\n  ".join(AVAILABLE_DATANAMES)
        )
    )
    
    args = parser.parse_args()

    # Determine which datasets to process
    if args.dataname == "all":
        datasets_to_process = AVAILABLE_DATANAMES
        print("Argument --dataname is 'all'. Processing ALL datasets.")
    elif args.dataname in AVAILABLE_DATANAMES:
        datasets_to_process = [args.dataname]
        print(f"Argument --dataname is '{args.dataname}'. Processing this single dataset.")
    else:
        print(f"Error: '{args.dataname}' is not a valid dataname or 'all'.")
        print("Please choose one of the available names or 'all'.")
        sys.exit(1) # Exit the script with an error code

    # Execute the core logic
    process_data_and_calculate_median(datasets_to_process)

if __name__ == "__main__":
    main()