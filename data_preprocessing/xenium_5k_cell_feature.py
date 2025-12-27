import anndata as ad
import zarr
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import zipfile
import tempfile
import os
import shutil

class read_xenium_5k:
    """
    A class to read and preprocess the Xenium cell_feature_matrix.zarr.zip
    file into an AnnData object.

    The class handles:
    1. Unzipping the Zarr archive into a temporary directory.
    2. Reading the sparse CSC matrix components (data, indices, indptr) from Zarr.
    3. Constructing the feature metadata (.var) and observation metadata (.obs).
    4. Assembling the final AnnData object.
    5. Cleaning up the temporary directory.

    Note: This class follows the specific Zarr structure provided in the prompt.
    """

    def __init__(self, zip_file_path: str):
        """
        Initializes the loader with the path to the zipped Zarr file.
        
        :param zip_file_path: Path to the 'cell_feature_matrix.zarr.zip' file.
        """
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"Input file not found at: {zip_file_path}")
        self.zip_file_path = zip_file_path
        self.temp_dir = None
        self.zarr_path = None
        self.adata = None
    def hex2string(self, cell_ids, dataset_suffix):
        """
        convert the hex to the cell_id string
        """
        hex_to_shifted = {
                "0": "a",
                "1": "b",
                "2": "c",
                "3": "d",
                "4": "e",
                "5": "f",
                "6": "g",
                "7": "h",
                "8": "i",
                "9": "j",
                "a": "k",
                "b": "l",
                "c": "m",
                "d": "n",
                "e": "o",
                "f": "p"
            }
        # print(self.cell_ids[:10])    

        pad_func = lambda x: x.rjust(8, "0")
        #if you want to map by hex
        # cell_id_prefix = [
        #         pad_func("".join(hex_to_shifted[str(i)] for i in hex(x)[2:]))
        #         for x in cell_ids
        #     ]
        #if you want to map by int, still low chance to repeat
        cell_id_prefix = [
                pad_func("".join(hex_to_shifted[str(i)] for i in str(x)))
                for x in cell_ids
            ]

        cell_string = [
            "".join([i, "-", str(j)])
            for i, j in zip(cell_id_prefix, dataset_suffix)
        ]

        return cell_string


    def _read_zarr_store(self, zarr_store_path: str) -> ad.AnnData:
        """
        Reads the Zarr store components and constructs the AnnData object.

        :param zarr_store_path: Path to the root directory of the unzipped Zarr store.
        :return: An AnnData object.
        """
        print(f"Reading Zarr store from: {self.zip_file_path}")

        # Open the root Zarr group
        store = zarr.ZipStore(self.zip_file_path, mode="r")
        zarr_root = zarr.open(store, mode='r')

        # --- 1. Read Metadata from Group Attributes ---
        attrs = zarr_root["cell_features"].attrs
        number_cells = attrs.get('number_cells')
        number_features = attrs.get('number_features')
        feature_keys = attrs.get('feature_keys')
        feature_ids = attrs.get('feature_ids')
        feature_types = attrs.get('feature_types')

        if not all([number_cells, number_features, feature_keys, feature_types]):
            raise ValueError("Missing critical metadata attributes in Zarr store.")

        # --- 2. Construct .var (Features/Genes) ---
        var_df = pd.DataFrame(
            {'feature_id': feature_ids, 'feature_type': feature_types},
            index=pd.Index(feature_keys, name='feature_name')
        )
        print(f"Created .var with {var_df.shape[0]} features.")
        
        # --- 3. Read and Construct the Sparse Matrix (adata.X) ---
        # Data is stored in Compressed Sparse Column (CSC) format
        try:

            data = zarr_root["cell_features"]['data'][:]
            indices = zarr_root["cell_features"]['indices'][:]
            indptr = zarr_root["cell_features"]['indptr'][:]

        except KeyError as e:
            raise KeyError(f"Missing expected Zarr array component: {e}")

        # Create the scipy CSC matrix
        # Shape is (number_cells, number_features)
        X = csc_matrix(
            (data, indices, indptr),
            shape=(number_cells, number_features)
        )
        print(f"Created sparse matrix X with shape {X.shape}.")

        # --- 4. Construct .obs (Cells) ---
        # The /cell_id array is a 2xN array of uint32, representing (prefix, suffix).
        # We need to convert this to unique string IDs for AnnData's obs_names.
        cell_id_array = zarr_root['cell_features/cell_id'][:]
        
        # Convert the 2-column uint32 cell_id array into a list of strings
        # Format: "{prefix}-{suffix}"
        cell_names = self.hex2string(cell_id_array[:, 0], cell_id_array[:, 1])
        
        obs_df = pd.DataFrame(
            index=pd.Index(cell_names, name='cell_id')
        )
        print(f"Created .obs for {obs_df.shape[0]} cells.")

        # --- 5. Assemble the AnnData Object ---
        adata = ad.AnnData(
            X=X,
            obs=obs_df,
            var=var_df,
            dtype=X.dtype
        )
        
        # Store attributes in .uns for reference
        for key, value in attrs.items():
            if key not in ['feature_ids', 'feature_types', 'feature_keys']:
                adata.uns[key] = value

        print("AnnData object successfully assembled.")
        return adata

    def _cleanup(self):
        """Removes the temporary directory and all extracted files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)

    def __call__(self) -> ad.AnnData:
        """
        The main method to execute the full loading workflow.

        :return: The final AnnData object.
        """
        try:
            
            # 1. Read the Zarr store and construct AnnData
            self.adata = self._read_zarr_store(self.zip_file_path)
            
            return self.adata
        finally:
            # 2. Ensure cleanup happens, even if an error occurred during reading
            self._cleanup()

