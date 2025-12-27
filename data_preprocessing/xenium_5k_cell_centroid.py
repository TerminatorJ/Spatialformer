import zarr
import json
import numpy as np
from typing import Dict, Any, List

class xenium_5k_read_cell:
    """
    A class to read cell ID and cell centroid coordinates from a
    Xenium 'cells.zarr.zip' file and output the data to a JSON file.
    """

    def __init__(self, input_zarr_path: str):
        """
        Initializes the Xenium5kReadCell reader.

        Args:
            input_zarr_path: Path to the input 'cells.zarr.zip' file.
        """
        self.input_zarr_path = input_zarr_path


    
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

    def process_file(self) -> Dict[str, Any]:
        """
        Reads the necessary data from the Zarr file, processes it, and
        writes the final output to a JSON file.

        The required data is:
        1. Cell ID from '/cell_id'.
        2. Cell centroid X and Y coordinates from '/cell_summary'.

        Returns:
            A dictionary containing the results read from the Zarr file.
        """
        print(f"Opening Zarr file: {self.input_zarr_path}")

        try:
            # Use zarr.open to access the contents of the ZIP-backed Zarr store
            
            store = zarr.ZipStore(self.input_zarr_path, mode='r')
            zarr_root = zarr.open(store, mode='r')

            # --- 1. Read Cell IDs ---
            if 'cell_id' not in zarr_root:
                raise KeyError("The required array '/cell_id' was not found.")
            # Reads the 2-column numpy array
            raw_cell_ids = zarr_root['cell_id'][:]
            cell_ids_str = self.hex2string(raw_cell_ids[:, 0], raw_cell_ids[:, 1])
            print(f"Successfully read {len(cell_ids_str)} cell IDs.")
            # --- 2. Read Cell Centroids from Cell Summary ---
            if 'cell_summary' not in zarr_root:
                raise KeyError("The required array '/cell_summary' was not found.")
            
            cell_summary = zarr_root['cell_summary']
            
            # The columns are defined as 'cell_centroid_x' and 'cell_centroid_y'
            # We access the data by column name, which is supported by Zarr/NumPy
            # if 'cell_summary' is a structured array. If it's a simple 2D array,
            # we must rely on column order (assuming 0=x, 1=y based on the
            # provided attribute description order).
            
            # Based on the documentation describing /cell_summary columns:
            # Field: cell_centroid_x (1st column)
            # Field: cell_centroid_y (2nd column)
            
            # Read the entire array and access columns by index if structure isn't guaranteed
            summary_data = cell_summary[:]
            # Assuming the columns are in the order provided in the documentation:
            # Column 0: cell_centroid_x
            # Column 1: cell_centroid_y
            
            # Check for sufficient columns
            if summary_data.ndim != 2 or summary_data.shape[1] < 2:
                 raise ValueError("'/cell_summary' is not a 2D array with at least 2 columns.")
            cell_centroid_x = summary_data[:, 0]
            cell_centroid_y = summary_data[:, 1]
            
            num_cells = len(cell_ids_str)
            if len(cell_centroid_x) != num_cells:
                raise ValueError(f"Mismatch in cell count: ID array has {num_cells} cells, "
                                 f"but summary has {len(cell_centroid_x)} rows.")

            print("Successfully read cell centroid coordinates.")
            
            # --- 3. Combine Data and Format for JSON ---
            output_data = {}
            for i in range(num_cells):
                cell_id = cell_ids_str[i]
                output_data[cell_id] = {
                    "centroid_x": float(cell_centroid_x[i]), # Ensure standard Python types for JSON
                    "centroid_y": float(cell_centroid_y[i])
                }


            print(f"Successfully processed {num_cells} cells.")
            
            return output_data

        except Exception as e:
            error_message = f"An error occurred during file processing: {e}"
            print(error_message)
            return {"status": "error", "message": error_message}
        finally:
            # Ensure the Zarr store is closed
            if 'store' in locals():
                store.close()

# --- Example Usage (Commented out) ---
# NOTE: This example requires an actual 'cells.zarr.zip' file to run.

if __name__ == '__main__':
    # Define paths
    INPUT_FILE = "/scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs/cells.zarr.zip"

    # Instantiate and run the class
    reader = xenium_5k_read_cell(INPUT_FILE)
    results = reader.process_file()
    print("\nProcessing Results:")
    print(json.dumps(results, indent=5))