import os
import glob
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm
from typing import List
import numpy as np
from xenium_5k_cell_centroid import xenium_5k_read_cell


BASE_DIR = "/scratch/project_465001820/Spatialformer/data/processed"
cache_dir = "/scratch/project_465001820/Spatialformer/cache"
RAW_DIR = "/scratch/project_465001820/Spatialformer/data/raw"


def map_cell_location(samples):
    """
    Map cell centroid coordinates to the dataset based on cell IDs.
    """
    batch_centroid_x = []
    batch_centroid_y = []
    batch_expression = []
    batch_Full_Tokens = []
    for i, cell_id in enumerate(samples["Cell_Ids"]):
        centroid_x = coordinate_dict[cell_id]["centroid_x"]
        centroid_y = coordinate_dict[cell_id]["centroid_y"]
        exp = samples["Expression"][i][0]
        # full_token = samples["Full_Tokens"][i]\
        full_token = np.array(samples["Full_Tokens"][i], dtype=np.int64).tolist()
        batch_centroid_x.append(centroid_x)
        batch_centroid_y.append(centroid_y)
        batch_expression.append(exp)
        batch_Full_Tokens.append(full_token)
    return {"centroid_x": batch_centroid_x, 
            "centroid_y": batch_centroid_y,
            "Expression": batch_expression,
            "Full_Tokens": batch_Full_Tokens}


def get_xenium_5k_dataset(datanames: List[str]):
    all_datasets = []

    for dataname in datanames:

        # match dataset directories: /processed/Xxx_arrow_2/
        pattern = f"{dataname}_arrow_[0-9]*"
        search_pattern = os.path.join(BASE_DIR, pattern)
        found_dirs = glob.glob(search_pattern)

        if not found_dirs:
            print(f"[WARN] No files found for {dataname}")
            continue

        dataset_list = []

        # ----- Load cell centroid coordinates -----
        INPUT_FILE = f"{RAW_DIR}/{dataname}/cells.zarr.zip"
        reader = xenium_5k_read_cell(INPUT_FILE)
        global coordinate_dict
        coordinate_dict = reader.process_file()
        # ------------------------------------------

        for dir_path in tqdm(found_dirs, desc=f"Loading {dataname}"):
            # load each dataset directory
            ds = load_from_disk(dir_path)

            # convert DatasetDict → Dataset
            if isinstance(ds, DatasetDict):
                if "train" in ds:
                    ds = ds["train"]
                else:
                    ds = concatenate_datasets(list(ds.values()))

            dataset_list.append(ds)

        # ---- Concatenate all shards for one sample ----
        combined = concatenate_datasets(dataset_list)

        # ---- Add sample name ----
        combined = combined.add_column(
            "Sample_Names",
            [dataname] * combined.num_rows
        )

        # ---- Map cell → centroid ----
        combined = combined.map(
            map_cell_location,
            batched=True,
            batch_size=1000,
            num_proc=32
        )

        all_datasets.append(combined)

    # ====== Final merge of ALL samples ======
    final_dataset = concatenate_datasets(all_datasets)

    out_path = os.path.join(cache_dir, "xenium_5k_combined_dataset")
    final_dataset.save_to_disk(out_path)

    print(f"[OK] Saved final combined dataset to {out_path}")
    return final_dataset


if __name__ == "__main__":
    datanames = [
        "Xenium_Prime_Human_Ovary_Cancer_FF_xe_outs",
        "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_xe_outs",
        "Xenium_Prime_Breast_Cancer_FFPE_xe_outs",
        "Xenium_Prime_Cervical_Cancer_FFPE_xe_outs",
        "Xenium_Prime_Human_Skin_FFPE_xe_outs",
        "Xenium_Prime_Human_Prostate_FFPE_xe_outs",
        "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_xe_outs",
        "Xenium_Prime_Human_Lung_Cancer_FFPE_xe_outs",
        "Xenium_Prime_Human_Ovary_FF_xe_outs"
    ]
    # get_xenium_5k_dataset(datanames)
    
    
    from datasets import Sequence, Value 
    dataset1 = load_from_disk("/scratch/project_465001820/Spatialformer/cache/xenium_pandavid_dataset6")
    dataset2 = load_from_disk("/scratch/project_465001820/Spatialformer/cache/xenium_5k_combined_dataset")
    # target_features = dataset2.features.copy()
    # target_features["Full_Tokens"] = Sequence(feature=Value(dtype='int64'))
    
    # target_features["Expression"] = Sequence(feature=Value(dtype='float64'))
    
    # dataset2_2 = dataset2.map(
    #     lambda x: x,  # Identity function - no actual transformation
    #     batched=True,
    #     batch_size=1000,
    #     num_proc=128,  # Use multiple CPUs
    #     features=target_features  # This enforces the int64 type
    # )
    # combined_dataset = concatenate_datasets([dataset1["train"], dataset2_2])
    # combined_dataset.save_to_disk("/scratch/project_465001820/Spatialformer/cache/xenium_5k_pandavid_dataset")
    lengths1 = dataset1["train"].map(
        lambda x: {"length": [len(tokens) for tokens in x["Full_Tokens"]]}, 
        remove_columns=dataset1["train"].column_names, 
        batched=True, 
        batch_size=1000, 
        num_proc=128
        )["length"]
    lengths2 = dataset2.map(
            lambda x: {"length": [len(tokens) for tokens in x["Full_Tokens"]]}, 
            remove_columns=dataset2.column_names, 
            batched=True, 
            batch_size=2000, 
            num_proc=256
        )["length"]
    import numpy as np
    print("lengths1 sum:", np.sum(lengths1))
    print("lengths2 sum:", lengths2.sum())
