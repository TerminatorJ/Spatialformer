from datasets import Dataset, load_dataset, load_from_disk

# Example dataset
dataset = load_from_disk("/scratch/project_465001820/Spatialformer/cache/xenium_5k_pandavid_dataset_v2")
dataset.cleanup_cache_files()

dataset = dataset.filter(
    lambda x: [
        len(x["Rows"][i]) != 0
        and len(x["Full_Tokens"][i]) != 0
        for i in range(len(x["Rows"]))
    ],
    batched=True,
    num_proc=128,
)

dataset.save_to_disk("/scratch/project_465001820/Spatialformer/cache/xenium_5k_pandavid_dataset_v2", num_proc=128)