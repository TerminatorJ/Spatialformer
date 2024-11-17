from datasets import load_from_disk, concatenate_datasets
import os
cache_dir = "/scratch/project_465001027/Spatialformer/cache"
cache_files = os.listdir(cache_dir)
os.environ["HF_TOKEN"] = "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"
# pair_files = [file for file in cache_files if file[-4:] == "pair"]
# all_datasets = []
# import pdb; pdb.set_trace()
# for pair_file in pair_files[40:]:
#     print(f"running {pair_file}")
#     # 
#     filename = os.path.join(cache_dir, pair_file)
#     dataset = load_from_disk(filename)
#     #adding the sample name
#     # import pdb; pdb.set_trace()
#     dataset = dataset.add_column("Sample_name", [pair_file] * len(dataset))
#     dataset.push_to_hub(pair_file)




filename = os.path.join(cache_dir, "xenium_THD0008_pair")
dataset = load_from_disk(filename)
dataset = dataset.add_column("Sample_name", [pair_file] * len(dataset))
dataset.push_to_hub(pair_file)

    # all_datasets.append(dataset)
# combined_datasets = concatenate_datasets(all_datasets)
# combined_datasets.push_to_hub(pair_file)


