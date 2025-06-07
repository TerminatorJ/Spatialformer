from datasets import load_from_disk
import os

# Set the base directory where cached files are stored
base_dir = "/scratch/project_465001027/Spatialformer/cache"

# List all files in the base directory
all_files = os.listdir(base_dir)

# Filter the files to only include those that have "pair" in their name
pair_files = [f for f in all_files if "pair" in f]

# Iterate over each pair file, loading and pushing it to the hub
for pair_file in pair_files:
    # Construct the full path to the pair file
    full_path = os.path.join(base_dir, pair_file)

    # Load the dataset from disk
    dataset = load_from_disk(full_path)

    # Push the dataset to the Hugging Face hub
    dataset.push_to_hub(f"TerminatorJ/{pair_file}", token="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")