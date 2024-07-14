---

## Step 1: Preprocessing the Dataset

1. **Dataset Source**: The dataset you're using is the "10X lung healthy" dataset, which can be found on the 10X official website.

2. **New Gene Vocabulary**: If you want to use a new dataset to generate a new gene vocabulary for tokens, you can do so. Make sure to follow the same preprocessing steps outlined below.

3. **Processed Data Storage**: All processed data related to this pipeline should be stored in the Hugging Face dataset.

4. **Transcript Criteria**:
   - The number of transcripts in a cell should be more than 100.
   - Adjacent transcripts should be identified when the cell itself and nearby nearest genes have more than 3 transcripts.
   - The quality score for each transcript should be higher than 20.
   - Filter out "unassigned" cells and "negative" or "blank" genes.

## Step 2: Running the "find_gene_interaction.py" Script

To obtain the gene-gene interaction matrix, follow these steps:

1. Run the "find_gene_interaction.py" script located in the script directory. You'll need to specify the following arguments:
   - `--transcript_file`: Path to the input transcript.csv file.
   - `--number_cell`: Number of cells used for calculation (useful for debugging).
   - `--partition`: Partition of cell IDs (e.g., 1, 2, 3) to run separately.
   - `--dataname`: Overall name of the dataset (e.g., "THD0008").
   - `--datapath_name`: Name of the data path to store raw and processed datasets.

Example commands for three partitions (adjust as needed):
```bash
python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/transcripts.csv --number_cell 57889 --partition 1 --dataname THD0008
python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/transcripts.csv --number_cell 57889 --partition 2 --dataname THD0008
python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/transcripts.csv --number_cell 57889 --partition 3 --dataname THD0008
```

# Gene Interaction Analysis

After running the `find_gene_interaction.py` script, the following files will be generated in the specified `--datapath_name` directory:

- `THD0008XXXX_1.h5`
- `THD0008XXXX_2.h5`
- `THD0008XXXX_3.h5`

## Arguments for `find_gene_interaction.py`

| Argument              | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `--transcript_file`   | Path to the input `transcript.csv` file                                                       |
| `--threshold`         | Threshold for filtering transcripts (default: 100)                                            |
| `--radius`            | Radius to separate compartments (default: 5)                                                  |
| `--pair_threshold`    | Pair threshold for same and different transcripts (default: 3)                                |
| `--number_cell`       | Number of cells used for calculation (useful for debugging, default: 2)                        |
| `--partition`         | Partition of cell IDs for separate runs (default: 1)                                          |
| `--chunks`            | Number of chunks for dividing cell IDs (default: 20000)                                        |
| `--dataname`          | Overall dataset name (default: None)                                                           |
| `--datapath_name`     | Name of the data path for storing raw and processed datasets                                   |

## Step 3: Embedding Information

To consolidate all information into a single object, run the `build_h5ad.py` script in the `utils` directory. Define sample status and assay types in the anndata object using the following command:

```bash
python build_h5ad.py --partitions 6 --data_name relabel_output-XETG00048__0003392__VUILD106__20230313__191400 --matrix_name VUILD106_gene_interaction --condition Disease --tissues Lung --species Human --assay Xenium
```

## Step 4

Once we got the processed "XX.h5ad", we can generate the dataset object and dataloader to be used when training the model.

```bash
python h5toloader.py --data_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/processed/relabel_output-XETG00048__0003392__THD0008__20230313__191400.h5ad 
```

The processed huggingface array files will be stored in the processed directory as the entitle "XX_array" at the end of the file name.






