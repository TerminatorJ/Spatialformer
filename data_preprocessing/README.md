---
# Preprocessing the Dataset  

After getting the original dataset, the preprocessing steps are required.  

**Preprocessing settings**

   - The number of transcripts in a cell should be more than 30.
   - Filter out "unassigned" cells and "negative" or "blank" genes.
   - Adjacent transcripts should be identified when the cell itself and nearby nearest genes have more than 3 transcripts.
   - The quality score for each transcript should be better for higher than 20.
   

## Step 1: Find the gene-gene co-occurrence

For convenience, you can run the scripts in this directory to get the gene-gene co-occurrence matrix.


Example commands for three partitions:
```bash
python find_gene_interaction.py --transcript_file /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/outs/transcripts.csv --number_cell 57889 --Partition 1 --dataname THD0008
```
## Arguments for `find_gene_interaction.py`

| Argument                 | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `--transcript_file`|  Path to the input `transcript.csv` file                                                       |
| `--threshold`         | Threshold for filtering transcripts (default: 100)                                            |
| `--radius`            | Radius to separate compartments (default: 5)                                                  |
| `--pair_threshold`    | Pair threshold for same and different transcripts (default: 3)                                |
| `--number_cell`       | Number of cells used for calculation (useful for debugging, default: 2)                        |
| `--partition`         | Partition of cell IDs for separate runs (default: 1)                                          |
| `--chunks`            | Number of chunks for dividing cell IDs (default: 20000)                                        |
| `--dataname`          | Overall dataset name (default: None)                                                           |


## Step 2: Combine all the information

To consolidate all information into a single object, run the `build_h5ad.py` script in the `utils` directory. Define sample status and assay types in the anndata object using the following command:

```bash
python build_h5ad.py --partitions 6 --data_name relabel_output-XETG00048__0003392__VUILD106__20230313__191400 --matrix_name VUILD106_gene_interaction --condition Disease --tissues Lung --species Human --assay Xenium
```

All the information of splitting, metadata, and gene-gene colocalization should be stored in the .h5ad file.

## Step 3: Push to the dataset hub/ save to local disk

Once we got the processed "XX.h5ad", we can generate the dataset object and push to the huggingface dataset hub.

```bash
python h5toloader.py --data_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/processed/relabel_output-XETG00048__0003392__THD0008__20230313__191400.h5ad 
```

The processed huggingface array files will be stored in the processed directory as the entitle "XX_array" at the end of the file name.

## Step 4: Concate the dataset and push to the huggingface dataset hub

```bash
python merge_pandata_with_David.py
```


## Step 5: Getting the pair-wise information

WARNNING!!!
This is too large, you should build yourself according to the pair selection algorithm

```bash
python build_pair.py
```

the pair-wise dataset can be generated for each slide.



