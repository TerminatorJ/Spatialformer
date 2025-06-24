---
# Preprocessing the Dataset

After acquiring the original dataset, several preprocessing steps are required to ensure data quality and suitability for subsequent analyses.

## Preprocessing Settings

- **Minimum Transcript Count**: A cell must have more than 30 transcripts to be included in the analysis.
- **Filtering**: Remove "unassigned" cells and any genes labeled as "negative" or "blank."
- **Transcript Co-occurrence**: Adjacent transcripts should be identified when both the cell in question and the nearest neighboring genes each have more than 3 transcripts.
- **Quality Score(optional)**: A quality score for each transcript should be greater than 20.

## Step 1: Find the Gene-Gene Co-occurrence

You can run the scripts in this directory to generate the gene-gene co-occurrence matrix.


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

To consolidate all information into a single object, run the `build_h5ad.py` script in the current directory. Define sample status and assay types in the anndata object using the following command:

```bash
python build_h5ad.py --partitions 6 --data_name relabel_output-XETG00048__0003392__VUILD106__20230313__191400 --matrix_name VUILD106_gene_interaction --condition Disease --tissues Lung --species Human --assay Xenium
```

All the information of splitting, metadata, and gene-gene colocalization should be stored in the .h5ad file.

## Arguments for `build_h5ad.py`

| Argument                 | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `--partition`         | The number of chunks that has been divided in the step 1                                         |
| `--data_name`         | Overall dataset name (default: None)                                                           |
|`--matrix_name`        | The name of the matrix you got from the step 1                                                   |
|`--condition`          | The metadata of the condition for the samples. it could be Disease or Healthy|
|`--tissues`            | The metadata of the tissue name for the samples. for example, Lung or Breast|
|`--species`            | The metadata of the species name for the sample. for example, Human or Mouse|
|`--assay`              | The metadata of the assay that generate the data, for example, Xenium or MERFISH|
## Step 3: Push to the dataset hub or save to local disk

Once we got the processed "XX.h5ad" from the step 3, we can generate the dataset object and push to the huggingface dataset hub.

```bash
python h5toloader.py --data_path /scratch/project_465001027/spatialformer/david_data/relabel_output-XETG00048__0003392__THD0008__20230313__191400/processed/relabel_output-XETG00048__0003392__THD0008__20230313__191400.h5ad 
```
| Argument                 | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `--data_path`         | The path of the processed h5ad that is generated from the step 3                              |


The processed huggingface array files will be stored in the processed directory as the entitle "XX_array" at the end of the file name.

## Step 4: Concate the dataset and push to the huggingface dataset hub

Here, we merge the dataset from different source, combining the dataset from 10X data repository with the Lung Xenium dataset. All these datasets are combined and push to the huggingface as the huggingface dataset.

```bash
python merge_pandata_with_David.py
```


## Step 5: Getting the pair-wise information


Afterwards, the pair-wise cell pairs should be generate to pretrain the model.

WARNNING!!!
This is too large, you should build yourself according to the pair selection algorithm by the "build_pair.py". Each generated file could be > 20G.

```bash
python build_pair.py
```
For single sample example, please refer to the jupyter notebook as: build_pair.ipynb

Finally, the pair-wise dataset can be generated for each slide.



