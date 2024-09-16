#How to refresh the dataset and save to the remote huggingface dataset repository and local disk


!!!attention, please mount the erda before running the codes below to avoid the out of memory error

Step 1: Getting the distance by running the fing_gene_distance.py

For instance, run the code below for a certain sample
#for THD0008
```python
# python find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv
# python find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv
# python find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv
```
you will get three partitions for this samples. Because of the speed restriction, separating in this way can get the pair-wise distance matrix out faster.

Afterwards, you will find the generated ".h5" files in the same directory as the "Xtranscripts.csv".

Step 2: mapping the results to the completed huggingface dataset. Because we have already generate a dataset ready for usage, it is more convenience to map the new generated data to the dataset, after that, you can filter according to the cell_id and get the new subdataset with distance matrix embedded.

```python
python find_gene_distance.py --mode dataset_map --split 1
```
Where split is convenience for run in different shells to accelerate it.


Step 3: Integrate all the dataset into one, and the new version tag is attached by running
```python
python combine_to_update2.py
```
you will get the new complete dataset in the erda path and also submit it to the remote huggingface dataset repository