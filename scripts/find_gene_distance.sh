
# Running jobs for THD0008
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990532_output-XETG00048__0003392__THD0008__20230313__191400_transcripts.csv


# Running job for TILD117MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990534_output-XETG00048__0003400__TILD117MF__20230313__191400_transcripts.csv
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990533_output-XETG00048__0003400__THD0011__20230313__191400_transcripts.csv
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990534_output-XETG00048__0003400__TILD117MF__20230313__191400_transcripts.csv
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990534_output-XETG00048__0003400__TILD117MF__20230313__191400_transcripts.csv

# Running jobs for TILD117LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990535_output-XETG00048__0003400__TILD117LF__20230313__191400_transcripts.csv
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990535_output-XETG00048__0003400__TILD117LF__20230313__191400_transcripts.csv


#THD0011
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990533_output-XETG00048__0003400__THD0011__20230313__191400_transcripts.csv

#TILD175
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990536_output-XETG00048__0003400__TILD175__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990536_output-XETG00048__0003400__TILD175__20230313__191400_transcripts.csv

#VUHD069
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990537_output-XETG00048__0003789__VUHD069__20230308__003731_transcripts.csv


# VUHD095
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990538_output-XETG00048__0003789__VUHD095__20230308__003731_transcripts.csv


# VUHD113
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990539_output-XETG00048__0003789__VUHD113__20230308__003731_transcripts.csv

# VUHD116A
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990540_output-XETG00048__0003817__VUHD116A__20230308__003730_transcripts.csv

# VUHD116B
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990541_output-XETG00048__0003817__VUHD116B__20230308__003731_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990541_output-XETG00048__0003817__VUHD116B__20230308__003731_transcripts.csv

# VUILD102MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990542_output-XETG00048__0003817__VUILD102MF__20230308__003730_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990542_output-XETG00048__0003817__VUILD102MF__20230308__003730_transcripts.csv

# VUILD102LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990543_output-XETG00048__0003817__VUILD102LF__20230308__003731_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990543_output-XETG00048__0003817__VUILD102LF__20230308__003731_transcripts.csv

# VUILD104MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990544_output-XETG00048__0003789__VUILD104MF__20230308__003731_transcripts.csv

# VUILD104LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990545_output-XETG00048__0003789__VUILD104LF__20230308__003731_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990545_output-XETG00048__0003789__VUILD104LF__20230308__003731_transcripts.csv

# VUILD105MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990546_output-XETG00048__0003789__VUILD105MF__20230308__003731_transcripts.csv

# VUILD105LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990547_output-XETG00048__0003789__VUILD105LF__20230308__003731_transcripts.csv

# VUILD106
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 5 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 6 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990548_output-XETG00048__0003392__VUILD106__20230313__191400_transcripts.csv

# VUILD107MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990549_output-XETG00048__0003817__VUILD107MF__20230308__003731_transcripts.csv

# VUILD110
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 5 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 6 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990550_output-XETG00048__0003392__VUILD110__20230313__191400_transcripts.csv

# VUILD115
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 4 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990551_output-XETG00048__0003392__VUILD115__20230313__191400_transcripts.csv

# VUILD48MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990552_output-XETG00048__0003789__VUILD48MF__20230308__003731_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990552_output-XETG00048__0003789__VUILD48MF__20230308__003731_transcripts.csv

# VUILD48LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990553_output-XETG00048__0003789__VUILD48LF__20230308__003731_transcripts.csv

# VUILD78MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990554_output-XETG00048__0003400__VUILD78MF__20230313__191400_transcripts.csv

# VUILD78LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990555_output-XETG00048__0003400__VUILD78LF__20230313__191400_transcripts.csv

# VUILD91MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990556_output-XETG00048__0003400__VUILD91MF__20230313__191400_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990556_output-XETG00048__0003400__VUILD91MF__20230313__191400_transcripts.csv

# VUILD91LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990557_output-XETG00048__0003400__VUILD91LF__20230313__191400_transcripts.csv

# VUILD96MF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990558_output-XETG00048__0003817__VUILD96MF__20230308__003730_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990558_output-XETG00048__0003817__VUILD96MF__20230308__003730_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990558_output-XETG00048__0003817__VUILD96MF__20230308__003730_transcripts.csv

# VUILD96LF
srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 1 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990559_output-XETG00048__0003817__VUILD96LF__20230308__003730_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 2 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990559_output-XETG00048__0003817__VUILD96LF__20230308__003730_transcripts.csv

srun --gres=gpu:1 ~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/Spatialformer/scripts/find_gene_distance.py --partition 3 --dataset_path TerminatorJ/xenium_25_lung_dataset_update2 --transcript_file /home/sxr280/Spatialformer/david_data/GSM7990559_output-XETG00048__0003817__VUILD96LF__20230308__003730_transcripts.csv

