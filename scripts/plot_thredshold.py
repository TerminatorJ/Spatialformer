import os 
import sys 
import h5py
import pandas as pd
import numpy as np
import multiprocessing
import argparse
from pathlib import Path
current_file_path = Path(os.getcwd()).resolve()
p_path = current_file_path.parents[0]
sys.path.append("p_path")
from process import KNN_Radius_Graph
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
'''

lung_annot_3D_tx = pd.read_csv("/scratch/project_465001027/nicheformer/src/nicheformer/data/raw/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/transcripts.csv")
lung_annot_3D_tx.rename(columns={'x_location': 'x', 'y_location':'y', 'z_location':'z', 'feature_name':'gene'}, inplace=True)

#filtering the weired genes
lung_annot_3D_tx = lung_annot_3D_tx[~(lung_annot_3D_tx['gene'].str.startswith('Neg') | lung_annot_3D_tx['gene'].str.startswith('BLANK'))]
#filtering the transcripts
value_counts = lung_annot_3D_tx['cell_id'].value_counts()
clean_value_counts = value_counts.drop("UNASSIGNED")


# Set the bin size
bin_size = 10

# Bin the data
binned_counts = pd.cut(clean_value_counts, bins=range(0, clean_value_counts.max() + bin_size, bin_size), right=False).value_counts().sort_index()

# Convert bins to string representation for the x-axis
binned_counts.index = [f"{int(interval.left)}-{int(interval.right-1)}" for interval in binned_counts.index]

sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(14, 7))
sns.barplot(x=binned_counts.index, y=binned_counts.values, palette="viridis")

# Add labels and title
plt.xlabel('Value Count Bins', fontsize=14)
plt.ylabel('Number of Cells', fontsize=14)
plt.title('Distribution of Cell Counts', fontsize=16)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/scratch/project_465001027/spatialformer/figure/lung_transcript_distribution.png", dpi = 300)

'''


#plot the abundance threshold
import pickle

density_thredshold = pickle.load(open("/scratch/project_465001027/spatialformer/data/density_thredshold.pkl", "rb"))
import pdb; pdb.set_trace()


data_flattened = []
for item in density_thredshold:
    for key, value in item.items():
        x, y = map(int, key.split('_'))
        data_flattened.append({'x': x, 'y': y, 'frequency': value})
import pdb; pdb.set_trace()
df = pd.DataFrame(data_flattened)
    

# Create a new column for x_y combinations
df['x_y'] = df['x'].astype(str) + '_' + df['y'].astype(str)
mean_values = df.groupby(['x', 'y'])['frequency'].mean().reset_index()
# Create a new column for x_y combinations
mean_values['x_y'] = mean_values['x'].astype(str) + '_' + mean_values['y'].astype(str)

# Plotting the mean values
plt.figure(figsize=(12, 6))
plt.bar(mean_values['x_y'], mean_values['frequency'], color='skyblue')

plt.xlabel('x_y Combinations')
plt.ylabel('Mean Value')
plt.title('Mean Values of x_y Combinations')
# Rotate x labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/scratch/project_465001027/spatialformer/figure/transcript_abundance_gene_number.png", dpi = 300)
plt.show()