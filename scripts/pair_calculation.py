from datasets import load_from_disk
import os
import pandas as pd

rootdir = "/scratch/project_465001027/Spatialformer/cache"
allfiles = os.listdir(rootdir)

paths = [os.path.join(rootdir, file) for file in allfiles if "pair" in file]

names = [file for file in allfiles if "pair" in file]

stat_dict = {}
for i,name in enumerate(names):
    # import pdb; pdb.set_trace()
    dataset = load_from_disk(paths[i])
    num = len(dataset)
    stat_dict[name] = num
# import pdb; pdb.set_trace()
# df = pd.DataFrame(stat_dict, index=[0]) 
# Create a DataFrame using keys as the index and values as the values
df = pd.DataFrame.from_dict(stat_dict, orient='index', columns=['Value'])

df.to_csv("/scratch/project_465001027/Spatialformer/data/pair_stat.csv")


