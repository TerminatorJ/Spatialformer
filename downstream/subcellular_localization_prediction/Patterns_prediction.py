# Add the path to the system path list
import sys
from pathlib import Path
# Get the current working directory
current_dir = Path.cwd()

# Assuming your notebook is in a subdirectory of the project, adjust as needed
project_root = current_dir.parent.parent

# Add the project root to sys.path
sys.path.append(str(project_root))
import spatialformer as sp

dataloader = sp.pp.build_graph(data_path = "/scratch/project_465001027/Spatialformer/downstream/subcellular_localization_prediction/data/transcripts.csv",
                 vocab_path = "/scratch/project_465001027/Spatialformer/spatialformer/tokenizer/tokenv3.json",
                 batch_size = 32,
                 graph_path = "/scratch/project_465001027/Spatialformer/downstream/subcellular_localization_prediction/data",
                 threshold = 10,
                 split = True)
train_dataloader, val_dataloader, test_dataloader = dataloader

model = sp.pp.load_pretrained_model(
                            vocab_path = "/scratch/project_465001027/Spatialformer/spatialformer/tokenizer/tokenv3.json",
                            hidden_dim = 256,
                            output_dim = 512,
                            batch_size = 32,
                            checkpoint = "/scratch/project_465001027/Spatialformer/output/GraphSAGE_model/checkpoints/step=0010000-train_loss=0.3983-val_loss=0.0000-train_acc=0.8256.ckpt",
                            device = "cuda")

sp.pp.train_pattern_model(
                        model,
                        lr = 0.001,
                        output_dim = 512,
                        strategy = "ddp",
                        output_dir = "/scratch/project_465001027/Spatialformer/output/GraphSAGE_model/checkpoints",
                        train_dataloader = train_dataloader,
                        val_dataloader = val_dataloader)