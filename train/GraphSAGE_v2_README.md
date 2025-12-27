# GraphSAGE v2 - Refactored for Spatial Transcriptomics

A cleaner, more modular implementation of GraphSAGE for spatial transcriptomics data, compatible with the data pipeline used in `find_gene_interaction.py`.

## Key Improvements

### 1. **Better Data Loading**
- ✅ Supports multiple formats: parquet directories, CSV, CSV.gz
- ✅ Uses same preprocessing as `find_gene_interaction.py`
- ✅ Memory-efficient with Dask for large datasets
- ✅ Automatic column name standardization

### 2. **Modular Architecture**
- `TranscriptDataLoader`: Handles all data loading and preprocessing
- `SpatialGraphBuilder`: Constructs spatial proximity graphs
- `GraphSAGEEncoder`: The core GraphSAGE model
- `GraphSAGELinkPrediction`: Self-supervised pretraining
- `GraphSAGEPatternClassifier`: Spatial pattern classification
- `GraphSAGETrainer`: Training pipeline management

### 3. **Cleaner Code**
- Clear separation of concerns
- Type hints throughout
- Comprehensive docstrings
- Better logging and progress tracking

## Usage

### Basic Training

```bash
python GraphSAGE_v2.py \
    --transcript_file /path/to/transcripts.csv \
    --gene_vocab /path/to/tokenv5.json \
    --output_dir ./output/graphsage
```

### Full Options

```bash
python GraphSAGE_v2.py \
    --transcript_files /scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Breast_Cancer_FFPE_xe_outs/transcript_processed /scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Skin_FFPE_xe_outs/transcript_processed /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_humanLung_Cancer_FFPE_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_hPancreas_nondiseased_section_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_hKidney_nondiseased_section_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_hHeart_nondiseased_section_FFPE_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_hColon_Non_diseased_Base_FFPE_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs/transcripts.csv.gz /scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Ovarian_Cancer_FFPE_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Human_Prostate_FFPE_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_Prime_Cervical_Cancer_FFPE_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_hLymphNode_nondiseased_section_outs/transcripts.parquet /scratch/project_465001820/Spatialformer/data/raw/Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs/transcripts.parquet \
    --save_path /home/sxr280/Spatialformer/data/saved_for_graphsave \
    --gene_vocab /home/sxr280/Spatialformer/tokenizer/tokenv5.json \
    --min_transcripts 30 \
    --max_cells 20000 \
    --radius 3.0 \
    --num_root_nodes 5000 \
    --num_hops 3 \
    --hidden_dim 256 \
    --output_dim 512 \
    --batch_size 512 \
    --lr 0.001 \
    --max_steps 50000 \
    --output_dir /home/sxr280/Spatialformer/output/graphsage \
    --save_graph \
    --use_wandb
```

## Input Data Formats

### 1. CSV/CSV.gz Files
```
x_location,y_location,z_location,feature_name,cell_id
123.4,567.8,0.5,GAPDH,cell_001
```

### 2. Parquet Directory
```
transcript_processed/
├── cell_chunk_1.parquet
├── cell_chunk_2.parquet
└── ...
```

The code automatically detects and handles both formats.

## Output

### Checkpoints
- Saved in `{output_dir}/checkpoints/link_prediction/`
- Format: `step=XXXXXXX-loss=X.XXXX-acc=X.XXXX.ckpt`

### Gene Embeddings
- Saved as `{output_dir}/gene_embeddings.pkl`
- Format: PyTorch tensor of shape `[num_tokens, embedding_dim]`
- Can be loaded with: `embeddings = pickle.load(open('gene_embeddings.pkl', 'rb'))`

### Graph Data (if --save_graph)
- Saved as `{output_dir}/processed_graph.pt`
- Format: PyTorch Geometric Data object
- Can be loaded with: `graph = torch.load('processed_graph.pt')`

## Architecture

### Data Flow

```
Transcript File (CSV/Parquet)
    ↓
TranscriptDataLoader
    ├── Filter genes (remove controls)
    ├── Filter cells (min transcripts)
    └── Sample cells (if max_cells set)
    ↓
SpatialGraphBuilder
    ├── One-hot encode genes
    ├── Build KDTree from coordinates
    └── Create edges within radius
    ↓
GraphSAGE Training
    ├── Link prediction (self-supervised)
    ├── 2-hop neighborhood sampling
    └── Binary classification loss
    ↓
Gene Embeddings
    └── Average transcript embeddings per gene
```

### Model Architecture

```
GraphSAGE Encoder:
    Input: [num_nodes, num_genes] (one-hot)
    ↓
    SAGEConv(num_genes → hidden_dim)
    ↓
    ReLU
    ↓
    SAGEConv(hidden_dim → output_dim)
    ↓
    Output: [num_nodes, output_dim]

Link Prediction:
    Positive edges: Existing spatial connections
    Negative edges: Random non-connected pairs
    Loss: Binary cross-entropy
```

## Comparison with Original Version

| Feature | Original GraphSAGE.py | GraphSAGE_v2.py |
|---------|---------------------|-----------------|
| Data loading | Mixed, hardcoded paths | Modular, flexible |
| File formats | Limited | CSV, CSV.gz, Parquet |
| Code organization | Monolithic | Modular classes |
| Documentation | Minimal | Comprehensive |
| Memory efficiency | OK | Better (Dask integration) |
| Configuration | JSON file | Command-line args |
| Logging | Basic | Detailed with progress bars |

## Common Use Cases

### 1. Process Single Sample
```bash
python GraphSAGE_v2.py \
    --transcript_file /data/sample1/transcripts.csv \
    --gene_vocab /path/to/tokenv5.json \
    --output_dir ./output/sample1
```

### 2. Process Large Dataset
```bash
python GraphSAGE_v2.py \
    --transcript_file /data/large_sample/transcript_processed \
    --gene_vocab /path/to/tokenv5.json \
    --max_cells 50000 \
    --batch_size 256 \
    --output_dir ./output/large_sample
```

### 3. Quick Test
```bash
python GraphSAGE_v2.py \
    --transcript_file /data/test.csv \
    --gene_vocab /path/to/tokenv5.json \
    --max_cells 5000 \
    --max_steps 5000 \
    --output_dir ./output/test
```

## Advanced Usage

### Using as a Library

```python
from GraphSAGE_v2 import TranscriptDataLoader, SpatialGraphBuilder, GraphSAGETrainer

# Load data
loader = TranscriptDataLoader(
    transcript_file="/path/to/transcripts.csv",
    gene_vocab_path="/path/to/tokenv5.json",
    min_transcripts_per_cell=30,
    max_cells=20000
)
df = loader.load_data()

# Build graph
builder = SpatialGraphBuilder(radius=3.0, is_3d=True)
graph = builder.build_graph(df, loader.gene_vocab)
subgraph = builder.create_subgraph(graph, num_root_nodes=5000)

# Train
config = {'hidden_dim': 256, 'output_dim': 512, 'lr': 1e-3}
trainer = GraphSAGETrainer(config, output_dir="./output")
model = trainer.train_link_prediction(subgraph, input_dim=len(loader.gene_vocab))

# Extract embeddings
trainer.extract_gene_embeddings(
    model, subgraph, loader.gene_vocab,
    "/path/to/tokenv5.json", "./gene_embeddings.pkl"
)
```

## Requirements

```
torch
torch-geometric
pytorch-lightning
dask[dataframe]
pandas
numpy
scipy
scikit-learn
networkx
torchmetrics
wandb (optional)
```

## Tips

1. **Memory Issues**: Reduce `--max_cells` or `--batch_size`
2. **Speed**: Increase `--batch_size` if you have GPU memory
3. **Quality**: Increase `--max_steps` for better embeddings
4. **Debugging**: Add `--save_graph` to inspect intermediate results

## Troubleshooting

### Out of Memory
- Reduce `--max_cells`
- Reduce `--batch_size`
- Reduce `--num_root_nodes`

### Too Slow
- Increase `--batch_size` (if memory allows)
- Reduce `--num_root_nodes`
- Use fewer cells with `--max_cells`

### Poor Results
- Increase `--max_steps`
- Adjust `--radius` (spatial threshold)
- Try different `--learning_rate`
