Some parameters may be deprecated, we present all the parameters used in this model across diverse version of SpatialFormer.
| Parameter                  | Type    | Description                                                                 |
|----------------------------|---------|-----------------------------------------------------------------------------|
| accumulate_grad_batches    | integer | Number of batches to accumulate gradients before optimization step          |
| ag_loss                    | boolean | Whether using the attention guidance strategy to train the model            |
| assay                      | boolean | Whether assay information is incorporated into the model                    |
| autoregressive             | boolean | Whether the model uses autoregressive generation                            |
| batch_first                | boolean | If true, input tensors have batch dimension first (batch, seq, features)    |
| batch_size                 | integer | Number of samples processed per batch                                       |
| bpp                        | boolean | Whether applying the convolution layer to the attention bias                |
| bpp_scale                  | integer | Scaling factor for the attention bias                                       |
| cls_token                  | integer | Token ID representing the classification token                              |
| pad_token                  | integer | Token ID representing the padding position                                  |
| condition                  | boolean | Whether experimental condition information is included                      |
| context_length             | integer | Maximum sequence length for model input                                     |
| dim_model                  | integer | Dimensionality of the model's hidden layers                                 |
| directionality             | boolean | Whether directionality information is incorporated in adding the attention bias  |
| dropout                    | float   | Dropout probability for regularization                                      |
| embedding_path             | string  | File path to pre-trained spatial embeddings                                 |
| input_mode                 | string  | Input format for model (e.g., "pair" for paired inputs)                     |
| lambda                     | float   | percentage of head in the attention layer that should be weighted by the attention guidance |
| lr                         | float   | Learning rate for optimization                                              |
| mask_token                 | integer | Token ID used for masked positions                                          |
| mask_way                   | string  | Masking strategy used during training (e.g., "MT" for masked token)         |
| masking_p                  | float   | Probability of masking input tokens during training                         |
| max_epochs                 | integer | Maximum number of training epochs                                           |
| mini_batch                 | boolean | Whether mini-batch processing is enabled                                    |
| n_atokens                  | integer | Number of auxiliary tokens in vocabulary                                    |
| n_tasks                    | integer | Number of prediction tasks in the model                                     |
| n_tokens                   | integer | Number of primary tokens in vocabulary                                      |
| nheads                     | integer | Number of attention heads in multi-head attention                           |
| nlayers                    | integer | Number of transformer layers                                                |
| objective                  | string  | Training objective function (normalized exponential). This is deprecated in currect SpatialFormer version |
| organ                      | boolean | Whether organ metadata is utilized                                          |
| pool                       | null    | Pooling method (currently not used)                                         |
| pretrained_path            | string  | Path to pretrained model checkpoint for resuming training                   |
| retake_training            | boolean | Whether to resume training from checkpoint                                  |
| scale                      | integer | General scaling factor for attention guidance                               |
| sep_token                  | integer | Token ID representing separator token                                       |
| spatial_embedding          | boolean | Whether spatial embeddings are enabled                                      |
| spatial_embedding_freeze   | boolean | Whether spatial embeddings are frozen during training                       |
| specie                     | boolean | Whether species information is incorporated                                 |
| strategy                   | string  | Distributed training strategy (DDP = Distributed Data Parallel)             |
| supervised_task            | boolean | Whether supervised learning task is active. This is deprecated in currect SpatialFormer version |
| total_step                 | integer | Total number of training steps                                              |
| warmup                     | integer | Number of warmup steps for learning rate scheduling                         |
| weight_strategy            | string  | Strategy for weighting losses (e.g., "DW" for Dynamic weight)               |
