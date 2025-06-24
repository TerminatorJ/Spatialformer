| Parameter                  | Type    | Description                                                                 |
|----------------------------|---------|-----------------------------------------------------------------------------|
| accumulate_grad_batches    | integer | Number of batches to accumulate gradients before optimization step          |
| assay                      | boolean | Whether assay information is incorporated into the model                    |
| autoregressive             | boolean | Whether the model uses autoregressive generation                            |
| batch_first                | boolean | If true, input tensors have batch dimension first (batch, seq, features)    |
| batch_size                 | integer | Number of samples processed per batch                                       |
| cls_token                  | integer | Token ID representing the classification token                              |
| condition                  | boolean | Whether experimental condition information is included                      |
| context_length             | integer | Maximum sequence length for model input                                     |
| dim_model                  | integer | Dimensionality of the model's hidden layers                                 |
| dropout                    | float   | Dropout probability for regularization                                      |
| embedding_path             | string  | File path to pre-trained spatial embeddings                                 |
| lr                         | float   | Learning rate for optimization                                              |
| mask_token                 | integer | Token ID used for masked positions                                          |
| masking_p                  | float   | Probability of masking input tokens during training                         |
| max_epochs                 | integer | Maximum number of training epochs                                           |
| mini_batch                 | boolean | Whether mini-batch processing is enabled                                    |
| n_atokens                  | integer | Number of auxiliary tokens in vocabulary                                    |
| n_tokens                   | integer | Number of primary tokens in vocabulary                                      |
| nheads                     | integer | Number of attention heads in multi-head attention                           |
| nlayers                    | integer | Number of transformer layers                                                |
| objective                  | string  | Training objective function (normalized exponential)                        |
| organ                      | boolean | Whether organ metadata is utilized                                          |
| pad_token                  | integer | Token ID used for padding                                                   |
| pool                       | null    | Pooling method (currently not used)                                         |
| sep_token                  | integer | Token ID representing separator token                                       |
| spatial_embedding          | boolean | Whether spatial embeddings are enabled                                      |
| spatial_embedding_freeze   | boolean | Whether spatial embeddings are frozen during training                       |
| specie                     | boolean | Whether species information is incorporated                                 |
| strategy                   | string  | Distributed training strategy (DDP = Distributed Data Parallel)             |
| supervised_task            | boolean | Whether supervised learning task is active                                  |
| total_step                 | integer | Total number of training steps                                              |
| warmup                     | integer | Number of warmup steps for learning rate scheduling                         |

