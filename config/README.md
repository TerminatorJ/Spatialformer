"pretrained_path": The pretrained path of the checkpoint
"retake_training": The path of the checkpoint if you want to continuesely train the model
"bpp": The pair-wise matic which indicated the gene-gene co-occurence
"bpp_scale": The scale factor the makes the bpp matrix capturable to the attention machanism
"scale": Scale of the attention guiding loss, the higher the scale we set, the small model we train
"lambda": Proportion of the attention heads that can be guided in the attention machenism
"ag_loss": Whether to use the attention guiding loss