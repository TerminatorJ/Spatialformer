For the cell type/niche annotation task, we didn't implement them in LUMI cluster which is deployed AMD GPUs. Instead, we use the DIKU cluster, from the KU computer science department, to conduct the benchmarking comparison with scFoundation, scGPT, and Geneformer, since they are all developed base on the NVIDIA gpus.

Importantly, we only generate the embeddings of these methods from the same gpu environment, the main benckmarking work are conducted in the LUMI cluster. 


For scFoundation, the evaluation code can be found at: /home/sxr280/scFoundation/model/get_embedding.py

```python
# nohup python get_embedding.py --task_name David1M_0.1fra_val --input_type singlecell --output_type cell --pool_type all --tgthighres f1 --data_path /home/sxr280/Spatialformer/data/val_frac10.csv --save_path ./examples/enhancement/ --pre_normalized F --version rde > val.log &
# nohup python get_embedding.py --task_name David1M_0.1fra_train --input_type singlecell --output_type cell --pool_type all --tgthighres f1 --data_path /home/sxr280/Spatialformer/data/train_frac10.csv --save_path ./examples/enhancement/ --pre_normalized F --version rde > train.log &
# nohup python get_embedding.py --task_name David1M_0.1fra_test --input_type singlecell --output_type cell --pool_type all --tgthighres f1 --data_path /home/sxr280/Spatialformer/data/test_frac10.csv --save_path ./examples/enhancement/ --pre_normalized F --version rde > test.log &
```


For scGPT, the evaluation codes and results can be found at: /home/sxr280/scGPT/tutorials/zero-shot




