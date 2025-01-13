---

### How to download the original data in this project?

The data involve in this project include two main parts:

- 37 Publicly available Xenium dataset from 10X websites

- 25 Case data of pulmonary fibrosis from GEO. preprinted at (http://biorxiv.org/lookup/doi/10.1101/2023.12.15.571954)

We programmly download the data from the 10X data repository and GEO website by running the following scripts.

```shell
bash run1.sh && bach run2.sh && bach run3.sh && bach run4.sh
```

For all the pulmonary fibrosis dataset, please refer to the GEO dataset with accession number GSE250346. The GEO dataset comprises all the data from 25 slides, including H&E images, expression count matrices, and standard Xenium output files.

The overview of all the details of the original dataset used for SpatialFormer can be retrieved in Supplementary File 1 of the paper.





