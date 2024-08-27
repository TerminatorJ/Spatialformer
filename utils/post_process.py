#binning the dataset into the normalized expression level
#was token from scGPT: https://github.com/bowang-lab/scGPT/blob/main/scgpt/preprocess.py


import numpy as np
from typing import Dict, Optional, Union
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
import logger


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    import pdb; pdb.set_trace()
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize
    import pdb; pdb.set_trace()
    if row.max() == 0:
        logger.warning(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        import pdb; pdb.set_trace()
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)

def run_one_sample(raw_gene, exp, ranked_gene, method = "quantile"):
    '''
    This is only suitable for short sequence, if you have long sequence, please use bins
    '''
    if method == "quantile":
        # import pdb; pdb.set_trace()
        ranked_exp = np.array([exp[0][raw_gene.index(gene)] for gene in ranked_gene])
        unique_vals = np.unique(ranked_exp)
        quantiles = np.linspace(0.1, 1, len(unique_vals))
        value_to_quantile = dict(zip(unique_vals, quantiles))
        normalized_exp = np.array([value_to_quantile[exp] for exp in ranked_exp])
    elif method == "bin":
        normalized_exp = binning(exp, n_bins = 57)
    return normalized_exp


def bin_dataset(examples):
    # import pdb; pdb.set_trace()
    
    raw_genes = examples["Gene"]
    exps = examples["Expression"]
    ranked_genes = examples["Ranked_Gene_Names"]
    normalized_exps = [run_one_sample(raw_gene, exp, ranked_gene) for raw_gene,exp,ranked_gene in zip(raw_genes, exps, ranked_genes)]
    # import pdb; pdb.set_trace()
    
    output = {"Normalized_Exp": normalized_exps}
    return output





if __name__ == "__main__":
    #loading the dataset
    login(token="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")
    hf_cache = "/home/sxr280/Spatialformer/cache"
    combined_dataset = load_dataset("TerminatorJ/xenium_25_lung_dataset_update1", cache_dir = hf_cache)
    processed_datasets = combined_dataset.map(bin_dataset, batched = True, batch_size = 500)
    # combined_dataset.save_to_disk("/home/sxr280/Spatialformer/data/xenium_25_lung_dataset_update2")
    # import pdb; pdb.set_trace()
    # combined_dataset = load_from_disk("/home/sxr280/Spatialformer/data/xenium_25_lung_dataset_update2")
    processed_datasets.push_to_hub("xenium_25_lung_dataset_update2")
