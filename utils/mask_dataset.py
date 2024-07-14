
from fairseq.data import (
    data_utils,
    Dictionary,
    iterators,
    FairseqDataset,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    MaskTokensDataset,
    PadDataset,
    BaseWrapperDataset,
    LRUCacheDataset,
)
from functools import lru_cache
from collections import namedtuple
import torch
import numpy as np
import os
from collections import Counter
# import pdb;pdb.set_trace()
from h5toloader import *
from pathlib import Path
import json
current_file_path = Path(__file__).resolve()
p_path = current_file_path.parents[1]
tokenizer_dir = os.path.join(p_path, "tokenizer")


class MaskTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.
    Input items are masked according to the specified masking probability.
    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, two_dim_score=True, two_dim_mask=-1)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        two_dim_score: bool = False,
        two_dim_mask: int = -1,
        mask_whole_words: torch.Tensor = None,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.two_dim_score = two_dim_score
        self.two_dim_mask = two_dim_mask
        self.mask_whole_words = mask_whole_words

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab.symbols))
            weights[: self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        # import pdb; pdb.set_trace()
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )
            # import pdb; pdb.set_trace()
            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))
            # import pdb; pdb.set_trace()
            # decide elements to mask
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz
                + np.random.rand()
            )
            # import pdb; pdb.set_trace()
            mask[np.random.choice(sz, num_mask, replace=False)] = True

            # return 2d-dim socre:
            if self.two_dim_score:
                item_len = len(item.numpy())
                two_dim_matrix = np.zeros((len(base_range_lst)*len(lamda_lst),item_len,item_len))
                padding_dim = 0
                for base_range in base_range_lst:
                    for lamda in lamda_lst:
                        new_matrix = creatmat(item.numpy(),base_range,lamda)
                        new_matrix[mask,:] = -1
                        new_matrix[:,mask] = -1
                        two_dim_matrix[padding_dim,:,:] = new_matrix
                        padding_dim += 1
                # use -1 represent mask
                # matrix[mask,:] = self.two_dim_mask
                # matrix[:,mask] = self.two_dim_mask
                # print(two_dim_matrix.shape)
                return torch.from_numpy(two_dim_matrix)
            # import pdb; pdb.set_trace()
            # return target
            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                if self.mask_whole_words is not None:
                    mask = np.repeat(mask, word_lens)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)
            # import pdb; pdb.set_trace()
            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None
            # import pdb; pdb.set_trace()
            if unmask is not None:
                mask = mask ^ unmask
            import pdb; pdb.set_trace()
            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)
            # import pdb; pdb.set_trace()
            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()
                    # import pdb; pdb.set_trace()
                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab.symbols),
                        num_rand,
                        p=self.weights,
                    )
            # import pdb; pdb.set_trace()
            return torch.from_numpy(new_item)



# A class to simulate the dataset structure (for demonstration purposes)
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_vocab(tokenized_datasets):
    Dictionary = namedtuple('Dictionary', ['symbols', 'count', 'nspecial'])
    with open(os.path.join(tokenizer_dir, "token.json"), 'r') as json_file:
        token_indice = json.load(json_file)
    full_gene_names = []
    for split in ["train", "test", "validation"]:
        full_gene_names.extend([gene_name for gene_list in tokenized_datasets[split]["Ranked_Gene_Names"] for gene_name in gene_list])
    total_counter = Counter(full_gene_names)
    sorted_genename = sorted(total_counter.keys())
    # Create token to index mapping
    sorted_count = [total_counter[gene_name] for idx, gene_name in enumerate(sorted_genename)]
    vocab = Dictionary(
            symbols=["<pad>", "<mask>"] +  sorted_genename,
            count=[0, 0] + sorted_count,
            nspecial=2  # <pad> and <mask> are special tokens
            )
    vocab_dict = vocab._asdict()

    # Save the dictionary to a JSON file
    with open(os.path.join(tokenizer_dir, "vocab_dict.json"), 'w') as f:
        json.dump(vocab_dict, f, indent=4)
    return vocab

        
if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # Define a simple dictionary with a few tokens
    # Dictionary = namedtuple('Dictionary', ['symbols', 'count', 'nspecial'])

    # vocab = Dictionary(
    #     symbols=["<pad>", "<mask>", "hello", "world", "this", "is", "a", "test"],
    #     count=[100, 100, 10, 10, 10, 10, 10, 10],
    #     nspecial=2  # <pad> and <mask> are special tokens
    # )
    # import pdb; pdb.set_trace()
    # Token indices for the vocabulary
    # token_indices = {token: idx for idx, token in enumerate(vocab.symbols)}

    # Define a simple dataset (list of tokenized sentences)
    # dataset = [
    #     torch.tensor([token_indices["hello"], token_indices["world"], token_indices["this"], token_indices["is"], token_indices["a"], token_indices["test"]]),
    #     torch.tensor([token_indices["this"], token_indices["is"], token_indices["a"], token_indices["test"]]),
    # ]
    
    data_path = "/scratch/project_465001027/spatialformer/data/processed/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.h5ad"
    tokenized_datasets = get_dataset(data_path)
    
    vocab = get_vocab(tokenized_datasets)
    
    # import pdb; pdb.set_trace()
    
    # Parameters for MaskTokensDataset
    pad_idx = 0
    mask_idx = 1
    mask_prob = 0.15  # Probability of masking a token
    leave_unmasked_prob = 0.1  # Probability of leaving a masked token unmasked
    random_token_prob = 0.1  # Probability of replacing a masked token with a random token
    
    
    
    test_dataset = tokenized_datasets["test"]["Full_Tokens"]
    # import pdb; pdb.set_trace()
    # Create MaskTokensDataset instance
    mask_dataset = MaskTokensDataset(
        dataset=test_dataset,
        vocab=vocab,
        pad_idx=pad_idx,
        mask_idx=mask_idx,
        return_masked_tokens=False,
        seed=1,
        mask_prob=mask_prob,
        leave_unmasked_prob=leave_unmasked_prob,
        random_token_prob=random_token_prob,
        freq_weighted_replacement=False
    )
    import pdb; pdb.set_trace()
    # input, output, _ = mask_dataset.apply_mask(mask_dataset)
    
    input_ds, output_ds, _ = MaskTokensDataset.apply_mask(test_dataset, vocab=vocab, pad_idx=pad_idx, mask_idx=mask_idx)
    # import pdb; pdb.set_trace()
    # Retrieve a masked item from the dataset
    index = 0
    # import pdb; pdb.set_trace()
    masked_item = mask_dataset[index]

    print("Original item:", test_dataset[index])
    print("Masked item:", masked_item)