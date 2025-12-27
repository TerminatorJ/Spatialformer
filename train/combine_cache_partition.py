import pickle
from dataclasses import dataclass, field
import json
import hashlib
import numpy as np
@dataclass
class AnchorPairIndices:
    """Stores precomputed pair indices for a single anchor cell."""
    anchor_idx: int
    slide_name: str
    positive_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    hard_negative_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    easy_negative_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))



all_pairs = []
for i in range(40):
    try:
        paired_data = pickle.load(open(f"/scratch/project_465001820/Spatialformer/cache/cache_pairs_{i}.pkl", "rb"))
        all_pairs += paired_data["pairs"]
        total_cells = paired_data["total_cells"]
    except:
        print(f"The partition {i} not exists")
import pdb; pdb.set_trace()
data = {
    'path': "/scratch/project_465001820/Spatialformer/cache/xenium_5k_pandavid_dataset_v2",
    'positive_threshold': 30,
    'hard_negative_min': 50,
    'hard_negative_max': 150,
}

fingerprint = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
import pdb; pdb.set_trace()
cache_data = {
            'pairs': all_pairs,
            'fingerprint': fingerprint,
            'positive_threshold': 30,
            'hard_negative_min': 50,
            'hard_negative_max': 150,
            'total_cells': total_cells,
            'valid_anchors': len(all_pairs),
        }
import pdb; pdb.set_trace()
with open("/scratch/project_465001820/Spatialformer/cache/cache_pairs.pkl", 'wb') as f:
    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)