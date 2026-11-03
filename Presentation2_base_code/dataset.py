"""
Dataset classes for MIL training
"""
import os
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from config import MODEL_CONFIG

class StainBagCasePooledFeatureDataset(Dataset):
    """
    Like StainBagCaseDataset, but loads precomputed pooled feature vectors (.pt)
    instead of loading images.

    For each CASE returns:
    {
        "case_id": int/str,
        "stain_slices": {
            "h&e":   [Tensor(P1, F4), Tensor(P2, F4), ...],
            ...
        },
        "label": LongTensor scalar
    }
    where F4 = 4*F (typically 4096 for DenseNet121 with AdaptiveAvgPool2d((2,2))).
    """

    def __init__(
        self,
        case_dict: Dict[Any, Dict[str, List[List[str]]]],
        label_map: Dict[Any, int],
        embeddings_dir: str,
        stains: Tuple[str, ...] = None,
        per_slice_cap: Optional[int] = None,
        max_slices_per_stain: Optional[int] = None,
        shuffle_patches: bool = True,
        drop_empty_slices: bool = True,
        # NEW: force a consistent dtype so fp16 embeddings won't crash fp32 model
        feature_dtype: torch.dtype = torch.float32,
    ):
        self.embeddings_dir = embeddings_dir
        self.stains = list(stains) if stains else list(MODEL_CONFIG["stains"])
        self.per_slice_cap = per_slice_cap if per_slice_cap is not None else MODEL_CONFIG["per_slice_cap"]
        self.max_slices_per_stain = (
            max_slices_per_stain if max_slices_per_stain is not None else MODEL_CONFIG["max_slices_per_stain"]
        )
        self.shuffle_patches = shuffle_patches
        self.drop_empty_slices = drop_empty_slices

        # NEW
        self.feature_dtype = feature_dtype

        self.items = []
        for case_id, stain_map in case_dict.items():
            if case_id not in label_map:
                raise KeyError(f"Missing label for case {case_id}")
            norm_map = {k.lower(): v for k, v in stain_map.items()}
            self.items.append((case_id, norm_map))
        self.label_map = label_map

    def __len__(self):
        return len(self.items)

    def _patch_path_to_feat_path(self, patch_path: str) -> str:
        # patch_path is full path like /projects/.../patches/<fname>.png
        fname = os.path.basename(patch_path)
        feat_name = os.path.splitext(fname)[0] + ".pt"
        return os.path.join(self.embeddings_dir, feat_name)

    def _load_slice_tensor(self, patch_paths: List[str]) -> Tuple[torch.Tensor, List[str]]:
        patch_paths = list(patch_paths)

        valid_paths = []
        for p in patch_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing patch: {p}")
            try:
                with Image.open(p) as img:
                    width, height = img.size
                    img.verify()
            except Exception as e:
                raise RuntimeError(f"Unreadable patch: {p}") from e
            if width >= 32 and height >= 32:
                valid_paths.append(p)

        patch_paths = valid_paths

        if self.shuffle_patches:
            random.shuffle(patch_paths)

        if self.per_slice_cap and len(patch_paths) > self.per_slice_cap:
            patch_paths = patch_paths[:self.per_slice_cap]

        vecs = []
        filenames = []
        for p in patch_paths:
            fp = self._patch_path_to_feat_path(p)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Missing embedding: {fp}")
            try:
                v = torch.load(fp, map_location="cpu")  # (4096,)
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding: {fp}") from e
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Expected tensor in {fp}, got {type(v)}")

            if v.shape != (4096,):
                raise ValueError(f"Invalid embedding shape: {fp}, got {v.shape}")

            if not torch.isfinite(v).all():
                raise ValueError(f"NaN or Inf in embedding: {fp}")

            v = v.to(self.feature_dtype)

            vecs.append(v)
            filenames.append(os.path.basename(p))

        if not vecs:
            raise ValueError("Slice has no patches")

        return torch.stack(vecs, dim=0), filenames  # (P, 4096) and list of filenames in the same order

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id, stain_map = self.items[idx]
        label = torch.tensor(self.label_map[case_id], dtype=torch.long)

        stain_slices: Dict[str, List[torch.Tensor]] = {}
        filenames = {}
        for stain in self.stains:
            slice_lists = stain_map.get(stain, []) or []

            if self.max_slices_per_stain is not None and len(slice_lists) > self.max_slices_per_stain:
                slice_lists = slice_lists[:self.max_slices_per_stain]

            tensors_for_stain = []
            filenames_for_stain = []
            for sl in slice_lists:
                if not sl:
                    raise ValueError(f"Empty slice: case {case_id}, stain {stain}")
                sl_tensor, sl_filenames = self._load_slice_tensor(sl)

                tensors_for_stain.append(sl_tensor)
                filenames_for_stain.append(sl_filenames)

            stain_slices[stain] = tensors_for_stain
            filenames[stain] = filenames_for_stain

        return {"case_id": case_id, "stain_slices": stain_slices, "patch_filenames": filenames, "label": label}


def case_collate_fn(batch):
    """
    Collate function for MIL dataset
    Keep variable-length structures; model.forward expects List[case_dict]
    Use batch_size=1 to avoid padding/masking
    """
    return batch