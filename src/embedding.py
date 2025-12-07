"""
ESM embedding utilities.

Core idea:
    - Load esm1b_t33_650M_UR50S once per process.
    - Convert sequences -> fixed-size 1280-d vectors (mean-pooled over residues).
    - Support batching + caching to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
import esm
from tqdm import tqdm


_ESM_MODEL = None
_ESM_ALPHABET = None
_BATCH_CONVERTER = None


def load_esm_model(device: Optional[str] = None):
    """
    Lazy-load ESM-1b model and alphabet.

    Parameters
    ----------
    device : {"cuda", "cpu", None}
        If None, auto-detect CUDA.
    """
    global _ESM_MODEL, _ESM_ALPHABET, _BATCH_CONVERTER

    if _ESM_MODEL is not None:
        return _ESM_MODEL, _ESM_ALPHABET, _BATCH_CONVERTER

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    model.to(device)

    batch_converter = alphabet.get_batch_converter()

    _ESM_MODEL = model
    _ESM_ALPHABET = alphabet
    _BATCH_CONVERTER = batch_converter

    return model, alphabet, batch_converter


def _mean_pool_representations(
    token_representations: torch.Tensor,
    tokens: torch.Tensor,
    padding_idx: int,
) -> torch.Tensor:
    """
    Mean-pool token representations over non-padding residues,
    excluding BOS/EOS tokens.
    """
    # tokens: (B, L)
    # token_representations: (B, L, D)
    with torch.no_grad():
        # mask of non-padding positions
        non_pad_mask = tokens != padding_idx
        # exclude BOS (index 0) and EOS (last non-pad)
        # we will compute lengths per sequence
        pooled = []
        for i in range(tokens.size(0)):
            row = token_representations[i]
            row_tokens = tokens[i]
            # indices of non-pad tokens
            non_pad_indices = (row_tokens != padding_idx).nonzero(as_tuple=True)[0]
            # usually: [0]=BOS, [1..L-2]=AA, [L-1]=EOS
            if len(non_pad_indices) <= 2:
                # degenerate case, just mean over all non-pad
                vec = row[non_pad_indices].mean(dim=0)
            else:
                start = non_pad_indices[1]
                end = non_pad_indices[-2] + 1  # slice is [start:end)
                vec = row[start:end].mean(dim=0)
            pooled.append(vec)

        pooled = torch.stack(pooled, dim=0)  # (B, D)
        return pooled


def embed_sequences(
    ids_and_seqs: Iterable[Tuple[str, str]],
    batch_size: int = 8,
    device: Optional[str] = None,
    progress: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute ESM-1b embeddings for a list of (id, sequence).

    Parameters
    ----------
    ids_and_seqs : iterable of (str, str)
        (identifier, amino-acid sequence).
    batch_size : int
        Number of sequences per batch.
    device : {"cuda", "cpu", None}
        Device for inference.
    progress : bool
        Show tqdm progress bar.

    Returns
    -------
    ids : list[str]
    embeddings : np.ndarray, shape (N, D=1280)
    """
    model, alphabet, batch_converter = load_esm_model(device=device)
    device = next(model.parameters()).device

    ids_and_seqs = list(ids_and_seqs)
    ids = [p[0] for p in ids_and_seqs]

    embeddings_list: List[np.ndarray] = []

    iterator = range(0, len(ids_and_seqs), batch_size)
    if progress:
        iterator = tqdm(iterator, desc="Embedding sequences")

    for start in iterator:
        batch = ids_and_seqs[start : start + batch_size]
        if not batch:
            continue

        data = [(seq_id, seq) for seq_id, seq in batch]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False,
            )
        token_reprs = results["representations"][33]  # (B, L, D)

        pooled = _mean_pool_representations(
            token_representations=token_reprs,
            tokens=batch_tokens,
            padding_idx=alphabet.padding_idx,
        )
        embeddings_list.append(pooled.cpu().numpy())

    embeddings = np.vstack(embeddings_list)
    return ids, embeddings


def cache_embeddings(
    df,
    id_col: str,
    seq_col: str,
    out_path: str | Path,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Path:
    """
    Compute embeddings for a DataFrame and save to .npz.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain id_col and seq_col.
    id_col, seq_col : str
    out_path : str or Path
        Where to save. Will create parent dirs if needed.

    Returns
    -------
    out_path : Path
    """
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids_and_seqs = list(zip(df[id_col].tolist(), df[seq_col].tolist()))
    ids, X = embed_sequences(ids_and_seqs, batch_size=batch_size, device=device)

    # verify order
    assert ids == df[id_col].tolist(), "Order mismatch between df and embeddings."

    np.savez_compressed(out_path, ids=np.array(ids, dtype=object), X=X)

    return out_path
