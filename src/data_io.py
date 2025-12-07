"""
Utilities for loading and preprocessing protein sequence data.

Expected labeled CSV format:
    uniprot_id,sequence,label
    P31749,MSDVEG...,kinase
    Q9Y6K9,MNKHLL...,receptor
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from Bio import SeqIO


def load_labeled_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a labeled CSV with columns: uniprot_id, sequence, label.

    Returns
    -------
    df : pd.DataFrame
        Standardized columns: ["uniprot_id", "sequence", "label"].
    """
    path = Path(path)
    df = pd.read_csv(path)

    # normalize column names
    col_map = {c.lower(): c for c in df.columns}
    # simple robustness: allow UNIPROT, id, etc.
    id_col = next((c for c in df.columns if c.lower() in ("uniprot_id", "id", "accession")), None)
    seq_col = next((c for c in df.columns if c.lower() in ("sequence", "seq", "aa_sequence")), None)
    label_col = next((c for c in df.columns if c.lower() in ("label", "family", "class")), None)

    if id_col is None or seq_col is None or label_col is None:
        raise ValueError(
            f"CSV {path} must contain identifier, sequence and label columns. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            id_col: "uniprot_id",
            seq_col: "sequence",
            label_col: "label",
        }
    )

    # basic cleaning
    df["sequence"] = df["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    df = df.dropna(subset=["sequence", "label"]).reset_index(drop=True)

    return df


def load_fasta(path: str | Path, id_prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Load sequences from a FASTA file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to FASTA.
    id_prefix : str, optional
        Optional prefix to add to sequence IDs.

    Returns
    -------
    df : pd.DataFrame
        Columns: ["uniprot_id", "sequence"].
    """
    path = Path(path)
    records = list(SeqIO.parse(str(path), "fasta"))

    data: List[Tuple[str, str]] = []
    for i, rec in enumerate(records):
        rec_id = rec.id or f"seq_{i}"
        if id_prefix:
            rec_id = f"{id_prefix}_{rec_id}"
        seq = str(rec.seq).replace("\n", "").upper()
        data.append((rec_id, seq))

    df = pd.DataFrame(data, columns=["uniprot_id", "sequence"])
    return df


def train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split by label (if available).

    Returns
    -------
    df_train, df_val, df_test
    """
    from sklearn.model_selection import train_test_split

    if "label" not in df.columns:
        raise ValueError("DataFrame must have a 'label' column for supervised split.")

    y = df["label"]
    strat = y if stratify else None

    df_train, df_tmp = train_test_split(
        df, test_size=val_size + test_size, random_state=random_state, stratify=strat
    )

    # recompute strat for tmp split
    y_tmp = df_tmp["label"]
    strat_tmp = y_tmp if stratify else None
    relative_test_size = test_size / (val_size + test_size)

    df_val, df_test = train_test_split(
        df_tmp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=strat_tmp,
    )

    for part in (df_train, df_val, df_test):
        part.reset_index(drop=True, inplace=True)

    return df_train, df_val, df_test
