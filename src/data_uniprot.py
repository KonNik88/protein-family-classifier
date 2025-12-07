import io
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

from .config import RAW_DIR, UNIPROT_URL, FAMILY_KEYWORDS, MIN_SEQ_LEN, MAX_SEQ_LEN


def fetch_family_from_uniprot(
    family_name: str,
    keyword_id: str,
    size: int = 500,
    min_len: int | None = None,
    max_len: int | None = None,
) -> pd.DataFrame:
    """
    Скачиваем N (size) белков для одного функционального класса из UniProtKB.
    Возвращаем DataFrame с колонками:
    [uniprot_id, protein_name, organism, length, sequence, family]
    """
    if min_len is None:
        min_len = MIN_SEQ_LEN
    if max_len is None:
        max_len = MAX_SEQ_LEN

    query = (
        f"(reviewed:true) "
        f"AND (keyword:{keyword_id}) "
        f"AND (length:[{min_len} TO {max_len}])"
    )

    params = {
        "query": query,
        "format": "tsv",
        "fields": ",".join([
            "accession",
            "id",
            "protein_name",
            "organism_name",
            "length",
            "sequence",
            "keyword",
        ]),
        "size": size,
    }

    headers = {"Accept": "text/tab-separated-values"}

    print(f"[{family_name}] Requesting {size} entries from UniProt...")
    resp = requests.get(UNIPROT_URL, params=params, headers=headers)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), sep="\t")

    if df.empty:
        print(f"[{family_name}] WARNING: empty result!")
        return df

    df = df.rename(columns={
        "Entry": "uniprot_id",
        "Entry Name": "uniprot_entry_name",
        "Protein names": "protein_name",
        "Organism": "organism",
        "Length": "length",
        "Sequence": "sequence",
        "Keywords": "keywords",
    }, errors="ignore")

    for col in ["uniprot_id", "protein_name", "organism", "length", "sequence"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in UniProt response. "
                f"Got columns: {df.columns.tolist()}"
            )

    df["family"] = family_name

    df = df.drop_duplicates(subset=["uniprot_id"])
    df = df.drop_duplicates(subset=["sequence"])

    print(f"[{family_name}] Got {len(df)} unique sequences.")
    return df[["uniprot_id", "protein_name", "organism", "length", "sequence", "family"]]


def build_protein_family_dataset(
    family_keywords: Dict[str, str] | None = None,
    n_per_class: int = 500,
    out_csv: Path | None = None,
    out_fasta: Path | None = None,
    min_len: int | None = None,
    max_len: int | None = None,
) -> pd.DataFrame:
    """
    Собираем датасет из нескольких функциональных классов.
    """
    if family_keywords is None:
        family_keywords = FAMILY_KEYWORDS

    if out_csv is None:
        out_csv = RAW_DIR / "protein_families_small.csv"
    if out_fasta is None:
        out_fasta = RAW_DIR / "raw_sequences_small.fasta"

    all_dfs = []

    for family, kw_id in family_keywords.items():
        df_family = fetch_family_from_uniprot(
            family_name=family,
            keyword_id=kw_id,
            size=n_per_class,
            min_len=min_len,
            max_len=max_len,
        )
        if df_family.empty:
            print(f"[{family}] Skipping (no data).")
            continue
        all_dfs.append(df_family)

    if not all_dfs:
        raise RuntimeError("No data downloaded from UniProt. Check queries / keywords.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    df_all = df_all.drop_duplicates(subset=["uniprot_id"])
    df_all = df_all.drop_duplicates(subset=["sequence"])

    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv} (n={len(df_all)})")

    with open(out_fasta, "w") as f:
        for row in df_all.itertuples():
            header = f">{row.uniprot_id}|{row.family}|{row.organism}".replace(" ", "_")
            f.write(header + "\n")
            f.write(row.sequence + "\n")

    print(f"Saved FASTA: {out_fasta}")

    print("\nClass distribution:")
    print(df_all["family"].value_counts())

    return df_all


if __name__ == "__main__":
    build_protein_family_dataset(
        n_per_class=500,
    )
