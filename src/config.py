from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(r"D:\ML\BioML\ESM")

# Пути
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STRUCTURES_DIR = DATA_DIR / "structures"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMB_DIR = ARTIFACTS_DIR / "embeddings"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIG_DIR = ARTIFACTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"

# Глобальные параметры
MAX_SEQ_LEN = 1000   # ограничение длины белка для ESM
MIN_SEQ_LEN = 50

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"

# Маппинг семейств -> UniProt keyword
FAMILY_KEYWORDS = {
    "kinase": "KW-0418",
    "transporter": "KW-0813",
    "ion_channel": "KW-0407",
    "transcription": "KW-0804",
    "chaperone": "KW-0143",
    "receptor": "KW-0675",  # Receptor :contentReference[oaicite:0]{index=0}
    "hydrolase": "KW-0378",  # Hydrolase :contentReference[oaicite:1]{index=1}
    "ligase": "KW-0436",  # Ligase :contentReference[oaicite:2]{index=2}
    "dna_binding": "KW-0238",  # DNA-binding proteins :contentReference[oaicite:3]{index=3}
    "protease": "KW-0645"
}

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, STRUCTURES_DIR,
          ARTIFACTS_DIR, EMB_DIR, MODELS_DIR, FIG_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
