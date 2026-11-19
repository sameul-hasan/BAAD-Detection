"""Central paths and constants for the BAAD KWS project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # .../BAAD/baad_kws
WORKSPACE = PROJECT_ROOT.parent                              # .../BAAD
DATASET_ROOT = WORKSPACE / "BAAD A Multipurpose Dataset for Automatic Bangla Offensive Speech Recognition"
CLIPS_ROOT = DATASET_ROOT / "Bangla Abusive Audio Dataset (BAAD)"
ABUSIVE_DIR = CLIPS_ROOT / "BAAD"
CLOSE_DIR = CLIPS_ROOT / "Abusive Close Words"
RAW_DIR = DATASET_ROOT / "Raw Data"
METADATA_DIR = DATASET_ROOT / "Metadata"

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST = DATA_DIR / "manifest.csv"
LABELS_JSON = DATA_DIR / "labels.json"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
EXTERNAL_DIR = DATA_DIR / "external"
SYNTH_DIR = DATA_DIR / "synth"

LOGS_DIR = PROJECT_ROOT / "logs"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SAMPLE_RATE = 16000
WINDOW_SEC = 1.25
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)   # 20,000 samples
HOP_SEC = 0.10
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SEC)         # 1,600 samples
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
