import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- Model Paths ----------
VLM_MODEL_PATH = os.environ.get("VLM_PATH", "/path/to/qwen2.5-vl-7b")
LORA_PATH = os.environ.get("LORA_PATH", "/path/to/lora/checkpoint")
BIOCLIP_MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"

# ---------- Data Paths ----------
TREE_JSON = os.environ.get("TREE_JSON", os.path.join(BASE_DIR, "data/full_animal_tree.json"))
CSV_DIR = os.environ.get("CSV_DIR", os.path.join(BASE_DIR, "data/layer_csv"))
RELIABILITY_CSV = os.environ.get("RELIABILITY_CSV", os.path.join(BASE_DIR, "data/bioclip_empirical_reliability.csv"))

# ---------- Inference Hyperparameters ----------
ATTR_TOP_K = int(os.environ.get("ATTR_TOP_K", "5"))
VISIBILITY_THRESHOLD = float(os.environ.get("VISIBILITY_THRESHOLD", "0.05"))
LAYER_WEIGHTS = [1.0, 1.3, 1.7, 2.2, 2.8, 3.5]  # L1 ~ L6
CONSISTENCY_BONUS = float(os.environ.get("CONSISTENCY_BONUS", "0.5"))

# ---------- Arbitration Thresholds ----------
BC_PRECISION_THRESH = float(os.environ.get("BC_PRECISION_THRESH", "0.85"))
WB_CONFIDENCE_THRESH = float(os.environ.get("WB_CONFIDENCE_THRESH", "0.30"))

# ---------- Hardware ----------
DEVICE = os.environ.get("DEVICE", "cuda:0")