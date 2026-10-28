from pathlib import Path

# Path resolve: parent x3 from code/business_entity_resolution/src/config.py is root
ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset"
TRAIN = DATASET / "train"
TEST = DATASET / "test"
OUTPUT = ROOT / "output"
ARTIFACTS = ROOT / "code" / "business_entity_resolution" / "artifacts"

RANDOM_STATE = 42
TOP_K_PER_BLOCK = 50
DEFAULT_THRESHOLD = 0.50

for p in (OUTPUT, ARTIFACTS):
    p.mkdir(parents=True, exist_ok=True)
