import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import urllib.parse
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code" / "business_entity_resolution"
SRC_DIR = CODE_DIR / "src"
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"
ARTIFACTS_DIR = CODE_DIR / "artifacts"

sys.path.insert(0, str(SRC_DIR))

try:
    from preprocessing import normalize_name, normalize_address, normalize_text
    from features import pair_features, FEATURE_NAMES
    from data_loader import load_test, load_training
    from utils import validate_basic_outputs
except Exception as e:
    print(f"Warning loading src modules: {e}")

class EntityResolutionHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/stats":
            self.handle_api_stats()
        elif path == "/api/entities":
            self.handle_api_entities()
        elif path == "/api/matching-results":
            self.handle_api_matching_results()
        elif path == "/api/validate":
            self.handle_api_validate()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b'{}'
        try:
            data = json.loads(post_body.decode('utf-8'))
        except Exception:
            data = {}

        if path == "/api/match-custom":
            self.handle_match_custom(data)
        elif path == "/api/adjust-threshold":
            self.handle_adjust_threshold(data)
        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, obj, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode('utf-8'))

    def handle_api_stats(self):
        config_path = ARTIFACTS_DIR / "config.json"
        cfg = {}
        if config_path.is_file():
            cfg = json.loads(config_path.read_text())

        s1_test = pd.read_csv(DATASET_DIR / "test" / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False) if (DATASET_DIR / "test" / "test_source1.tsv").is_file() else []
        s2_test = pd.read_csv(DATASET_DIR / "test" / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False) if (DATASET_DIR / "test" / "test_source2.tsv").is_file() else []
        s3_test = pd.read_csv(DATASET_DIR / "test" / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False) if (DATASET_DIR / "test" / "test_source3.tsv").is_file() else []

        match_df = pd.read_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False) if (OUTPUT_DIR / "matching_results.tsv").is_file() else pd.DataFrame()
        matched_count = 0
        singleton_count = 0
        if not match_df.empty:
            for _, r in match_df.iterrows():
                if r.get("matched_entity_ids", "").strip():
                    matched_count += 1
                else:
                    singleton_count += 1

        res = {
            "validation_f05": cfg.get("validation_f05", 1.0),
            "precision": cfg.get("precision", 1.0),
            "recall": cfg.get("recall", 1.0),
            "blocking_recall": cfg.get("blocking_recall", 1.0),
            "threshold": cfg.get("threshold", 0.05),
            "test_s1_count": len(s1_test),
            "test_s2_count": len(s2_test),
            "test_s3_count": len(s3_test),
            "predicted_matched": matched_count,
            "predicted_singletons": singleton_count,
        }
        self.send_json(res)

    def handle_api_entities(self):
        test_dir = DATASET_DIR / "test"
        s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False).to_dict(orient="records")
        s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False).to_dict(orient="records")
        s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False).to_dict(orient="records")
        self.send_json({"source1": s1, "source2": s2, "source3": s3})

    def handle_api_matching_results(self):
        match_p = OUTPUT_DIR / "matching_results.tsv"
        cand_p = OUTPUT_DIR / "candidate_pairs.tsv"
        
        matches = pd.read_csv(match_p, sep="\t", dtype=str, keep_default_na=False).to_dict(orient="records") if match_p.is_file() else []
        candidates = pd.read_csv(cand_p, sep="\t", dtype=str, keep_default_na=False).to_dict(orient="records") if cand_p.is_file() else []
        
        self.send_json({"matching_results": matches, "candidate_pairs": candidates})

    def handle_api_validate(self):
        import subprocess
        val_script = ROOT / "utils" / "validate_submission.py"
        cmd = [
            sys.executable, str(val_script),
            "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
            "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
            "--test-dir", str(DATASET_DIR / "test")
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        is_pass = "PASS" in res.stdout
        self.send_json({"status": "PASS" if is_pass else "FAIL", "output": res.stdout, "error": res.stderr})

    def handle_match_custom(self, data):
        name1 = data.get("name1", "")
        addr1 = data.get("address1", "")
        c1 = data.get("country1", "")

        name2 = data.get("name2", "")
        addr2 = data.get("address2", "")
        c2 = data.get("country2", "")

        a_row = {
            "business_name": name1, "business_address": addr1, "country": c1,
            "name_norm": normalize_name(name1), "address_norm": normalize_address(addr1), "country_norm": normalize_text(c1)
        }
        b_row = {
            "business_name": name2, "business_address": addr2, "country": c2,
            "name_norm": normalize_name(name2), "address_norm": normalize_address(addr2), "country_norm": normalize_text(c2)
        }

        feat = pair_features(a_row, b_row)
        feat_dict = dict(zip(FEATURE_NAMES, [float(v) for v in feat]))

        prob = 0.0
        model_p = ARTIFACTS_DIR / "model.joblib"
        if model_p.is_file():
            model = joblib.load(model_p)
            prob = float(model.predict_proba([feat])[0, 1])

        cfg_p = ARTIFACTS_DIR / "config.json"
        threshold = 0.05
        if cfg_p.is_file():
            threshold = float(json.loads(cfg_p.read_text()).get("threshold", 0.05))

        self.send_json({
            "probability": prob,
            "threshold": threshold,
            "is_match": prob >= threshold,
            "features": feat_dict,
            "entity1_norm": a_row,
            "entity2_norm": b_row
        })

    def handle_adjust_threshold(self, data):
        new_t = float(data.get("threshold", 0.05))
        # Re-eval predictions on test set with new threshold
        model_p = ARTIFACTS_DIR / "model.joblib"
        if not model_p.is_file():
            self.send_json({"error": "Model artifact missing"}, 400)
            return

        model = joblib.load(model_p)
        test_dir = DATASET_DIR / "test"
        s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
        s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
        s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)

        cand_p = OUTPUT_DIR / "candidate_pairs.tsv"
        cand_df = pd.read_csv(cand_p, sep="\t", dtype=str, keep_default_na=False)
        cand_map = dict(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]))

        target_dict = {r["entity_id"]: r for _, r in pd.concat([s2, s3], ignore_index=True).iterrows()}
        s1_dict = {r["entity_id"]: r for _, r in s1.iterrows()}

        for df in (s1_dict.values(), target_dict.values()):
            for r in df:
                r["name_norm"] = normalize_name(r["business_name"])
                r["address_norm"] = normalize_address(r["business_address"])
                r["country_norm"] = normalize_text(r["country"])

        matched_count = 0
        singleton_count = 0

        for sid, s1_row in s1_dict.items():
            raw_cids = cand_map.get(sid, "")
            cids = [x.strip() for x in raw_cids.split(",") if x.strip()] if raw_cids else []
            matches = []
            if cids:
                mat = np.asarray([pair_features(s1_row, target_dict[cid]) for cid in cids if cid in target_dict], dtype=np.float32)
                if len(mat) > 0:
                    probs = model.predict_proba(mat)[:, 1]
                    matches = [cid for cid, p in zip(cids, probs) if p >= new_t]
            if matches:
                matched_count += 1
            else:
                singleton_count += 1

        self.send_json({
            "threshold": new_t,
            "predicted_matched": matched_count,
            "predicted_singletons": singleton_count,
            "total_entities": len(s1)
        })

def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, EntityResolutionHandler)
    print(f"==================================================")
    print(f" Business Entity Resolution Server Online")
    print(f" Server URL: http://localhost:{port}")
    print(f"==================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)
