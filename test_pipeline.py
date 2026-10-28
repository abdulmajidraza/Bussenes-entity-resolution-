import sys
import unittest
from pathlib import Path
import pandas as pd

# Add src to path
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
ROOT_DIR = HERE.parents[2]

sys.path.insert(0, str(SRC_DIR))

from data_loader import load_source, load_test, load_training
from preprocessing import normalize_name, normalize_address, normalize_text
from blocking import generate_candidates
from config import TRAIN, TEST, OUTPUT

class TestEntityResolutionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure sample dataset exists before testing
        if not (TRAIN / "train_source1.tsv").is_file():
            gen_script = ROOT_DIR / "dataset" / "generate_dataset.py"
            import subprocess
            subprocess.run([sys.executable, str(gen_script)], check=True)

    def test_01_tsv_loading(self):
        """Test 1: TSV loading with tab separator."""
        df = load_source(TRAIN / "train_source1.tsv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_02_required_columns(self):
        """Test 2: Required columns present in all source files."""
        for path in [
            TRAIN / "train_source1.tsv",
            TRAIN / "train_source2.tsv",
            TRAIN / "train_source3.tsv",
            TEST / "test_source1.tsv",
            TEST / "test_source2.tsv",
            TEST / "test_source3.tsv",
        ]:
            df = load_source(path)
            for col in ["entity_id", "business_name", "business_address", "country"]:
                self.assertIn(col, df.columns, f"Column {col} missing in {path}")

    def test_03_normalization(self):
        """Test 3: Normalization rules for names and addresses."""
        norm1 = normalize_name("ABC Pvt. Ltd.")
        norm2 = normalize_name("ABC PRIVATE LIMITED")
        norm3 = normalize_name("A.B.C. Pvt Ltd")
        
        self.assertEqual(norm1, norm2)
        self.assertIn("abc", norm3)
        self.assertIn("pvt", norm1)

        addr1 = normalize_address("100 Main Street, Suite 400")
        addr2 = normalize_address("100 MAIN ST STE 400")
        self.assertIn("st", addr1)
        self.assertIn("ste", addr1)
        self.assertIn("st", addr2)

    def test_04_candidate_generation_prefixes(self):
        """Test 4: Candidate generation returns only S2- and S3- IDs."""
        s1, s2, s3 = load_test(TEST)
        from preprocessing import enrich
        s1, s2, s3 = map(enrich, (s1, s2, s3))
        candidates, _ = generate_candidates(s1, s2, s3, top_k=10)

        for _, c_ids in candidates:
            for cid in c_ids:
                self.assertTrue(
                    cid.startswith("S2-") or cid.startswith("S3-"),
                    f"Candidate ID {cid} has invalid prefix."
                )

    def test_05_no_invalid_matches(self):
        """Test 5: Verify no S1 ID is predicted as a match."""
        cand_path = OUTPUT / "candidate_pairs.tsv"
        match_path = OUTPUT / "matching_results.tsv"

        if match_path.is_file():
            df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
            for _, r in df.iterrows():
                raw = r["matched_entity_ids"]
                ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
                for m in ids:
                    self.assertFalse(m.startswith("S1-"), f"S1 ID predicted as match: {m}")

    def test_06_output_completeness(self):
        """Test 6: Verify every Source 1 test entity has exactly one row in output."""
        s1 = load_source(TEST / "test_source1.tsv")
        match_path = OUTPUT / "matching_results.tsv"
        cand_path = OUTPUT / "candidate_pairs.tsv"

        if match_path.is_file() and cand_path.is_file():
            m_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
            c_df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
            
            self.assertEqual(len(m_df), len(s1))
            self.assertEqual(len(c_df), len(s1))
            self.assertEqual(set(m_df["source1_entity_id"]), set(s1["entity_id"]))
            self.assertEqual(set(c_df["source1_entity_id"]), set(s1["entity_id"]))

    def test_07_no_duplicate_ids(self):
        """Test 7: No duplicate IDs in any match list or candidate list."""
        match_path = OUTPUT / "matching_results.tsv"
        cand_path = OUTPUT / "candidate_pairs.tsv"

        if match_path.is_file() and cand_path.is_file():
            m_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
            c_df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)

            for _, r in m_df.iterrows():
                raw = r["matched_entity_ids"]
                ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
                self.assertEqual(len(ids), len(set(ids)), f"Duplicate match IDs: {ids}")

            for _, r in c_df.iterrows():
                raw = r["candidate_entity_ids"]
                ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
                self.assertEqual(len(ids), len(set(ids)), f"Duplicate candidate IDs: {ids}")

    def test_08_candidate_consistency(self):
        """Test 8: Every predicted match MUST appear in candidate_pairs.tsv."""
        match_path = OUTPUT / "matching_results.tsv"
        cand_path = OUTPUT / "candidate_pairs.tsv"

        if match_path.is_file() and cand_path.is_file():
            m_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
            c_df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
            c_map = dict(zip(c_df["source1_entity_id"], c_df["candidate_entity_ids"]))

            for _, r in m_df.iterrows():
                sid = r["source1_entity_id"]
                m_raw = r["matched_entity_ids"]
                matches = set(x.strip() for x in m_raw.split(",") if x.strip()) if m_raw else set()
                
                c_raw = c_map.get(sid, "")
                cands = set(x.strip() for x in c_raw.split(",") if x.strip()) if c_raw else set()
                
                self.assertTrue(matches.issubset(cands), f"Matches {matches - cands} not in candidate set for {sid}")

    def test_09_test_ids(self):
        """Test 9: Every predicted S2/S3 ID must exist in test dataset."""
        s2 = load_source(TEST / "test_source2.tsv")
        s3 = load_source(TEST / "test_source3.tsv")
        valid_ids = set(s2["entity_id"]) | set(s3["entity_id"])

        match_path = OUTPUT / "matching_results.tsv"
        if match_path.is_file():
            m_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
            for _, r in m_df.iterrows():
                raw = r["matched_entity_ids"]
                ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
                for m in ids:
                    self.assertIn(m, valid_ids, f"Predicted ID {m} does not exist in test data")

    def test_10_empty_matches(self):
        """Test 10: Empty candidate/match lists are formatted correctly as empty strings."""
        match_path = OUTPUT / "matching_results.tsv"
        if match_path.is_file():
            m_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
            for _, r in m_df.iterrows():
                val = r["matched_entity_ids"]
                self.assertIsInstance(val, str)
                if not val:
                    self.assertEqual(val, "")

if __name__ == "__main__":
    unittest.main()
