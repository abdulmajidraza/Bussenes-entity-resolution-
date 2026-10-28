import random
from pathlib import Path
import pandas as pd

random.seed(42)

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Base business entity templates across US, India, France
SEED_ENTITIES = [
    # US Entities
    ("Apex Technologies Corp", "100 Industrial Parkway, Suite 400, Austin, TX 78701", "United States"),
    ("Blue Sky Logistics LLC", "550 Market Street, Floor 12, San Francisco, CA 94105", "United States"),
    ("Cascade Health Systems Inc", "800 Bellevue Way NE, Bellevue, WA 98004", "United States"),
    ("Dynamic Digital Media Group", "200 Fifth Avenue, New York, NY 10010", "United States"),
    ("Evergreen Renewable Energy", "1200 6th Ave, Seattle, WA 98101", "United States"),
    ("Frontier Financial Services", "300 S Wacker Dr, Chicago, IL 60606", "United States"),
    ("Global Tech Solutions", "1 Penn Plaza, New York, NY 10119", "United States"),
    ("Horizon BioPharma Labs", "400 Technology Sq, Cambridge, MA 02139", "United States"),
    ("Imperial Automotive Group", "700 Michigan Ave, Detroit, MI 48226", "United States"),
    ("Jupiter Retail Enterprises", "500 Pine St, St. Louis, MO 63101", "United States"),
    
    # India Entities
    ("Tata Consultancy Services Ltd", "Nirmal Building, 9th Floor, Nariman Point, Mumbai 400021", "India"),
    ("Infosys Technologies Pvt Ltd", "Electronics City, Hosur Road, Bengaluru 560100", "India"),
    ("Wipro Digital Solutions", "Doddakannelli, Sarjapur Road, Bengaluru 560035", "India"),
    ("Reliance Industries Limited", "Maker Chambers IV, 222 Nariman Point, Mumbai 400021", "India"),
    ("HDFC Financial Services Ltd", "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai 400013", "India"),
    ("Mahindra & Mahindra Enterprises", "Gateway Building, Apollo Bunder, Mumbai 400001", "India"),
    ("Bharti Airtel Telecommunications", "Bharti Crescent, 1 Nelson Mandela Road, Vasant Kunj, New Delhi 110070", "India"),
    ("Larsen & Toubro Construction", "L&T House, Ballard Estate, Mumbai 400001", "India"),
    ("Sun Pharmaceutical Industries", "SUN House, CTS No 201 B/1, Western Express Highway, Goregaon East, Mumbai 400063", "India"),
    ("Adani Enterprises Limited", "Adani Corporate House, Shantigram, Near Vaishno Devi Circle, Ahmedabad 382421", "India"),
    
    # France Entities (unseen in training, tests open-set processing)
    ("Société Générale SA", "29 Boulevard Haussmann, 75009 Paris", "France"),
    ("TotalEnergies Marketing SARL", "2 Place Jean Millier, 92400 Courbevoie", "France"),
    ("Schneider Electric SE", "35 Rue Joseph Monier, 92500 Rueil-Malmaison", "France"),
    ("Danone Agro-Alimentaire", "17 Boulevard Haussmann, 75009 Paris", "France"),
    ("Air France-KLM Group", "45 Rue de Paris, 95747 Roissy CDG", "France"),
    ("Capgemini Engineering Services", "11 Rue de Tilsitt, 75017 Paris", "France"),
    ("Sanofi Healthcare France", "54 Rue La Boétie, 75008 Paris", "France"),
    ("L'Oréal Beauty & Cosmetics", "41 Rue Martre, 92117 Clichy", "France"),
]

def vary_name(name):
    # Apply synthetic variation (legal suffixes, typos, abbreviations)
    vars_ = [
        name,
        name.replace(" Corporation", " Corp").replace(" Limited", " Ltd").replace(" Private", " Pvt"),
        name.replace(" & ", " and "),
        name.lower(),
        name.upper(),
        name.replace(".", ""),
        name + " Inc.",
    ]
    return random.choice(vars_)

def vary_address(addr):
    vars_ = [
        addr,
        addr.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave").replace("Suite", "Ste"),
        addr.lower(),
        addr.replace(",", ""),
        addr.replace("Floor", "Flr"),
    ]
    return random.choice(vars_)

def create_dataset_partition(partition_name, base_offset, num_s1=60):
    s1_rows = []
    s2_rows = []
    s3_rows = []
    gt_rows = []

    s2_id_counter = base_offset * 10
    s3_id_counter = base_offset * 10

    for i in range(num_s1):
        s1_id = f"S1-{base_offset + i + 1:05d}"
        seed_idx = i % len(SEED_ENTITIES)
        b_name, b_addr, cty = SEED_ENTITIES[seed_idx]
        
        # Add random suffix or variation to prevent exact duplicates across indices
        if i >= len(SEED_ENTITIES):
            b_name = f"{b_name} Branch {i // len(SEED_ENTITIES)}"
            b_addr = f"{i * 10} " + b_addr

        # Source 1 record
        s1_name = vary_name(b_name)
        s1_addr = vary_address(b_addr)
        s1_rows.append({
            "entity_id": s1_id,
            "business_name": s1_name,
            "business_address": s1_addr,
            "country": cty
        })

        matches = []
        match_type = random.choice(["single_s2", "single_s3", "both", "none"])

        if match_type in ("single_s2", "both"):
            s2_id_counter += 1
            s2_id = f"S2-{s2_id_counter:05d}"
            s2_rows.append({
                "entity_id": s2_id,
                "business_name": vary_name(b_name),
                "business_address": vary_address(b_addr),
                "country": cty
            })
            matches.append(s2_id)

        if match_type in ("single_s3", "both"):
            s3_id_counter += 1
            s3_id = f"S3-{s3_id_counter:05d}"
            s3_rows.append({
                "entity_id": s3_id,
                "business_name": vary_name(b_name),
                "business_address": vary_address(b_addr),
                "country": cty
            })
            matches.append(s3_id)

        gt_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches)
        })

    # Add distractor non-matching records to S2 and S3 (hard negatives)
    for j in range(20):
        s2_id_counter += 1
        s3_id_counter += 1
        distractor_seed = SEED_ENTITIES[j % len(SEED_ENTITIES)]
        s2_rows.append({
            "entity_id": f"S2-{s2_id_counter:05d}",
            "business_name": f"Distractor {distractor_seed[0]}",
            "business_address": f"9999 Disconnected Rd, {distractor_seed[1]}",
            "country": distractor_seed[2]
        })
        s3_rows.append({
            "entity_id": f"S3-{s3_id_counter:05d}",
            "business_name": f"Unrelated {distractor_seed[0]}",
            "business_address": f"8888 Nowhere St, {distractor_seed[1]}",
            "country": distractor_seed[2]
        })

    s1_df = pd.DataFrame(s1_rows)
    s2_df = pd.DataFrame(s2_rows)
    s3_df = pd.DataFrame(s3_rows)
    gt_df = pd.DataFrame(gt_rows)

    return s1_df, s2_df, s3_df, gt_df

def main():
    print("Generating training dataset...")
    tr_s1, tr_s2, tr_s3, tr_gt = create_dataset_partition("train", base_offset=100, num_s1=80)
    tr_s1.to_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", index=False)
    tr_s2.to_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", index=False)
    tr_s3.to_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", index=False)
    tr_gt.to_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", index=False)

    print("Generating test dataset...")
    te_s1, te_s2, te_s3, _ = create_dataset_partition("test", base_offset=500, num_s1=40)
    te_s1.to_csv(TEST_DIR / "test_source1.tsv", sep="\t", index=False)
    te_s2.to_csv(TEST_DIR / "test_source2.tsv", sep="\t", index=False)
    te_s3.to_csv(TEST_DIR / "test_source3.tsv", sep="\t", index=False)

    print(f"Train Source 1 records: {len(tr_s1)}")
    print(f"Test Source 1 records: {len(te_s1)}")
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
