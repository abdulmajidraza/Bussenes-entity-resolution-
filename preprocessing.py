import re
import unicodedata

LEGAL_SUFFIXES = {
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "limited": "ltd",
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "gesellschaft mit beschrankter haftung": "gmbh",
    "societe anonyme": "sa",
    "societe a responsabilite limitee": "sarl",
    "limited liability company": "llc",
}

ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "apartment": "apt",
    "suite": "ste",
    "highway": "hwy",
    "parkway": "pkwy",
    "drive": "dr",
    "lane": "ln",
    "floor": "flr",
    "building": "bldg",
}

def pd_isna(val):
    return val != val or val is None

def normalize_text(x):
    if x is None or pd_isna(x):
        return ""
    x = str(x)
    # Unicode NFKC normalization
    x = unicodedata.normalize("NFKC", x).lower()
    x = x.replace("&", " and ")
    # Handle acronym dots like a.b.c. -> abc
    x = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b|\s|$)", "", x)
    # Replace remaining non-word, non-whitespace characters with space
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    # Normalize multiple whitespace to single space
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_name(x):
    s = normalize_text(x)
    if not s:
        return ""
    for full, abbrev in LEGAL_SUFFIXES.items():
        s = re.sub(rf"\b{re.escape(full)}\b", abbrev, s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_address(x):
    s = normalize_text(x)
    if not s:
        return ""
    for full, abbrev in ADDRESS_ABBREVIATIONS.items():
        s = re.sub(rf"\b{re.escape(full)}\b", abbrev, s)
    return re.sub(r"\s+", " ", s).strip()

def enrich(df):
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].map(normalize_text)
    return out
