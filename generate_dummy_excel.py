import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import string
import os

# Initialize Faker
fake = Faker()
Faker.seed(42)  # For reproducibility
random.seed(42)

# Configuration
NUM_ROWS = 20000
OUTPUT_FILE = "/home/ayman/workspace/LLM_Project/Data_Preparation/patent_database_sample.xlsx"

def generate_random_date(start_year=1995, end_year=2024):
    """Generate a random date between start_year and end_year"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    return (start_date + timedelta(days=random_days)).date()

def generate_expiry_date(filing_date):
    """Generate an expiry date (typically 20 years after filing)"""
    if filing_date:
        try:
            # Exactly 20 years after filing to avoid validation issues
            return filing_date.replace(year=filing_date.year + 20)
        except:
            # Handle leap year edge case
            return filing_date.replace(year=filing_date.year + 20, day=28)
    return None

def generate_patent_number():
    """Generate a realistic-looking patent number"""
    formats = [
        f"US{random.randint(6000000, 11999999)}",  # US format
        f"EP{random.randint(1000000, 3999999)}",   # EP format
        f"JP{random.randint(2000000, 6999999)}",   # JP format
        f"CN{random.randint(100000000, 299999999)}" # CN format
    ]
    return random.choice(formats)

def generate_publication_number():
    """Generate a publication number"""
    country_codes = ["US", "EP", "JP", "CN", "KR", "GB", "DE", "FR", "CA", "AU"]
    country = random.choice(country_codes)
    year = random.randint(2000, 2024)
    number = random.randint(10000, 999999)
    
    formats = [
        f"{country}{year}{number:06d}",
        f"{country}{year}/{number:06d}",
        f"{country}-{year}-{number:06d}"
    ]
    return random.choice(formats)

def generate_text_field(min_words=3, max_words=15):
    """Generate a text field with variable length"""
    return fake.sentence(nb_words=random.randint(min_words, max_words))

def generate_inventor_names():
    """Generate 1-5 inventor names"""
    num_inventors = random.randint(1, 5)
    inventors = [fake.name() for _ in range(num_inventors)]
    return "; ".join(inventors)

def generate_company_name():
    """Generate a company name"""
    company_types = ["Inc.", "LLC", "Corp.", "Ltd.", "GmbH", "S.A.", "Co., Ltd."]
    return f"{fake.company()} {random.choice(company_types)}"

def generate_legal_entity():
    """Generate a legal entity (person or company)"""
    if random.random() < 0.7:  # 70% companies, 30% individuals
        return generate_company_name()
    else:
        return fake.name()

def generate_url():
    """Generate a URL for patent documents"""
    domains = ["patentscope.wipo.int", "patents.google.com", "uspto.gov", "epo.org", "j-platpat.inpit.go.jp"]
    domain = random.choice(domains)
    path = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"https://{domain}/patents/{path}"

print("Generating patent database sample data...")
print(f"Creating {NUM_ROWS} rows with 60+ columns...")

# Initialize the dataframe
data = {
    "id": list(range(1, NUM_ROWS + 1)),  # Primary key (ensure uniqueness)
    "docketidIndex": [random.randint(1000, 9999) for _ in range(NUM_ROWS)],
    "casenumberIndex": [f"CASE-{random.randint(2000, 2024)}-{random.randint(10000, 99999)}" for _ in range(NUM_ROWS)],
    "combined_inventors": [generate_inventor_names() for _ in range(NUM_ROWS)],
    "inforce": [random.choice([0, 1]) for _ in range(NUM_ROWS)],
    "priority": [random.choice([0, 1]) for _ in range(NUM_ROWS)],
}

# Generate dates - ensure no future dates for filing_date
current_date = datetime.now().date()
filing_dates = [
    min(generate_random_date(2000, 2023), current_date - timedelta(days=30))  # Ensure no future dates
    for _ in range(NUM_ROWS)
]

# Generate priority dates that are always before filing date
priority_dates = []
for date in filing_dates:
    if random.random() < 0.8:
        # Make priority date at least 1 day and up to 2 years before filing date
        days_before = random.randint(1, 730)
        priority_date = date - timedelta(days=days_before)
        priority_dates.append(priority_date)
    else:
        priority_dates.append(None)

# Add all date fields
data["priority_date"] = priority_dates
data["due_date_national"] = [
    date + timedelta(days=random.randint(180, 730)) if date else None 
    for date in filing_dates
]
data["current_status_date"] = [
    min(generate_random_date(max(2015, date.year), 2024), current_date)
    if date else None
    for date in filing_dates
]
data["filing_date"] = filing_dates
data["is_first_filing"] = [random.choice([0, 1]) for _ in range(NUM_ROWS)]
data["due_date_foreign"] = [
    date + timedelta(days=random.randint(270, 450)) if date else None
    for date in filing_dates
]
data["expiry_date"] = [generate_expiry_date(date) for date in filing_dates]

# Generate grant dates (some patents not granted yet) - always after filing date
data["grant_date"] = []
for filing_date in filing_dates:
    if filing_date and random.random() < 0.7:
        # Make grant date at least 1 day after filing date
        days_after = random.randint(365, 1825)
        grant_date = filing_date + timedelta(days=days_after)
        # Ensure grant date is not in the future
        if grant_date > current_date:
            grant_date = None
        data["grant_date"].append(grant_date)
    else:
        data["grant_date"].append(None)

# Add remaining fields
data["is_first_grant"] = [random.choice([0, 1]) if grant_date else 0 for grant_date in data["grant_date"]]
data["grant_number"] = [generate_patent_number() if grant_date else "" for grant_date in data["grant_date"]]
data["title"] = [fake.sentence(nb_words=random.randint(5, 15)) for _ in range(NUM_ROWS)]

# Publication dates (usually 18 months after filing but before current date)
data["publication_date"] = []
for filing_date in filing_dates:
    if filing_date:
        pub_date = filing_date + timedelta(days=random.randint(540, 600))
        # Ensure publication date is not in the future
        if pub_date > current_date:
            pub_date = current_date - timedelta(days=random.randint(1, 30))
        data["publication_date"].append(pub_date)
    else:
        data["publication_date"].append(None)

data["publicationno"] = [generate_publication_number() for _ in range(NUM_ROWS)]
data["publicationlink"] = [generate_url() for _ in range(NUM_ROWS)]
data["espace_publink"] = [f"https://worldwide.espacenet.com/patent/search?q=pn%3D{pub_no}" 
                         if pub_no else "" for pub_no in data["publicationno"]]
data["is_fam_publication"] = [random.choice([0, 1]) for _ in range(NUM_ROWS)]
data["memotech_pubno"] = [f"MT-{random.randint(10000, 99999)}" for _ in range(NUM_ROWS)]
data["memotech_publink"] = [generate_url() for _ in range(NUM_ROWS)]
data["casekey"] = [f"CK{random.randint(1000, 9999)}" for _ in range(NUM_ROWS)]
data["archived"] = [random.choice(["Yes", "No"]) for _ in range(NUM_ROWS)]
data["first_filing"] = [f"FF-{random.randint(10000, 99999)}" for _ in range(NUM_ROWS)]

# First filing date should be before or same as filing date
data["first_filing_date"] = []
for filing_date in filing_dates:
    if filing_date and random.random() < 0.8:
        # Make first filing date at most 2 years before filing date
        days_before = random.randint(0, 730)
        first_filing_date = filing_date - timedelta(days=days_before)
        data["first_filing_date"].append(first_filing_date)
    else:
        data["first_filing_date"].append(None)

data["first_priority"] = [f"FP-{random.randint(10000, 99999)}" for _ in range(NUM_ROWS)]
data["applicants"] = [generate_legal_entity() for _ in range(NUM_ROWS)]
data["legal_owners"] = [generate_legal_entity() for _ in range(NUM_ROWS)]
data["registered_owners"] = [generate_legal_entity() for _ in range(NUM_ROWS)]
data["product_structure_stc"] = [generate_text_field(3, 10) for _ in range(NUM_ROWS)]
data["export_restriction"] = [random.choice(["None", "Export Controlled", "Limited Distribution"]) for _ in range(NUM_ROWS)]
data["accession_number"] = [f"ACC{random.randint(100000, 999999)}" for _ in range(NUM_ROWS)]
data["case_fc_count"] = [random.randint(0, 10) for _ in range(NUM_ROWS)]
data["countryidIndex"] = [random.randint(1, 200) for _ in range(NUM_ROWS)]
data["filingtypeidIndex"] = [random.randint(1, 10) for _ in range(NUM_ROWS)]
data["filing_number"] = [f"FN-{random.randint(100000, 999999)}" for _ in range(NUM_ROWS)]
data["prosecution_status_idIndex"] = [random.randint(1, 8) for _ in range(NUM_ROWS)]
data["sbuidIndex"] = [random.randint(1, 50) for _ in range(NUM_ROWS)]
data["buidIndex"] = [random.randint(1, 100) for _ in range(NUM_ROWS)]
data["blidIndex"] = [random.randint(1, 200) for _ in range(NUM_ROWS)]
data["latest_annuity_number"] = [random.randint(0, 20) for _ in range(NUM_ROWS)]
data["current_annuity_caseIndex"] = [random.randint(1, 5000) for _ in range(NUM_ROWS)]
data["current_attorneyIndex"] = [random.randint(1, 500) for _ in range(NUM_ROWS)]
data["current_activated"] = [random.choice([0, 1]) for _ in range(NUM_ROWS)]

# Current activation dates - ensure they're not in the future
data["current_activation_date"] = []
for activated in data["current_activated"]:
    if activated == 1:
        activation_date = generate_random_date(2015, 2024)
        # Ensure activation date is not in the future
        if activation_date > current_date:
            activation_date = current_date - timedelta(days=random.randint(1, 90))
        data["current_activation_date"].append(activation_date)
    else:
        data["current_activation_date"].append(None)

data["current_decision"] = [random.choice([0, 1, 2, 3]) for _ in range(NUM_ROWS)]

# Decision dates - ensure they're not in the future
data["current_decision_date"] = []
for decision in data["current_decision"]:
    if decision > 0:
        decision_date = generate_random_date(2015, 2024)
        # Ensure decision date is not in the future
        if decision_date > current_date:
            decision_date = current_date - timedelta(days=random.randint(1, 90))
        data["current_decision_date"].append(decision_date)
    else:
        data["current_decision_date"].append(None)

data["decision_taken_before_days"] = [random.randint(0, 180) for _ in range(NUM_ROWS)]
data["current_review_status"] = [random.choice([0, 1, 2, 3, 4]) for _ in range(NUM_ROWS)]
data["prm_current_review_status"] = [random.choice([0, 1, 2, 3, 4]) for _ in range(NUM_ROWS)]
data["filenumberIndex"] = [random.randint(1000, 9999) for _ in range(NUM_ROWS)]
data["typename"] = [random.choice(["Utility Patent", "Design Patent", "Plant Patent", "Provisional"]) for _ in range(NUM_ROWS)]

# Make sure patent_design_number is consistent with typename and grant_date
data["patent_design_number"] = []
for typ, grant_date in zip(data["typename"], data["grant_date"]):
    if grant_date:  # Only granted patents have numbers
        if typ == "Design Patent":
            data["patent_design_number"].append(f"D{random.randint(100000, 999999)}")
        elif typ == "Plant Patent":
            data["patent_design_number"].append(f"PP{random.randint(10000, 99999)}")
        else:  # Utility patent or provisional
            data["patent_design_number"].append(f"US{random.randint(6000000, 11999999)}")
    else:
        data["patent_design_number"].append("")

# Ensure addition_timestamp is never in the future
data["addition_timestamp"] = [
    min(
        datetime.now() - timedelta(days=random.randint(0, 1095), 
                                 hours=random.randint(0, 23),
                                 minutes=random.randint(0, 59),
                                 seconds=random.randint(0, 59)),
        datetime.now()
    )
    for _ in range(NUM_ROWS)
]

data["has_annuity_data"] = [random.choice([0, 1]) for _ in range(NUM_ROWS)]
data["application_number"] = [f"APP{random.randint(100000, 999999)}" for _ in range(NUM_ROWS)]

# PCT filing date should be after filing date but before current date
data["pct_filing_date"] = []
for filing_date in filing_dates:
    if filing_date and random.random() < 0.3:
        pct_date = filing_date + timedelta(days=random.randint(120, 360))
        # Ensure PCT date is not in the future
        if pct_date > current_date:
            pct_date = None
        data["pct_filing_date"].append(pct_date)
    else:
        data["pct_filing_date"].append(None)

# Add current_statusIndex
data["current_statusIndex"] = [random.randint(1, 15) for _ in range(NUM_ROWS)]

# Ensure dates are consistent with inforce flag
for i in range(NUM_ROWS):
    if data["expiry_date"][i] and data["expiry_date"][i] <= current_date:
        # If expired, should not be in force
        data["inforce"][i] = 0

# Create DataFrame and export to Excel
df = pd.DataFrame(data)

# Verify no duplicate IDs before saving
if not df['id'].is_unique:
    print("Warning: Duplicate IDs detected. Fixing before saving...")
    df['id'] = list(range(1, len(df) + 1))

# Verify required fields have no nulls
required_fields = ['id', 'filing_date', 'title', 'applicants', 'current_statusIndex', 'countryidIndex']
for field in required_fields:
    null_count = df[field].isna().sum()
    if null_count > 0:
        print(f"Warning: {null_count} NULL values in required field '{field}'. Fixing...")
        if field == 'filing_date':
            # Fill missing filing dates with reasonable values
            df.loc[df['filing_date'].isna(), 'filing_date'] = current_date - timedelta(days=365)
        elif field in ['title', 'applicants']:
            # Fill missing text fields
            if field == 'title':
                df.loc[df[field].isna(), field] = df.loc[df[field].isna()].apply(lambda _: fake.sentence(nb_words=10))
            else:
                df.loc[df[field].isna(), field] = df.loc[df[field].isna()].apply(lambda _: generate_legal_entity())
        else:
            # Fill missing numeric fields
            if field == 'id':
                df.loc[df[field].isna(), field] = df.loc[df[field].isna()].index + 1
            elif field == 'current_statusIndex':
                df.loc[df[field].isna(), field] = df.loc[df[field].isna()].apply(lambda _: random.randint(1, 15))
            elif field == 'countryidIndex':
                df.loc[df[field].isna(), field] = df.loc[df[field].isna()].apply(lambda _: random.randint(1, 200))

print(f"Writing data to {OUTPUT_FILE}...")
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Done! Generated {NUM_ROWS} rows of patent database sample data.")
print(f"Output file saved as: {os.path.abspath(OUTPUT_FILE)}")