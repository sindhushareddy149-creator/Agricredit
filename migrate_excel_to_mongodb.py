"""
AgriCredit — Excel to MongoDB Migration Script
Run this ONCE to move final_dataset.xlsx data into MongoDB.
After this, you can stop using the Excel file completely.

Usage:
    python migrate_excel_to_mongodb.py
"""

from pymongo import MongoClient
import pandas as pd
import numpy as np
import os

# ── Connect ──────────────────────────────────────────
client   = MongoClient("mongodb://localhost:27017/")
mongo_db = client["farmer_loan_db"]

farmers_collection = mongo_db["farmers"]
weather_collection = mongo_db["weather"]
config_col         = mongo_db["config"]

# ── Load Excel ────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH   = os.path.join(BASE_DIR, "final_dataset.xlsx")

print(f"📂 Loading: {EXCEL_PATH}")
df = pd.read_excel(EXCEL_PATH)
print(f"✅ Loaded {len(df)} rows, columns: {list(df.columns)}")

# ── Print column names (helps debug) ─────────────────
print("\nColumn names in your Excel:")
for col in df.columns:
    print(f"  • {col}")

# ── CLEAR EXISTING DATA ───────────────────────────────
farmers_collection.delete_many({})
weather_collection.delete_many({})
config_col.delete_many({"key": "market_score"})
print("\n🗑️  Cleared existing farmers, weather, config collections")

# ═══════════════════════════════════════════════════════
# 1. MIGRATE FARMER RECORDS
#    Each row in Excel → one document in MongoDB
# ═══════════════════════════════════════════════════════
print("\n📤 Migrating farmer records...")

farmer_docs = []
skipped     = 0

for _, row in df.iterrows():
    try:
        farmer_id = int(row["Farmer_ID"])
        year      = int(row["YEAR"])

        doc = {
            "farmer_id":            farmer_id,
            "year":                 year,

            # ── Identity ──────────────────────────────
            "name":                 str(row.get("name",    f"Farmer #{farmer_id}")),
            "password":             str(row.get("Password", f"farmer{farmer_id}@agri")),

            # ── Farm details ──────────────────────────
            "land_size_acres":      float(row.get("land_size_acres", 0)),
            "soil_type":            str(row.get("Soil_Type", "")),
            "annual_income":        float(row.get("annual_income",  0)),

            # ── Scores (stored for reference) ─────────
            "past_repayment_score": float(row.get("Past_Repayment_Score", 65)),
            "combined_score":       float(row.get("Combined_Score",       0)),
            "risk_label":           str(row.get("Risk_Label", "")),
        }

        # Optional columns — add if present in your Excel
        for optional_col, mongo_key in [
            ("Crop_Type",      "crop_type"),
            ("Location",       "location"),
            ("Nitrogen",       "nitrogen"),
            ("Phosphorus",     "phosphorus"),
            ("Potassium",      "potassium"),
            ("Loan_Amount",    "loan_amount"),
        ]:
            if optional_col in df.columns and not pd.isna(row.get(optional_col)):
                doc[mongo_key] = row[optional_col]

        farmer_docs.append(doc)

    except Exception as e:
        skipped += 1
        print(f"  ⚠️  Skipped row (Farmer_ID={row.get('Farmer_ID','?')}): {e}")

if farmer_docs:
    farmers_collection.insert_many(farmer_docs)
    print(f"✅ Inserted {len(farmer_docs)} farmer records into MongoDB")
else:
    print("❌ No farmer records inserted — check column names above")

if skipped:
    print(f"⚠️  Skipped {skipped} rows due to errors")

# ═══════════════════════════════════════════════════════
# 2. MIGRATE WEATHER SCORES
#    df.groupby("YEAR")["Weather_Score"].first() → weather collection
# ═══════════════════════════════════════════════════════
print("\n🌦️  Migrating weather scores...")

if "Weather_Score" in df.columns and "YEAR" in df.columns:
    weather_by_year = df.groupby("YEAR")["Weather_Score"].first().to_dict()
    weather_docs    = [
        {"year": int(yr), "weather_score": float(score)}
        for yr, score in weather_by_year.items()
    ]
    weather_collection.insert_many(weather_docs)
    print(f"✅ Inserted weather scores for {len(weather_docs)} years: {sorted(weather_by_year.keys())}")
else:
    print("⚠️  Weather_Score or YEAR column not found — inserting default weather scores")
    default_weather = [
        {"year": 2020, "weather_score": 88.0},
        {"year": 2021, "weather_score": 85.0},
        {"year": 2022, "weather_score": 90.0},
        {"year": 2023, "weather_score": 87.0},
        {"year": 2024, "weather_score": 91.0},
        {"year": 2025, "weather_score": 89.0},
        {"year": 2026, "weather_score": 88.5},
    ]
    weather_collection.insert_many(default_weather)
    print(f"✅ Inserted {len(default_weather)} default weather records")

# ═══════════════════════════════════════════════════════
# 3. STORE MARKET SCORE IN CONFIG
#    df["Market_Score"].median() → config collection
# ═══════════════════════════════════════════════════════
print("\n📈 Storing market score...")

if "Market_Score" in df.columns:
    market_score = float(df["Market_Score"].median())
    print(f"   Market Score (median from Excel): {market_score:.2f}")
else:
    market_score = 72.5
    print(f"   Market_Score column not found — using default: {market_score}")

config_col.insert_one({"key": "market_score", "value": market_score})
print(f"✅ Market score {market_score:.2f} saved to config collection")

# ═══════════════════════════════════════════════════════
# 4. CREATE INDEXES (speeds up lookups)
# ═══════════════════════════════════════════════════════
print("\n⚡ Creating MongoDB indexes...")
farmers_collection.create_index([("farmer_id", 1), ("year", -1)])
farmers_collection.create_index("farmer_id")
weather_collection.create_index("year")
print("✅ Indexes created")

# ═══════════════════════════════════════════════════════
# 5. VERIFY MIGRATION
# ═══════════════════════════════════════════════════════
print("\n📊 Migration Summary:")
print(f"   farmers    collection: {farmers_collection.count_documents({})} documents")
print(f"   weather    collection: {weather_collection.count_documents({})} documents")
print(f"   config     collection: {config_col.count_documents({})} documents")

# Show sample farmer
sample = farmers_collection.find_one({}, {"_id": 0})
if sample:
    print(f"\n🔍 Sample farmer document:")
    for k, v in sample.items():
        print(f"   {k}: {v}")

print("\n🎉 Migration complete! You can now run app.py without final_dataset.xlsx")
print("   (Keep the Excel file as backup — the app no longer needs it)")