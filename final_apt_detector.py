
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pymongo import MongoClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model and metadata
pipeline = joblib.load("apt_stage_pipeline.pkl")
stage_encoder = joblib.load("stage_encoder.pkl")
with open("expected_features.json", "r") as f:
    expected_features = json.load(f)

# Connect to MongoDB only (store suspects)
def connect_mongodb():
    try:
        mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        mongo_db = mongo_client['apt_detection_final']
        logger.info("Connected to MongoDB successfully.")
        return mongo_db
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        exit(1)

# Improved semantic feature extractor
def extract_features_from_event(event):
    keywords = event.lower().split("_")
    return {
        "Flow Duration": 80000 if "exfiltration" in keywords else 4000,
        "Total Fwd Packet": keywords.count("fwd"),
        "Total Bwd packets": keywords.count("bwd"),
        "Flow Bytes/s": 850.0 if "exfiltration" in keywords or "c2" in keywords else 150.0,
        "Flow Packets/s": 50.0 if "scan" not in keywords else 300.0,
        "Packet Length Mean": 700.0 if "ack" in keywords else 400.0,
        "Packet Length Std": 50.0,
        "Packet Length Variance": 1200.0,
        "FIN Flag Count": 1 if "fin" in keywords else 0,
        "SYN Flag Count": 1 if "syn" in keywords or "scan" in keywords else 0,
        "RST Flag Count": 1 if "rst" in keywords else 0,
        "PSH Flag Count": 1 if "psh" in keywords else 0,
        "ACK Flag Count": 1 if "ack" in keywords else 0,
        "URG Flag Count": 0,
        "Average Packet Size": 900.0,
        "Fwd Segment Size Avg": 1000.0,
        "Bwd Segment Size Avg": 900.0
    }

# Read and parse logs
def load_logs(filename):
    df = pd.read_csv(filename, header=None, names=['timestamp', 'source', 'destination', 'event'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    parsed_features = [extract_features_from_event(row['event']) for _, row in df.iterrows()]
    df_features = pd.DataFrame(parsed_features)

    # Ensure all expected features exist
    for col in expected_features:
        if col not in df_features.columns:
            df_features[col] = 0

    df_features = df_features[expected_features]
    return df, df_features

# Main logic
def main():
    mongo_db = connect_mongodb()
    collection = mongo_db['suspected_flows']

    original_df, features_df = load_logs("logs.txt")
    predictions = pipeline.predict(features_df)
    predicted_stages = stage_encoder.inverse_transform(predictions)
    original_df["predicted_stage"] = predicted_stages

    suspect_count = 0
    for _, row in original_df.iterrows():
        if row["predicted_stage"] != "benign":
            doc = {
                "timestamp": row["timestamp"].isoformat(),
                "source": row["source"],
                "destination": row["destination"],
                "event": row["event"],
                "predicted_stage": row["predicted_stage"]
            }
            collection.insert_one(doc)
            suspect_count += 1

    logger.info(f"✓ {suspect_count} suspected flows saved to MongoDB (non-benign only).")
    print(f"✓ {suspect_count} suspected flows stored successfully.")

if __name__ == "__main__":
    main()
