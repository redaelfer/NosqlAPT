"""
APT Detection - Training Demo Script
A simple script to demonstrate training the APT detection model
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import our APT detection modules
from src.apt_detector import APTDetector
from src.apt_data_prep import APTDataPreprocessor
from src.combined_model import CombinedAPTModel

print(f"Starting APT detection model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Available methods in APTDataPreprocessor:", [method for method in dir(APTDataPreprocessor) if not method.startswith('__')])

# Step 1: Create output directories
print("\n--- Step 1: Setting up directories ---")
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
print("✓ Created output directories")

# Step 2: Initialize the detector and connection to MongoDB
print("\n--- Step 2: Initializing detector ---")
detector = APTDetector('config/apt_detection_config.json')
print("✓ Initialized APT detector")

# Step 3: Load data (either from MongoDB or generate synthetic)
print("\n--- Step 3: Loading data ---")
try:
    data = detector.fetch_data(limit=5000)
    if data is not None and len(data) > 0:
        print(f"✓ Loaded {len(data)} records from MongoDB")
    else:
        raise ValueError("No data returned from MongoDB")
except Exception as e:
    print(f"Could not load data from MongoDB: {str(e)}")
    print("Generating synthetic data instead...")
    preprocessor = APTDataPreprocessor()
    data = preprocessor.generate_synthetic_apt_data(n_samples=5000, contamination=0.05)
    print(f"✓ Generated {len(data)} synthetic records with {data['is_apt'].sum()} APTs")

# Step 4: Preprocess data
print("\n--- Step 4: Preprocessing data ---")
preprocessor = APTDataPreprocessor()

# Create a basic preprocessing pipeline using the available methods
print("  > Handling missing values...")
data = preprocessor.handle_missing_values(data)

# Check if target column exists
if 'is_apt' not in data.columns:
    print("  > Creating target column 'is_apt'...")
    # Create a synthetic target column with 5% APTs
    data['is_apt'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])

# Save target column
target = data['is_apt'].copy()
data_features = data.drop(columns=['is_apt'])

print("  > Processing timestamps...")
data_features = preprocessor.handle_timestamps(data_features)

print("  > Processing IP addresses...")
data_features = preprocessor.handle_ip_addresses(data_features)

print("  > Processing ports and protocols...")
data_features = preprocessor.handle_ports_and_protocols(data_features)

print("  > Normalizing features...")
data_features = preprocessor.normalize_features(data_features, fit=True)

print("  > Encoding categorical variables...")
data_features = preprocessor.encode_categorical(data_features)

# Add target back
data_features['is_apt'] = target

# Use this as our processed data
processed_data = data_features

print(f"✓ Preprocessed data: {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")

# Step 5: Train model
print("\n--- Step 5: Training model ---")
print("  > Initializing model...")
model = CombinedAPTModel('config/apt_detection_config.json')

print("  > Training autoencoder and XGBoost...")
start_time = datetime.now()
results = model.train(processed_data, target_col='is_apt')
training_time = (datetime.now() - start_time).total_seconds()

print(f"✓ Model training completed in {training_time:.1f} seconds")

# Step 6: Save model and results
print("\n--- Step 6: Saving model ---")
model.save('models')
print("✓ Model saved to 'models' directory")

# Print performance metrics
print("\n--- Training Results ---")
if 'performance' in results and 'combined_model' in results['performance']:
    metrics = results['performance']['combined_model']
    print(f"Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
    print(f"Recall:    {metrics.get('recall', 'N/A'):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 'N/A'):.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics.get('roc_auc', 'N/A'):.4f}")
else:
    print(f"Accuracy:  {results.get('accuracy', 'N/A')}")
    print(f"Precision: {results.get('precision', 'N/A')}")
    print(f"Recall:    {results.get('recall', 'N/A')}")
    print(f"F1 Score:  {results.get('f1', 'N/A')}")

print("\nTraining completed successfully!")
print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")