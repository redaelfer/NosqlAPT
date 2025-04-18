#!/usr/bin/env python3
"""
APT Detection System Demo
This script demonstrates the complete APT detection workflow.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import APT modules
try:
    from apt_detector import APTDetector
    from apt_data_prep import APTDataPreprocessor
    from combined_model import CombinedAPTModel
except ImportError as e:
    print(f"Error importing APT modules: {e}")
    print("Make sure you have the correct directory structure or PYTHONPATH set.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("apt_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('apt_demo')


def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'src', 'visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories


def generate_synthetic_data():
    """Generate synthetic data for demo"""
    logger.info("Generating synthetic APT data")

    # Create preprocessor
    preprocessor = APTDataPreprocessor()

    # Generate synthetic data
    data = preprocessor.generate_synthetic_apt_data(n_samples=5000, contamination=0.05)

    # Save data
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/synthetic_apt_data.csv', index=False)

    # Log info
    logger.info(f"Generated {len(data)} samples with {data['is_apt'].sum()} APTs")

    return data


def train_model(data):
    """Train the APT detection model"""
    logger.info("Training APT detection model")

    # Create model
    model = CombinedAPTModel()

    # Train model
    results = model.train(data, target_col='is_apt')

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models')

    # Save src
    os.makedirs('src', exist_ok=True)
    with open('src/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    logger.info("Model training complete")

    return model, results


def evaluate_model(model, data):
    """Evaluate the model on test data"""
    logger.info("Evaluating model performance")

    # Split ground truth from features
    y_true = data['is_apt'].values

    # Make predictions
    prediction_results = model.predict(data, return_details=True)

    # Extract predictions and details
    y_pred = prediction_results['predictions']
    y_proba = prediction_results['probabilities']
    rec_errors = prediction_results['reconstruction_errors']

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    # Generate comprehensive report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Save evaluation src
    os.makedirs('src', exist_ok=True)
    with open('src/evaluation_results.json', 'w') as f:
        json.dump(report, f, indent=4)

    # Create visualizations
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)

    # Visualize src
    viz_results = model.visualize_results(
        data.drop(columns=['is_apt']).values,
        y_true,
        save_path=viz_dir
    )

    logger.info("Model evaluation complete")

    return report


def explain_model(model, data):
    """Explain model predictions"""
    logger.info("Explaining model predictions")

    # Find a positive example (APT)
    apt_samples = data[data['is_apt'] == 1]

    if len(apt_samples) > 0:
        # Get a sample
        sample = apt_samples.iloc[[0]]

        # Get explanation
        explanation = model.explain_prediction(sample)

        # Save explanation
        os.makedirs('src', exist_ok=True)
        with open('src/apt_explanation.json', 'w') as f:
            json.dump(explanation, f, indent=4)

        # Print top contributing features
        if 'feature_contributions' in explanation:
            logger.info("Top contributing features for APT detection:")
            for i, feat in enumerate(explanation['feature_contributions'][:5]):
                logger.info(f"  {i + 1}. {feat['feature']}: {feat['contribution']:.4f}")
    else:
        logger.warning("No APT samples found for explanation")


def run_mongodb_demo():
    """Run demo with MongoDB connection if available"""
    logger.info("Checking MongoDB connection for demo")

    try:
        from pymongo import MongoClient

        # Try to connect to MongoDB
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        client.server_info()  # Will raise exception if connection fails

        logger.info("MongoDB connection successful")

        # Read configuration
        with open('config/apt_detection_config.json', 'r') as f:
            config = json.load(f)

        # Initialize detector
        detector = APTDetector('config/apt_detection_config.json')

        # Check if collection exists and has data
        collection = detector.mongodb[config['mongodb']['collection']]
        count = collection.count_documents({})

        if count > 0:
            logger.info(f"Found {count} documents in MongoDB collection")

            # Fetch some data
            data = detector.fetch_data(limit=1000)

            if data is not None and len(data) > 0:
                logger.info(f"Successfully fetched {len(data)} records from MongoDB")

                # Create a target column if not present
                if 'is_apt' not in data.columns:
                    logger.info("Creating synthetic target column for demonstration")

                    # Simple rule: mark records with high packet variance as potential APTs
                    if 'Packet Length Variance' in data.columns:
                        threshold = data['Packet Length Variance'].quantile(0.95)
                        data['is_apt'] = (data['Packet Length Variance'] > threshold).astype(int)
                    else:
                        # Random labels with 5% APTs
                        data['is_apt'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])

                # Preprocess data
                preprocessor = APTDataPreprocessor()
                processed_data = preprocessor.preprocess_apt_data(data, target_col='is_apt')

                # Initialize model
                model = CombinedAPTModel('config/apt_detection_config.json')

                # Train model
                logger.info("Training model on MongoDB data")
                results = model.train(processed_data, target_col='is_apt')

                # Save model
                model.save('models/mongodb_model')

                # Save src
                with open('src/mongodb_training_results.json', 'w') as f:
                    json.dump(results, f, indent=4)

                logger.info("MongoDB demo completed successfully")
                return True

    except Exception as e:
        logger.warning(f"MongoDB demo failed: {str(e)}")

    logger.info("MongoDB demo not available or failed")
    return False


def main():
    """Main function"""
    logger.info("Starting APT Detection System Demo")

    # Setup directories
    setup_directories()

    # Check if we can use MongoDB
    mongodb_demo_success = run_mongodb_demo()

    if not mongodb_demo_success:
        # Generate synthetic data
        logger.info("Using synthetic data for demo")
        data = generate_synthetic_data()

        # Split data for training and testing
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['is_apt'], random_state=42)

        # Train model
        model, training_results = train_model(train_data)

        # Evaluate model
        evaluation_results = evaluate_model(model, test_data)

        # Explain predictions
        explain_model(model, test_data)

    logger.info("APT Detection System Demo completed")


if __name__ == "__main__":
    main()