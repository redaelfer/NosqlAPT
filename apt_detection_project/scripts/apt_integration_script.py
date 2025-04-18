#!/usr/bin/env python3
"""
APT Detection System - MongoDB Integration Script
This script integrates the APT detection model with MongoDB to detect Advanced Persistent Threats.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import APT detector modules
try:
    from apt_detector import APTDetector
    from apt_data_prep import APTDataPreprocessor
    from combined_model import CombinedAPTModel
except ImportError as e:
    print(f"Error importing APT detector modules: {e}")
    print("Make sure you have the correct directory structure or PYTHONPATH set.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("apt_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('apt_integration')


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='APT Detection Integration with MongoDB')

    parser.add_argument('--config', type=str, default='config/apt_detection_config.json',
                        help='Path to configuration file')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'compare', 'monitor'],
                        help='Operation mode')

    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV data file (if not using MongoDB)')

    parser.add_argument('--target', type=str, default='is_apt',
                        help='Target column name for labeled data')

    parser.add_argument('--output', type=str, default='src',
                        help='Output directory for src')

    parser.add_argument('--limit', type=int, default=10000,
                        help='Maximum number of records to fetch from MongoDB')

    parser.add_argument('--collection', type=str, default=None,
                        help='MongoDB collection to use (overrides config)')

    parser.add_argument('--optimize', action='store_true',
                        help='Optimize model hyperparameters')

    return parser.parse_args()


def setup_mongodb_connection(config_path, collection_override=None):
    """
    Set up a connection to MongoDB

    Parameters:
    -----------
    config_path : str
        Path to configuration file
    collection_override : str, optional
        Collection name to override the one in config

    Returns:
    --------
    APTDetector
        Configured APT detector with MongoDB connection
    """
    try:
        # Initialize the detector with configuration
        detector = APTDetector(config_path)

        # Override collection if specified
        if collection_override and detector.mongo_client:
            mongo_config = detector.config['mongodb']
            detector.mongodb = detector.mongo_client[mongo_config['db_name']]
            logger.info(f"Using collection: {collection_override}")

        # Verify MongoDB connection
        if detector.mongo_client is None:
            logger.error("Failed to connect to MongoDB. Check your configuration.")
            return None

        return detector

    except Exception as e:
        logger.error(f"Error setting up MongoDB connection: {str(e)}")
        return None


def load_data(args, detector):
    """
    Load data from CSV file or MongoDB

    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    detector : APTDetector
        Initialized APT detector

    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    # Load from CSV file if specified
    if args.data and os.path.exists(args.data):
        logger.info(f"Loading data from CSV file: {args.data}")
        try:
            data = pd.read_csv(args.data)
            logger.info(f"Loaded {len(data)} records from CSV file")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            return None

    # Load from MongoDB
    elif detector and detector.mongo_client:
        logger.info(f"Fetching data from MongoDB (limit: {args.limit})")
        try:
            collection = args.collection or detector.config['mongodb']['collection']

            # Create a query to fetch data
            query = {}  # Empty query to fetch all records

            # Fetch data
            data = detector.fetch_data(query=query, limit=args.limit)

            if data is None or len(data) == 0:
                logger.error("No data retrieved from MongoDB")
                return None

            logger.info(f"Fetched {len(data)} records from MongoDB")
            return data

        except Exception as e:
            logger.error(f"Error fetching data from MongoDB: {str(e)}")
            return None

    else:
        logger.error("No data source available (neither CSV file nor MongoDB connection)")
        return None


def preprocess_data(data, target_col='is_apt'):
    """
    Preprocess data for APT detection

    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    target_col : str, default='is_apt'
        Target column name

    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    logger.info("Preprocessing data...")

    # Initialize preprocessor
    preprocessor = APTDataPreprocessor()

    # Handle missing target column for training data
    if target_col not in data.columns:
        logger.warning(f"Target column '{target_col}' not found. Creating synthetic targets.")

        # Create target based on MongoDB data characteristics
        # This is just a heuristic approach - in real scenarios, you would have labeled data

        # Method 1: Check for known APT indicators
        if 'Stage' in data.columns:
            # If 'Stage' column exists, use it to mark 'exfiltration' as APT
            data[target_col] = data['Stage'].str.contains('exfiltration', case=False, na=False).astype(int)
            logger.info(f"Created target column based on 'Stage' field. Found {data[target_col].sum()} potential APTs.")

        # Method 2: Use statistical approach to identify anomalies
        else:
            # Find columns related to flow/packet characteristics
            flow_cols = [col for col in data.columns if any(term in str(col) for term in
                                                            ['Flow', 'Packet', 'flag', 'Duration'])]

            if flow_cols:
                # Calculate z-scores for these columns
                z_scores = pd.DataFrame()
                for col in flow_cols:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        z_scores[col] = (data[col] - data[col].mean()) / data[col].std()

                # Mark as APT if multiple metrics are outliers
                outlier_count = (abs(z_scores) > 2.5).sum(axis=1)
                data[target_col] = (outlier_count >= 3).astype(int)

                # Ensure reasonable class balance (not too imbalanced)
                apt_count = data[target_col].sum()
                target_apt_rate = 0.05  # 5% APT rate

                if apt_count / len(data) > 0.1:
                    # Too many marked as APT, increase threshold
                    threshold = np.percentile(outlier_count, 95)
                    data[target_col] = (outlier_count >= threshold).astype(int)
                elif apt_count == 0:
                    # No APTs found, mark the most anomalous as APTs
                    threshold = np.percentile(outlier_count, 100 - target_apt_rate * 100)
                    data[target_col] = (outlier_count >= threshold).astype(int)

                logger.info(
                    f"Created target column based on statistical approach. Found {data[target_col].sum()} potential APTs.")
            else:
                # No flow columns found, create random labels for demonstration
                logger.warning("No suitable columns found for automatic labeling. Creating random labels.")
                data[target_col] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])

    # Preprocess the data
    processed_data = preprocessor.preprocess_apt_data(data, target_col=target_col)

    # Log class distribution
    if target_col in processed_data.columns:
        class_counts = processed_data[target_col].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")

    return processed_data


def train_mode(args, detector):
    """
    Train the APT detection model

    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    detector : APTDetector
        Initialized APT detector
    """
    logger.info(f"Starting training mode")

    # Load data
    data = load_data(args, detector)
    if data is None:
        logger.error("Could not load training data")
        return

    # Preprocess data
    processed_data = preprocess_data(data, target_col=args.target)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize combined model
    model = CombinedAPTModel(args.config)

    # Train model
    logger.info(f"Training model with {len(processed_data)} samples")
    results = model.train(processed_data, target_col=args.target)

    # Save model
    model_dir = os.path.join(args.output, 'models')
    logger.info(f"Saving model to {model_dir}")
    model.save(model_dir)

    # Save training src
    results_file = os.path.join(args.output, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Generate visualizations
    viz_dir = os.path.join(args.output, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Plot training history
    if 'autoencoder_history' in results:
        history = results['autoencoder_history']

        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(viz_dir, 'autoencoder_training.png'))
        plt.close()

        # Plot feature importance
        if 'feature_importance' in results:
            feature_importance = results['feature_importance']
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())

            # Sort by importance
            sorted_idx = np.argsort(importance)
            top_n = min(20, len(features))  # Show top 20 features or all if less

            plt.figure(figsize=(12, 8))
            plt.barh([features[i] for i in sorted_idx[-top_n:]],
                     [importance[i] for i in sorted_idx[-top_n:]])
            plt.title('Top Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
            plt.close()

        # Print summary of src
        logger.info(f"Training completed successfully")
        logger.info(f"Model performance:")

        if 'performance' in results and 'combined_model' in results['performance']:
            perf = results['performance']['combined_model']
            logger.info(f"  Accuracy: {perf.get('accuracy', 'N/A')}")
            logger.info(f"  Precision: {perf.get('precision', 'N/A')}")
            logger.info(f"  Recall: {perf.get('recall', 'N/A')}")
            logger.info(f"  F1 Score: {perf.get('f1', 'N/A')}")
            logger.info(f"  ROC AUC: {perf.get('roc_auc', 'N/A')}")
        else:
            logger.info(f"  Accuracy: {results.get('accuracy', 'N/A')}")
            logger.info(f"  Precision: {results.get('precision', 'N/A')}")
            logger.info(f"  Recall: {results.get('recall', 'N/A')}")
            logger.info(f"  F1 Score: {results.get('f1', 'N/A')}")

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Visualizations saved to {viz_dir}")

    def predict_mode(args, detector):
        """
        Use the trained model to make predictions

        Parameters:
        -----------
        args : argparse.Namespace
            Command-line arguments
        detector : APTDetector
            Initialized APT detector
        """
        logger.info(f"Starting prediction mode")

        # Load data
        data = load_data(args, detector)
        if data is None:
            logger.error("Could not load data for prediction")
            return

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Initialize combined model
        model = CombinedAPTModel(args.config)

        # Load the trained model
        logger.info("Loading trained model")
        model_loaded = model.load()

        if not model_loaded:
            logger.error("Failed to load model. Please train the model first.")
            return

        # Make predictions
        logger.info(f"Making predictions on {len(data)} samples")
        prediction_results = model.predict(data, return_details=True)

        # Extract prediction details
        predictions = prediction_results['predictions']
        probabilities = prediction_results['probabilities']
        reconstruction_errors = prediction_results['reconstruction_errors']

        # Add predictions to the data
        result_df = data.copy()
        result_df['apt_prediction'] = predictions
        result_df['apt_probability'] = probabilities
        result_df['reconstruction_error'] = reconstruction_errors
        result_df['detection_time'] = datetime.now()

        # Save predictions to CSV
        output_file = os.path.join(args.output, 'apt_predictions.csv')
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")

        # Generate visualizations
        viz_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Plot probability distribution
        plt.figure(figsize=(10, 6))
        plt.hist(probabilities, bins=50, alpha=0.7)
        plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
        plt.title('APT Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(viz_dir, 'probability_distribution.png'))
        plt.close()

        # Plot reconstruction error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(reconstruction_errors, bins=50, alpha=0.7)
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.savefig(os.path.join(viz_dir, 'reconstruction_error_distribution.png'))
        plt.close()

        # Check if we have ground truth for evaluation
        if args.target in result_df.columns:
            from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

            y_true = result_df[args.target].values

            # Generate evaluation metrics
            report = classification_report(y_true, predictions, output_dict=True)

            # Save metrics
            metrics_file = os.path.join(args.output, 'prediction_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(report, f, indent=4)

            # Plot confusion matrix
            cm = confusion_matrix(y_true, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'APT'],
                        yticklabels=['Normal', 'APT'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'))
            plt.close()

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_true, probabilities)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(viz_dir, 'roc_curve.png'))
            plt.close()

            # Print metrics
            logger.info("Evaluation metrics:")
            logger.info(f"  Accuracy: {report['accuracy']:.4f}")
            logger.info(f"  Precision: {report['1']['precision']:.4f}")
            logger.info(f"  Recall: {report['1']['recall']:.4f}")
            logger.info(f"  F1 Score: {report['1']['f1-score']:.4f}")
            logger.info(f"  ROC AUC: {roc_auc:.4f}")

        # Save predictions to MongoDB if requested
        save_to_mongo = input("Save predictions back to MongoDB? (y/n): ").strip().lower() == 'y'

        if save_to_mongo and detector.mongo_client:
            try:
                # Create a new collection for predictions
                prediction_collection = 'apt_predictions'
                collection = detector.mongodb[prediction_collection]

                # Convert DataFrame to records
                records = result_df.to_dict(orient='records')

                # Insert into MongoDB
                result = collection.insert_many(records)

                logger.info(
                    f"Saved {len(result.inserted_ids)} predictions to MongoDB collection '{prediction_collection}'")

            except Exception as e:
                logger.error(f"Error saving predictions to MongoDB: {str(e)}")

        # Summary
        apt_count = np.sum(predictions)
        logger.info(
            f"Prediction summary: Found {apt_count} potential APTs out of {len(data)} samples ({apt_count / len(data):.2%})")

    def compare_mode(args, detector):
        """
        Compare ML approach with SIEM rules

        Parameters:
        -----------
        args : argparse.Namespace
            Command-line arguments
        detector : APTDetector
            Initialized APT detector
        """
        logger.info(f"Starting comparison mode")

        # Load data
        data = load_data(args, detector)
        if data is None:
            logger.error("Could not load data for comparison")
            return

        # Check for SIEM detection column
        siem_col = 'siem_detection'
        if siem_col not in data.columns:
            logger.error(f"SIEM detection column '{siem_col}' not found in data")
            logger.info("Generating synthetic SIEM detections for demonstration")

            # Generate synthetic SIEM detections with specific properties:
            # - Good at detecting known patterns (high precision)
            # - May miss novel attacks (lower recall)
            # - No probabilistic output (binary decisions)

            # Create synthetic SIEM detections
            np.random.seed(42)  # For reproducibility

            # If we have ground truth
            if args.target in data.columns:
                y_true = data[args.target].values

                # SIEM is good at finding some APTs (70% of them)
                apt_indices = np.where(y_true == 1)[0]
                detected_indices = np.random.choice(apt_indices, size=int(len(apt_indices) * 0.7), replace=False)

                # SIEM has some false positives (about 10%)
                normal_indices = np.where(y_true == 0)[0]
                false_positive_indices = np.random.choice(normal_indices, size=int(len(normal_indices) * 0.1),
                                                          replace=False)

                # Create SIEM detections
                siem_detections = np.zeros(len(data))
                siem_detections[detected_indices] = 1
                siem_detections[false_positive_indices] = 1

            else:
                # No ground truth, create random detections
                # Assume 5% detection rate
                siem_detections = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])

            # Add to data
            data[siem_col] = siem_detections

            # Add detection times (SIEM is usually faster for known patterns)
            now = datetime.now()
            data['siem_detection_time'] = now

            logger.info(f"Created synthetic SIEM detections with {np.sum(siem_detections)} alerts")

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Initialize combined model
        model = CombinedAPTModel(args.config)

        # Load the trained model
        logger.info("Loading trained model")
        model_loaded = model.load()

        if not model_loaded:
            logger.error("Failed to load model. Please train the model first.")
            return

        # Run comparison
        logger.info("Comparing ML approach with SIEM rules")
        comparison = model.compare_with_siem(
            data,
            siem_col=siem_col,
            target_col=args.target,
            save_path=os.path.join(args.output, 'visualizations')
        )

        # Save comparison src
        comparison_file = os.path.join(args.output, 'siem_ml_comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=4)

        # Print comparison summary
        logger.info("Comparison summary:")

        if 'ml' in comparison and 'metrics' in comparison['ml']:
            ml_metrics = comparison['ml']['metrics']
            siem_metrics = comparison['siem']['metrics']

            logger.info("Performance metrics:")
            logger.info(f"  Metric    | SIEM Rules | ML Approach | Difference")
            logger.info(f"  ----------|------------|-------------|------------")
            logger.info(
                f"  Accuracy  | {siem_metrics['accuracy']:.4f}     | {ml_metrics['accuracy']:.4f}      | {ml_metrics['accuracy'] - siem_metrics['accuracy']:.4f}")
            logger.info(
                f"  Precision | {siem_metrics['precision']:.4f}     | {ml_metrics['precision']:.4f}      | {ml_metrics['precision'] - siem_metrics['precision']:.4f}")
            logger.info(
                f"  Recall    | {siem_metrics['recall']:.4f}     | {ml_metrics['recall']:.4f}      | {ml_metrics['recall'] - siem_metrics['recall']:.4f}")
            logger.info(
                f"  F1 Score  | {siem_metrics['f1']:.4f}     | {ml_metrics['f1']:.4f}      | {ml_metrics['f1'] - siem_metrics['f1']:.4f}")

        if 'time_difference' in comparison and comparison['time_difference']:
            time_diff = comparison['time_difference']
            logger.info("Detection timing comparison:")
            logger.info(f"  Mean time difference: {time_diff['mean']:.2f} seconds")
            logger.info(f"  ML faster: {time_diff['ml_faster_count']} cases")
            logger.info(f"  SIEM faster: {time_diff['siem_faster_count']} cases")

        logger.info(f"Detailed comparison saved to {comparison_file}")

    def monitor_mode(args, detector):
        """
        Monitor and adapt to evolving threats

        Parameters:
        -----------
        args : argparse.Namespace
            Command-line arguments
        detector : APTDetector
            Initialized APT detector
        """
        logger.info(f"Starting monitoring mode")

        # Load data
        data = load_data(args, detector)
        if data is None:
            logger.error("Could not load data for monitoring")
            return

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Split data into batches to simulate streaming data
        batch_size = 200  # Small batch size for demonstration
        n_batches = max(5, len(data) // batch_size)  # At least 5 batches

        logger.info(f"Splitting data into {n_batches} batches for simulated monitoring")

        # Create a data generator for streaming
        def data_stream():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(data))

                # For demonstration, inject some evolving patterns in later batches
                batch = data.iloc[start_idx:end_idx].copy()

                # If we're in the second half of batches, modify some patterns
                # This simulates evolving APT tactics
                if i >= n_batches // 2 and args.target in batch.columns:
                    apt_indices = batch[batch[args.target] == 1].index

                    # Modify some numeric features to simulate changing patterns
                    for col in batch.select_dtypes(include=['number']).columns:
                        if col != args.target:
                            # Shift the distribution for APTs
                            batch.loc[apt_indices, col] *= (1 + (i - n_batches // 2) * 0.1)

                yield batch

        # Initialize combined model
        model = CombinedAPTModel(args.config)

        # Load the trained model
        logger.info("Loading trained model")
        model_loaded = model.load()

        if not model_loaded:
            logger.error("Failed to load model. Please train the model first.")
            return

        # Create monitoring function to replace default implementation
        # for demonstration purposes
        def monitor_adapt_fn(data_stream, target_col, window_size=500, threshold=0.1):
            """Custom monitoring function for demonstration"""
            performance_history = []
            adaptations = []
            baseline_f1 = None

            # Handle each batch
            for batch_idx, batch in enumerate(data_stream()):
                # Check for target column
                if target_col not in batch.columns:
                    logger.error(f"Target column {target_col} not found in batch {batch_idx}")
                    continue

                # Get ground truth
                y_true = batch[target_col].values

                # Make predictions
                prediction_results = model.predict(batch, return_details=True)
                y_pred = prediction_results['predictions']

                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                # Record performance
                batch_perf = {
                    'batch': batch_idx,
                    'samples': len(batch),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                }

                performance_history.append(batch_perf)

                # Set baseline if not set
                if baseline_f1 is None:
                    baseline_f1 = f1
                    logger.info(f"Initial F1 score: {baseline_f1:.4f}")

                # Check for performance degradation
                if f1 < baseline_f1 - threshold:
                    logger.warning(
                        f"Performance drop detected in batch {batch_idx}: F1={f1:.4f} vs baseline={baseline_f1:.4f}")

                    # Retrain model with this batch
                    logger.info(f"Adapting model with {len(batch)} samples from batch {batch_idx}")

                    adaptation_start = datetime.now()
                    results = model.train(batch, target_col=target_col)
                    adaptation_time = (datetime.now() - adaptation_start).total_seconds()

                    # Record adaptation
                    adapt_info = {
                        'batch': batch_idx,
                        'samples': len(batch),
                        'previous_f1': float(baseline_f1),
                        'new_f1': float(results['performance']['combined_model']['f1']),
                        'adaptation_time_seconds': float(adaptation_time)
                    }

                    adaptations.append(adapt_info)

                    # Update baseline
                    baseline_f1 = results['performance']['combined_model']['f1']
                    logger.info(f"Model adapted. New F1 score: {baseline_f1:.4f}")

                # Log progress
                logger.info(f"Processed batch {batch_idx + 1}/{n_batches}. F1 score: {f1:.4f}")

            # Compile src
            monitoring_results = {
                'performance_history': performance_history,
                'adaptations': adaptations,
                'total_batches': len(performance_history)
            }

            return monitoring_results

        # Run monitoring
        logger.info("Starting model monitoring and adaptation")
        monitoring_results = monitor_adapt_fn(data_stream, args.target)

        # Save monitoring src
        monitoring_file = os.path.join(args.output, 'monitoring_results.json')
        with open(monitoring_file, 'w') as f:
            json.dump(monitoring_results, f, indent=4)

        # Visualize performance over time
        viz_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Plot performance metrics over time
        performance_history = monitoring_results['performance_history']
        batches = [item['batch'] for item in performance_history]
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        plt.figure(figsize=(12, 8))
        for metric in metrics:
            values = [item[metric] for item in performance_history]
            plt.plot(batches, values, marker='o', label=metric.capitalize())

        # Add adaptation points
        if monitoring_results['adaptations']:
            for adaptation in monitoring_results['adaptations']:
                plt.axvline(x=adaptation['batch'], color='r', linestyle='--', alpha=0.5)
                plt.annotate('Adaptation',
                             xy=(adaptation['batch'], 0.5),
                             xytext=(adaptation['batch'] + 0.2, 0.6),
                             arrowprops=dict(arrowstyle='->'))

        plt.xlabel('Batch')
        plt.ylabel('Score')
        plt.title('Model Performance Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'performance_over_time.png'))
        plt.close()

        # Print monitoring summary
        logger.info("Monitoring summary:")
        logger.info(f"  Total batches processed: {monitoring_results['total_batches']}")
        logger.info(f"  Model adaptations performed: {len(monitoring_results['adaptations'])}")

        if monitoring_results['adaptations']:
            logger.info("  Adaptation details:")
            for i, adaptation in enumerate(monitoring_results['adaptations']):
                logger.info(
                    f"    Adaptation {i + 1}: Batch {adaptation['batch']}, F1 improvement: {adaptation['new_f1'] - adaptation['previous_f1']:.4f}")

        logger.info(f"Monitoring src saved to {monitoring_file}")

    def main():
        """Main function"""
        # Parse arguments
        args = parse_arguments()

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Setup MongoDB connection
        detector = setup_mongodb_connection(args.config, args.collection)

        # Execute the requested mode
        if args.mode == 'train':
            train_mode(args, detector)
        elif args.mode == 'predict':
            predict_mode(args, detector)
        elif args.mode == 'compare':
            compare_mode(args, detector)
        elif args.mode == 'monitor':
            monitor_mode(args, detector)
        else:
            logger.error(f"Invalid mode: {args.mode}")

    if __name__ == "__main__":
        main()