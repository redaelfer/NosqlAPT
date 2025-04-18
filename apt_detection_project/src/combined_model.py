"""
APT Detection System - Combined Autoencoder and XGBoost Model

This script implements a hybrid APT detection system using:
1. Autoencoder for anomaly detection and feature extraction
2. XGBoost for classification of normal vs. APT traffic

The integration provides both unsupervised anomaly detection capabilities 
and supervised classification for effective APT identification.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("apt_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('apt_detection')


class CombinedAPTModel:
    """
    Combined Autoencoder and XGBoost model for APT detection
    """

    def __init__(self, config_path=None):
        """
        Initialize the model

        Parameters:
        -----------
        config_path : str, optional
            Path to model configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.autoencoder = None
        self.encoder = None
        self.xgb_model = None
        self.feature_names = None

        # Create model directories
        os.makedirs(self.config['model_paths']['base_dir'], exist_ok=True)

    def _load_config(self, config_path):
        """
        Load model configuration

        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file

        Returns:
        --------
        dict
            Model configuration
        """
        # Default configuration
        default_config = {
            'model_paths': {
                'base_dir': 'models',
                'autoencoder': 'models/autoencoder_model.h5',
                'encoder': 'models/encoder_model.h5',
                'xgboost': 'models/xgboost_model.json',
                'preprocessor': 'models/preprocessor.pkl'
            },
            'autoencoder': {
                'architecture': [128, 64, 32, 16],  # Layer sizes from input to bottleneck
                'activation': 'relu',
                'bottleneck_activation': 'relu',
                'output_activation': 'sigmoid',
                'dropout_rate': 0.2,
                'use_batch_norm': True,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 50,
                'patience': 5,
                'validation_split': 0.2
            },
            'xgboost': {
                'params': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'eta': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'alpha': 0.1,
                    'lambda': 1.0,
                    'min_child_weight': 1,
                    'silent': 1,
                    'scale_pos_weight': 1,
                    'tree_method': 'hist'
                },
                'num_boost_round': 100,
                'early_stopping_rounds': 10
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'use_reconstruction_error': True,  # Include reconstruction error as a feature
                'use_encoded_features': True,  # Include encoded features for XGBoost
                'anomaly_threshold_percentile': 95,  # Percentile for anomaly threshold
                'use_smote': True  # Use SMOTE for handling class imbalance
            },
            'inference': {
                'classification_threshold': 0.5,  # Threshold for XGBoost predictions
                'batch_size': 1024  # Batch size for making predictions
            }
        }

        # Load user configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)

                # Recursively update the default config
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and k in d:
                            d[k] = update_dict(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d

                default_config = update_dict(default_config, user_config)

        return default_config

    def save_config(self, path=None):
        """
        Save the current configuration

        Parameters:
        -----------
        path : str, optional
            Path to save configuration
        """
        if path is None:
            path = os.path.join(self.config['model_paths']['base_dir'], 'config.json')

        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def build_autoencoder(self, input_dim):
        """
        Build the autoencoder model

        Parameters:
        -----------
        input_dim : int
            Input dimension (number of features)

        Returns:
        --------
        tuple
            (autoencoder, encoder) models
        """
        # Get autoencoder configuration
        ae_config = self.config['autoencoder']

        # Determine architecture
        encoder_layers = ae_config['architecture']
        decoder_layers = encoder_layers[:-1][::-1]  # Reverse and exclude bottleneck
        bottleneck_dim = encoder_layers[-1]

        # Input layer
        input_layer = Input(shape=(input_dim,))

        # Build encoder
        x = input_layer
        for i, units in enumerate(encoder_layers[:-1]):  # Exclude bottleneck
            x = Dense(units, activation=ae_config['activation'])(x)

            if ae_config['use_batch_norm']:
                x = BatchNormalization()(x)

            if ae_config['dropout_rate'] > 0:
                x = Dropout(ae_config['dropout_rate'])(x)

        # Bottleneck layer
        bottleneck = Dense(bottleneck_dim,
                           activation=ae_config['bottleneck_activation'],
                           name='bottleneck')(x)

        # Build decoder
        x = bottleneck
        for i, units in enumerate(decoder_layers):
            x = Dense(units, activation=ae_config['activation'])(x)

            if ae_config['use_batch_norm']:
                x = BatchNormalization()(x)

            if ae_config['dropout_rate'] > 0 and i < len(decoder_layers) - 1:  # No dropout on last layer
                x = Dropout(ae_config['dropout_rate'])(x)

        # Output layer
        output_layer = Dense(input_dim, activation=ae_config['output_activation'])(x)

        # Create models
        autoencoder = Model(inputs=input_layer, outputs=output_layer, name='autoencoder')
        encoder = Model(inputs=input_layer, outputs=bottleneck, name='encoder')

        # Compile autoencoder
        autoencoder.compile(
            optimizer=Adam(learning_rate=ae_config['learning_rate']),
            loss='mean_squared_error'
        )

        logger.info(f"Built autoencoder with architecture: {encoder_layers + decoder_layers[::-1]}")
        logger.info(f"Bottleneck dimension: {bottleneck_dim}")

        return autoencoder, encoder

    def train_autoencoder(self, X_train):
        """
        Train the autoencoder model

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data

        Returns:
        --------
        dict
            Training history
        """
        # Ensure data is in the right format for TensorFlow
        X_train = np.array(X_train, dtype=np.float32)
        # Get autoencoder configuration
        ae_config = self.config['autoencoder']

        # Split validation data
        val_split = ae_config['validation_split']
        if val_split > 0:
            X_train_ae, X_val_ae = train_test_split(
                X_train,
                test_size=val_split,
                random_state=self.config['training']['random_state']
            )
        else:
            X_train_ae, X_val_ae = X_train, None

        # Build models if they don't exist
        if self.autoencoder is None:
            self.autoencoder, self.encoder = self.build_autoencoder(X_train.shape[1])

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val_ae is not None else 'loss',
                patience=ae_config['patience'],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=self.config['model_paths']['autoencoder'],
                monitor='val_loss' if X_val_ae is not None else 'loss',
                save_best_only=True
            )
        ]

        # Train the model
        history = self.autoencoder.fit(
            X_train_ae, X_train_ae,  # Autoencoder tries to reconstruct input
            epochs=ae_config['epochs'],
            batch_size=ae_config['batch_size'],
            validation_data=(X_val_ae, X_val_ae) if X_val_ae is not None else None,
            callbacks=callbacks,
            verbose=1
        )

        # Save the encoder separately
        self.encoder.save(self.config['model_paths']['encoder'])

        logger.info(f"Autoencoder training complete. Final loss: {history.history['loss'][-1]:.4f}")
        if X_val_ae is not None:
            logger.info(f"Validation loss: {history.history['val_loss'][-1]:.4f}")

        return history.history

    def compute_reconstruction_error(self, X):
        """
        Compute reconstruction error for input data

        Parameters:
        -----------
        X : numpy.ndarray
            Input data

        Returns:
        --------
        numpy.ndarray
            Reconstruction error for each sample
        """
        # Ensure autoencoder is loaded
        if self.autoencoder is None:
            self.load_autoencoder()

        # Get reconstructions
        X_pred = self.autoencoder.predict(X, batch_size=self.config['inference']['batch_size'])

        # Calculate MSE for each sample
        reconstruction_error = np.mean(np.square(X - X_pred), axis=1)

        return reconstruction_error

    def compute_anomaly_threshold(self, reconstruction_errors):
        """
        Compute anomaly threshold based on reconstruction errors

        Parameters:
        -----------
        reconstruction_errors : numpy.ndarray
            Reconstruction errors

        Returns:
        --------
        float
            Anomaly threshold
        """
        # Use percentile method to determine threshold
        threshold_percentile = self.config['training']['anomaly_threshold_percentile']
        threshold = np.percentile(reconstruction_errors, threshold_percentile)

        logger.info(f"Computed anomaly threshold: {threshold:.6f} (percentile: {threshold_percentile})")

        return threshold

    def encode_features(self, X):
        """
        Encode features using the autoencoder's bottleneck layer

        Parameters:
        -----------
        X : numpy.ndarray
            Input data

        Returns:
        --------
        numpy.ndarray
            Encoded features
        """
        # Ensure encoder is loaded
        if self.encoder is None:
            self.load_autoencoder()

        # Get encoded features
        encoded_features = self.encoder.predict(X, batch_size=self.config['inference']['batch_size'])

        return encoded_features

    def prepare_xgboost_features(self, X, include_reconstruction=None, include_encoded=None):
        """
        Prepare features for XGBoost, including autoencoder-derived features

        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        include_reconstruction : bool, optional
            Whether to include reconstruction error, defaults to config setting
        include_encoded : bool, optional
            Whether to include encoded features, defaults to config setting

        Returns:
        --------
        numpy.ndarray
            Features for XGBoost
        """
        # Use configuration defaults if not specified
        if include_reconstruction is None:
            include_reconstruction = self.config['training']['use_reconstruction_error']

        if include_encoded is None:
            include_encoded = self.config['training']['use_encoded_features']

        # Start with original features
        features_list = [X]

        # Add reconstruction error if requested
        if include_reconstruction:
            reconstruction_errors = self.compute_reconstruction_error(X)
            features_list.append(reconstruction_errors.reshape(-1, 1))

        # Add encoded features if requested
        if include_encoded:
            encoded_features = self.encode_features(X)
            features_list.append(encoded_features)

        # Combine all features
        if len(features_list) > 1:
            combined_features = np.hstack(features_list)
        else:
            combined_features = features_list[0]

        return combined_features

    def train_xgboost(self, X, y):
        """
        Train the XGBoost model

        Parameters:
        -----------
        X : numpy.ndarray
            Features for XGBoost
        y : numpy.ndarray
            Target labels

        Returns:
        --------
        xgb.Booster
            Trained XGBoost model
        """
        # Handle class imbalance if needed
        if self.config['training']['use_smote'] and len(np.unique(y)) > 1:
            try:
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=self.config['training']['random_state'])
                X_resampled, y_resampled = smote.fit_resample(X, y)

                logger.info(f"Applied SMOTE resampling. Original class counts: {np.bincount(y)}")
                logger.info(f"Resampled class counts: {np.bincount(y_resampled)}")

                X, y = X_resampled, y_resampled
            except ImportError:
                logger.warning("imblearn not installed. SMOTE resampling skipped.")

        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )

        # Convert to DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Get XGBoost parameters
        params = self.config['xgboost']['params']
        num_boost_round = self.config['xgboost']['num_boost_round']
        early_stopping_rounds = self.config['xgboost']['early_stopping_rounds']

        # Train model
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )

        # Save model
        self.xgb_model.save_model(self.config['model_paths']['xgboost'])

        # Get best iteration and score
        best_iteration = self.xgb_model.best_iteration
        best_score = self.xgb_model.best_score

        logger.info(f"XGBoost training complete. Best iteration: {best_iteration}, Score: {best_score:.4f}")

        # Evaluate on validation set
        y_pred = self.xgb_model.predict(dval)
        y_pred_binary = (y_pred > self.config['inference']['classification_threshold']).astype(int)

        # Print classification report
        report = classification_report(y_val, y_pred_binary)
        logger.info(f"Validation set performance:\n{report}")

        return self.xgb_model

    def train(self, data, target_col='is_apt'):
        """
        Train the complete model (Autoencoder + XGBoost)

        Parameters:
        -----------
        data : pandas.DataFrame
            Training data
        target_col : str, default='is_apt'
            Name of the target column

        Returns:
        --------
        dict
            Training src
        """
        logger.info(f"Starting training with {len(data)} samples")

        # Import data preprocessor if not already done
        try:
            from apt_data_prep import APTDataPreprocessor
            preprocessor = APTDataPreprocessor()
        except ImportError:
            # If module not found, use basic preprocessing
            logger.warning("apt_data_prep module not found. Using basic preprocessing.")
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline

            # Create a basic preprocessor
            class BasicPreprocessor:
                def __init__(self):
                    self.feature_names = None
                    self.pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ])

                def preprocess_apt_data(self, df, target_col=None):
                    df = df.copy()

                    # Store target if provided
                    target = None
                    if target_col in df.columns:
                        target = df[target_col].copy()
                        df = df.drop(columns=[target_col])

                    # Drop non-numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df = df[numeric_cols]

                    # Store feature names
                    self.feature_names = df.columns.tolist()

                    # Apply preprocessing
                    df[self.feature_names] = self.pipeline.fit_transform(df[self.feature_names])

                    # Add target back
                    if target is not None:
                        df[target_col] = target

                    return df

            preprocessor = BasicPreprocessor()

        # Preprocess data
        logger.info("Preprocessing data...")
        if target_col in data.columns:
            # Save target
            target = data[target_col].copy()
            data_features = data.drop(columns=[target_col])

            # Apply preprocessing steps
            data_features = preprocessor.handle_missing_values(data_features)
            data_features = preprocessor.handle_timestamps(data_features)
            data_features = preprocessor.handle_ip_addresses(data_features)
            data_features = preprocessor.handle_ports_and_protocols(data_features)
            data_features = preprocessor.normalize_features(data_features, fit=True)
            data_features = preprocessor.encode_categorical(data_features)

            # Add target back
            data_features[target_col] = target

            # Use this as our processed data
            processed_data = data_features
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Store feature names
        self.feature_names = processed_data.columns.tolist()
        if target_col in self.feature_names:
            self.feature_names.remove(target_col)

        # Extract features and target
        X = processed_data[self.feature_names].values
        y = processed_data[target_col].values

        # Save preprocessor
        preprocessor_path = self.config['model_paths']['preprocessor']
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)

        # Train autoencoder
        logger.info("Training autoencoder...")
        # Extract features and target
        X = processed_data[self.feature_names].values
        y = processed_data[target_col].values

        # Ensure data is in the right format for TensorFlow
        X = X.astype(np.float32)  # Convert to float32 for TensorFlow
        autoencoder_history = self.train_autoencoder(X)

        # Prepare features for XGBoost
        logger.info("Preparing features for XGBoost...")
        X_xgb = self.prepare_xgboost_features(X)

        # Train XGBoost
        logger.info("Training XGBoost...")
        self.train_xgboost(X_xgb, y)

        # Calculate metrics on the whole dataset
        X_xgb_full = self.prepare_xgboost_features(X)
        dmatrix = xgb.DMatrix(X_xgb_full)
        y_pred_proba = self.xgb_model.predict(dmatrix)
        y_pred = (y_pred_proba > self.config['inference']['classification_threshold']).astype(int)

        # Calculate reconstruction errors
        reconstruction_errors = self.compute_reconstruction_error(X)
        anomaly_threshold = self.compute_anomaly_threshold(reconstruction_errors)

        # Calculate autoencoder-only anomaly detection (for comparison)
        y_pred_ae = (reconstruction_errors > anomaly_threshold).astype(int)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # Return training src
        results = {
            'autoencoder_history': autoencoder_history,
            'feature_importance': self._get_feature_importance(),
            'reconstruction_errors': {
                'mean': float(np.mean(reconstruction_errors)),
                'std': float(np.std(reconstruction_errors)),
                'min': float(np.min(reconstruction_errors)),
                'max': float(np.max(reconstruction_errors)),
                'anomaly_threshold': float(anomaly_threshold)
            },
            'performance': {
                'combined_model': {
                    'accuracy': float(accuracy_score(y, y_pred)),
                    'precision': float(precision_score(y, y_pred)),
                    'recall': float(recall_score(y, y_pred)),
                    'f1': float(f1_score(y, y_pred)),
                    'roc_auc': float(roc_auc_score(y, y_pred_proba))
                },
                'autoencoder_only': {
                    'accuracy': float(accuracy_score(y, y_pred_ae)),
                    'precision': float(precision_score(y, y_pred_ae)),
                    'recall': float(recall_score(y, y_pred_ae)),
                    'f1': float(f1_score(y, y_pred_ae))
                }
            }
        }

        logger.info("Training complete")

        return results

    def predict(self, data, return_details=False):
        """
        Make predictions using the trained model

        Parameters:
        -----------
        data : pandas.DataFrame
            Data to predict
        return_details : bool, default=False
            Whether to return detailed prediction information

        Returns:
        --------
        numpy.ndarray or dict
            Predictions or detailed prediction information
        """
        logger.info(f"Making predictions for {len(data)} samples")

        # Load preprocessor
        try:
            preprocessor_path = self.config['model_paths']['preprocessor']
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

            # Preprocess data
            logger.info("Preprocessing data...")
            processed_data = preprocessor.preprocess_apt_data(data)
        except:
            logger.warning("Error loading preprocessor or preprocessing data. Using raw data.")
            processed_data = data

        # Get features
        if self.feature_names:
            # Use stored feature names
            available_features = [f for f in self.feature_names if f in processed_data.columns]
            if len(available_features) < len(self.feature_names):
                logger.warning(
                    f"Some features are missing. Using {len(available_features)} of {len(self.feature_names)} features")

            X = processed_data[available_features].values
        else:
            # Use all numeric features
            numeric_cols = processed_data.select_dtypes(include=['number']).columns
            logger.warning(f"No stored feature names. Using all {len(numeric_cols)} numeric features")
            X = processed_data[numeric_cols].values

        # Prepare features for XGBoost
        X_xgb = self.prepare_xgboost_features(X)

        # Make predictions
        dmatrix = xgb.DMatrix(X_xgb)
        y_pred_proba = self.xgb_model.predict(dmatrix)
        y_pred = (y_pred_proba > self.config['inference']['classification_threshold']).astype(int)

        # Calculate reconstruction errors
        reconstruction_errors = self.compute_reconstruction_error(X)

        if return_details:
            # Return detailed prediction information
            return {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'reconstruction_errors': reconstruction_errors,
                'encoded_features': self.encode_features(X)
            }
        else:
            # Return just the predictions
            return y_pred

    def _get_feature_importance(self):
        """
        Get feature importance from XGBoost model

        Returns:
        --------
        dict
            Feature importance
        """
        if self.xgb_model is None:
            logger.warning("XGBoost model not trained, cannot get feature importance")
            return {}

        # Get feature scores
        importance = self.xgb_model.get_score(importance_type='gain')

        # Convert feature indices to names
        renamed_importance = {}

        if self.feature_names:
            # Original features
            orig_feature_count = len(self.feature_names)

            for feature_idx, score in importance.items():
                # Convert from f0, f1, etc. to actual feature names
                idx = int(feature_idx.replace('f', ''))

                if idx < orig_feature_count:
                    # Original feature
                    feature_name = self.feature_names[idx]
                elif idx == orig_feature_count:
                    # Reconstruction error
                    feature_name = 'reconstruction_error'
                else:
                    # Encoded feature
                    enc_idx = idx - orig_feature_count - 1
                    feature_name = f'encoded_{enc_idx}'

                renamed_importance[feature_name] = float(score)
        else:
            # No feature names, keep indices
            for feature_idx, score in importance.items():
                renamed_importance[feature_idx] = float(score)

        return renamed_importance

    def save(self, base_path=None):
        """
        Save the complete model

        Parameters:
        -----------
        base_path : str, optional
            Base path for saving models
        """
        if base_path:
            # Update model paths
            self.config['model_paths']['base_dir'] = base_path
            self.config['model_paths']['autoencoder'] = os.path.join(base_path, 'autoencoder_model.h5')
            self.config['model_paths']['encoder'] = os.path.join(base_path, 'encoder_model.h5')
            self.config['model_paths']['xgboost'] = os.path.join(base_path, 'xgboost_model.json')
            self.config['model_paths']['preprocessor'] = os.path.join(base_path, 'preprocessor.pkl')

        # Create directory
        os.makedirs(self.config['model_paths']['base_dir'], exist_ok=True)

        # Save configuration
        self.save_config()

        # Save autoencoder if available
        if self.autoencoder:
            self.autoencoder.save(self.config['model_paths']['autoencoder'])
            logger.info(f"Saved autoencoder to {self.config['model_paths']['autoencoder']}")

        # Save encoder if available
        if self.encoder:
            self.encoder.save(self.config['model_paths']['encoder'])
            logger.info(f"Saved encoder to {self.config['model_paths']['encoder']}")

        # Save XGBoost model if available
        if self.xgb_model:
            self.xgb_model.save_model(self.config['model_paths']['xgboost'])
            logger.info(f"Saved XGBoost model to {self.config['model_paths']['xgboost']}")

    def load(self, config_path=None):
        """
        Load the complete model

        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file

        Returns:
        --------
        bool
            Whether all components were loaded successfully
        """
        # Update configuration if provided
        if config_path:
            self.config = self._load_config(config_path)

        # Load autoencoder and encoder
        try:
            autoencoder_path = self.config['model_paths']['autoencoder']
            if os.path.exists(autoencoder_path):
                self.autoencoder = load_model(autoencoder_path)
                logger.info(f"Loaded autoencoder from {autoencoder_path}")
            else:
                logger.warning(f"Autoencoder model not found at {autoencoder_path}")
                return False

            # Load encoder
            encoder_path = self.config['model_paths']['encoder']
            if os.path.exists(encoder_path):
                self.encoder = load_model(encoder_path)
                logger.info(f"Loaded encoder from {encoder_path}")
            else:
                # Try to extract encoder from autoencoder
                logger.info("Attempting to extract encoder from autoencoder...")

                # Find bottleneck layer
                bottleneck_layer = None
                for layer in self.autoencoder.layers:
                    if layer.name == 'bottleneck':
                        bottleneck_layer = layer
                        break

                if bottleneck_layer:
                    # Create encoder model
                    self.encoder = Model(
                        inputs=self.autoencoder.input,
                        outputs=bottleneck_layer.output,
                        name='encoder'
                    )
                    logger.info("Successfully extracted encoder from autoencoder")
                else:
                    logger.warning("Could not find bottleneck layer in autoencoder")
                    return False
        except Exception as e:
            logger.error(f"Error loading autoencoder: {str(e)}")
            return False

        # Load XGBoost model
        try:
            xgboost_path = self.config['model_paths']['xgboost']

            if os.path.exists(xgboost_path):
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgboost_path)
                logger.info(f"Loaded XGBoost model from {xgboost_path}")
                return True
            else:
                logger.warning(f"XGBoost model not found at {xgboost_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            return False

    def visualize_results(self, X, y=None, save_path=None):
        """
        Visualize model src

        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray, optional
            True labels
        save_path : str, optional
            Path to save visualizations

        Returns:
        --------
        dict
            Visualization information
        """
        # Create directory for visualizations
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # Get predictions and details
        X_xgb = self.prepare_xgboost_features(X)
        dmatrix = xgb.DMatrix(X_xgb)
        y_pred_proba = self.xgb_model.predict(dmatrix)
        y_pred = (y_pred_proba > self.config['inference']['classification_threshold']).astype(int)

        # Calculate reconstruction errors
        reconstruction_errors = self.compute_reconstruction_error(X)

        # Visualization info
        viz_info = {
            'reconstruction_error': {
                'mean': float(np.mean(reconstruction_errors)),
                'std': float(np.std(reconstruction_errors)),
                'min': float(np.min(reconstruction_errors)),
                'max': float(np.max(reconstruction_errors))
            },
            'prediction_distribution': {
                'mean': float(np.mean(y_pred_proba)),
                'std': float(np.std(y_pred_proba)),
                'min': float(np.min(y_pred_proba)),
                'max': float(np.max(y_pred_proba))
            }
        }

        # Visualize reconstruction error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(reconstruction_errors, bins=50, alpha=0.7)
        plt.axvline(x=np.percentile(reconstruction_errors,
                                    self.config['training']['anomaly_threshold_percentile']),
                    color='r', linestyle='--', label='Anomaly Threshold')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.legend()

        if save_path:
            plt.savefig(os.path.join(save_path, 'reconstruction_error_dist.png'))
            plt.close()

        # Visualize prediction probability distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba, bins=50, alpha=0.7)
        plt.axvline(x=self.config['inference']['classification_threshold'],
                    color='r', linestyle='--', label='Classification Threshold')
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability of APT')
        plt.ylabel('Count')
        plt.legend()

        if save_path:
            plt.savefig(os.path.join(save_path, 'prediction_prob_dist.png'))
            plt.close()

        # If true labels are provided, visualize performance
        if y is not None:
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'APT'],
                        yticklabels=['Normal', 'APT'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            if save_path:
                plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
                plt.close()

            # ROC curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
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
            plt.legend(loc='lower right')

            if save_path:
                plt.savefig(os.path.join(save_path, 'roc_curve.png'))
                plt.close()

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            pr_auc = auc(recall, precision)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'PR curve (area = {pr_auc:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')

            if save_path:
                plt.savefig(os.path.join(save_path, 'pr_curve.png'))
                plt.close()

            # Reconstruction error vs prediction probability
            plt.figure(figsize=(10, 6))
            plt.scatter(reconstruction_errors, y_pred_proba, alpha=0.5, c=y, cmap='coolwarm')
            plt.axhline(y=self.config['inference']['classification_threshold'],
                        color='r', linestyle='--', label='Classification Threshold')
            plt.axvline(x=np.percentile(reconstruction_errors,
                                        self.config['training']['anomaly_threshold_percentile']),
                        color='g', linestyle='--', label='Anomaly Threshold')
            plt.colorbar(label='True Label')
            plt.title('Reconstruction Error vs Prediction Probability')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Probability of APT')
            plt.legend()

            if save_path:
                plt.savefig(os.path.join(save_path, 'error_vs_prob.png'))
                plt.close()

            # Add performance metrics to visualization info
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            viz_info['performance'] = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred)),
                'recall': float(recall_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred)),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc)
            }

        return viz_info

    def explain_prediction(self, sample_data, index=0):
        """
        Explain a prediction using feature contributions

        Parameters:
        -----------
        sample_data : pandas.DataFrame
            Data containing the sample to explain
        index : int, default=0
            Index of the sample to explain

        Returns:
        --------
        dict
            Explanation of the prediction
        """
        # Preprocess data
        try:
            # Load preprocessor
            preprocessor_path = self.config['model_paths']['preprocessor']
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

            processed_data = preprocessor.preprocess_apt_data(sample_data)
        except:
            logger.warning("Error loading preprocessor. Using raw data.")
            processed_data = sample_data

        # Get features
        if self.feature_names:
            available_features = [f for f in self.feature_names if f in processed_data.columns]
            X = processed_data[available_features].values
        else:
            numeric_cols = processed_data.select_dtypes(include=['number']).columns
            X = processed_data[numeric_cols].values
            available_features = numeric_cols.tolist()

        # Prepare features for XGBoost
        X_xgb = self.prepare_xgboost_features(X)

        # Make prediction
        dmatrix = xgb.DMatrix(X_xgb)
        prediction = self.xgb_model.predict(dmatrix)[0]

        # Get feature importances for this prediction
        try:
            import shap
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer.shap_values(X_xgb)

            # Get sample index
            sample_idx = min(index, X_xgb.shape[0] - 1)

            # Get feature contributions
            contributions = []
            feature_count = len(available_features)

            for i in range(X_xgb.shape[1]):
                if i < feature_count:
                    # Original feature
                    feature_name = available_features[i]
                elif i == feature_count:
                    # Reconstruction error
                    feature_name = 'reconstruction_error'
                else:
                    # Encoded feature
                    enc_idx = i - feature_count - 1
                    feature_name = f'encoded_{enc_idx}'

                contributions.append({
                    'feature': feature_name,
                    'value': float(X_xgb[sample_idx, i]),
                    'contribution': float(shap_values[sample_idx, i])
                })

            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

            # Create explanation
            explanation = {
                'predicted_probability': float(prediction),
                'predicted_class': int(prediction > self.config['inference']['classification_threshold']),
                'feature_contributions': contributions,
                'base_value': float(explainer.expected_value)
            }

        except ImportError:
            # If SHAP not available, use simpler approach with feature importance
            feature_importance = self._get_feature_importance()

            # Get prediction details
            reconstruction_error = self.compute_reconstruction_error(X)[0]

            # Create basic explanation
            explanation = {
                'predicted_probability': float(prediction),
                'predicted_class': int(prediction > self.config['inference']['classification_threshold']),
                'reconstruction_error': float(reconstruction_error),
                'feature_importance': feature_importance
            }

        return explanation

    def compare_with_siem(self, data, siem_col='siem_detection', target_col='is_apt', save_path=None):
        """
        Compare ML approach with SIEM rules

        Parameters:
        -----------
        data : pandas.DataFrame
            Data with SIEM detection src and ground truth
        siem_col : str, default='siem_detection'
            Column with SIEM detection src
        target_col : str, default='is_apt'
            Column with ground truth
        save_path : str, optional
            Path to save visualization

        Returns:
        --------
        dict
            Comparison src
        """
        # Check required columns
        if siem_col not in data.columns:
            logger.error(f"SIEM column '{siem_col}' not found in data")
            return None

        if target_col not in data.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            return None

        # Get ground truth and SIEM predictions
        y_true = data[target_col].values
        y_siem = data[siem_col].values

        # Get ML predictions
        predictions = self.predict(data, return_details=True)
        y_ml = predictions['predictions']
        y_ml_proba = predictions['probabilities']

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # ML metrics
        ml_metrics = {
            'accuracy': float(accuracy_score(y_true, y_ml)),
            'precision': float(precision_score(y_true, y_ml)),
            'recall': float(recall_score(y_true, y_ml)),
            'f1': float(f1_score(y_true, y_ml)),
            'roc_auc': float(roc_auc_score(y_true, y_ml_proba))
        }

        # SIEM metrics
        siem_metrics = {
            'accuracy': float(accuracy_score(y_true, y_siem)),
            'precision': float(precision_score(y_true, y_siem)),
            'recall': float(recall_score(y_true, y_siem)),
            'f1': float(f1_score(y_true, y_siem))
        }

        # Calculate confusion matrices
        cm_ml = confusion_matrix(y_true, y_ml)
        cm_siem = confusion_matrix(y_true, y_siem)

        # Calculate timings if available
        time_diff = None
        if 'detection_time' in data.columns and 'siem_detection_time' in data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(data['detection_time']):
                data['detection_time'] = pd.to_datetime(data['detection_time'])

            if not pd.api.types.is_datetime64_dtype(data['siem_detection_time']):
                data['siem_detection_time'] = pd.to_datetime(data['siem_detection_time'])

            # Calculate time differences for true positives
            true_pos = (y_true == 1) & (y_ml == 1) & (y_siem == 1)

            if np.any(true_pos):
                time_diffs = (data.loc[true_pos, 'detection_time'] -
                              data.loc[true_pos, 'siem_detection_time']).dt.total_seconds()

                time_diff = {
                    'mean': float(time_diffs.mean()),
                    'median': float(time_diffs.median()),
                    'min': float(time_diffs.min()),
                    'max': float(time_diffs.max()),
                    'ml_faster_count': int(np.sum(time_diffs < 0)),
                    'siem_faster_count': int(np.sum(time_diffs > 0))
                }

        # Visualize comparison if save_path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            # Plot metrics comparison
            plt.figure(figsize=(10, 6))
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            x = np.arange(len(metrics))
            width = 0.35

            plt.bar(x - width / 2, [siem_metrics[m] for m in metrics], width, label='SIEM Rules')
            plt.bar(x + width / 2, [ml_metrics[m] for m in metrics], width, label='ML Approach')

            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('SIEM Rules vs ML Approach')
            plt.xticks(x, metrics)
            plt.legend()

            plt.savefig(os.path.join(save_path, 'siem_ml_comparison.png'))
            plt.close()

            # Plot confusion matrices
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            sns.heatmap(cm_siem, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('SIEM Rules')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')

            sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('ML Approach')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'confusion_matrices.png'))
            plt.close()

        # Create comparison result
        comparison = {
            'ml': {
                'metrics': ml_metrics,
                'confusion_matrix': cm_ml.tolist()
            },
            'siem': {
                'metrics': siem_metrics,
                'confusion_matrix': cm_siem.tolist()
            },
            'time_difference': time_diff
        }

        return comparison


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='APT Detection Model')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--train', type=str, default=None, help='Path to training data')
    parser.add_argument('--predict', type=str, default=None, help='Path to data for prediction')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    args = parser.parse_args()

    # Create model
    model = CombinedAPTModel(args.config)

    # Train or predict
    if args.train:
        # Load training data
        data = pd.read_csv(args.train)

        # Train model
        results = model.train(data)

        # Save model
        model.save(args.output)

        # Save src
        with open(os.path.join(args.output, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    elif args.predict:
        # Load model
        model.load()

        # Load data
        data = pd.read_csv(args.predict)

        # Make predictions
        predictions = model.predict(data, return_details=True)

        # Save predictions
        output_df = data.copy()
        output_df['apt_probability'] = predictions['probabilities']
        output_df['apt_prediction'] = predictions['predictions']
        output_df['reconstruction_error'] = predictions['reconstruction_errors']

        output_df.to_csv(os.path.join(args.output, 'predictions.csv'), index=False)

        print(f"Predictions saved to {os.path.join(args.output, 'predictions.csv')}")

    else:
        print("Please specify --train or --predict arguments")