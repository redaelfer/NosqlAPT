import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from datetime import datetime
import logging
import pymongo
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('apt_detector')


class APTDetector:
    def __init__(self, config_path=None):
        """
        Initialize the APT detector with configuration

        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.autoencoder = None
        self.xgb_model = None
        self.preprocessor = None
        self.feature_names = None
        self._setup_db_connections()

    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            'mongodb': {
                'host': 'localhost',
                'port': 27017,
                'db_name': 'apt_detection',
                'collection': 'events'
            },
            'elasticsearch': {
                'host': 'localhost',
                'port': 9200,
                'index': 'logs'
            },
            'model_paths': {
                'autoencoder': 'models/autoencoder.h5',
                'xgboost': 'models/xgboost_model.pkl',
                'preprocessor': 'models/preprocessor.pkl'
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'use_smote': True,
                'smote_ratio': 0.7,  # Minority class will be 70% of majority class
                'autoencoder': {
                    'encoding_dim': 16,
                    'hidden_layers': [64, 32],
                    'activation': 'relu',
                    'loss': 'mse',
                    'optimizer': 'adam',
                    'epochs': 50,
                    'batch_size': 32,
                    'patience': 5
                },
                'xgboost': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'use_label_encoder': False,
                    'tree_method': 'hist',  # for faster computation
                    'early_stopping_rounds': 10
                }
            },
            'feature_engineering': {
                'include_autoencoder_features': True,
                'reconstruction_error_threshold': 0.05,
                'use_feature_selection': True
            },
            'inference': {
                'batch_size': 1000,
                'threshold': 0.5  # Default threshold for classification
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)

                # Recursively update default config with user config
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and k in d:
                            d[k] = update_dict(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d

                default_config = update_dict(default_config, user_config)

        return default_config

    def _setup_db_connections(self):
        """Set up connections to MongoDB and Elasticsearch"""
        try:
            # MongoDB connection
            mongo_config = self.config['mongodb']
            self.mongo_client = MongoClient(
                mongo_config['host'],
                mongo_config['port']
            )
            self.mongodb = self.mongo_client[mongo_config['db_name']]
            logger.info(f"Successfully connected to MongoDB: {mongo_config['db_name']}")

            # Elasticsearch connection
            es_config = self.config['elasticsearch']
            try:
                from elasticsearch import Elasticsearch
                self.es_client = Elasticsearch(
                    [{'host': es_config['host'], 'port': es_config['port']}]
                )
                if self.es_client.ping():
                    logger.info("Successfully connected to Elasticsearch")
                else:
                    logger.warning("Could not connect to Elasticsearch")
                    self.es_client = None
            except ImportError:
                logger.warning("Elasticsearch not installed, ES functionality will be disabled")
                self.es_client = None

        except Exception as e:
            logger.error(f"Error setting up database connections: {str(e)}")
            self.mongo_client = None
            self.mongodb = None
            self.es_client = None

    def fetch_data(self, query=None, limit=None, from_es=False):
        """
        Fetch data from MongoDB or Elasticsearch

        Parameters:
        -----------
        query : dict, optional
            Query to filter data
        limit : int, optional
            Maximum number of records to fetch
        from_es : bool, default=False
            Whether to fetch from Elasticsearch instead of MongoDB

        Returns:
        --------
        pandas.DataFrame
            Fetched data
        """
        if query is None:
            query = {}

        try:
            if from_es and self.es_client:
                es_config = self.config['elasticsearch']
                es_query = {
                    "query": {
                        "match_all": {}
                    }
                }
                if limit:
                    es_query["size"] = limit

                response = self.es_client.search(
                    index=es_config['index'],
                    body=es_query
                )

                # Extract hits and convert to DataFrame
                hits = response['hits']['hits']
                data = [hit['_source'] for hit in hits]
                df = pd.DataFrame(data)

            elif self.mongo_client:
                mongo_config = self.config['mongodb']
                collection = self.mongodb[mongo_config['collection']]

                if limit:
                    cursor = collection.find(query).limit(limit)
                else:
                    cursor = collection.find(query)

                df = pd.DataFrame(list(cursor))

            else:
                logger.error("No database connection available")
                return None

            logger.info(f"Fetched {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None

    def preprocess_data(self, df, fit=False):
        """
        Preprocess the data for model training/inference

        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        fit : bool, default=False
            Whether to fit the preprocessor or just transform

        Returns:
        --------
        numpy.ndarray
            Preprocessed features
        """
        # Handle missing values
        df = df.copy()

        # Drop unnecessary columns
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])

        # Handle missing values in categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('missing')

        # Convert categorical to numerical using one-hot encoding
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Ensure all columns are numeric
        numeric_cols = df_encoded.select_dtypes(include=['number']).columns.tolist()
        df_encoded = df_encoded[numeric_cols]

        # Store feature names
        if fit:
            self.feature_names = df_encoded.columns.tolist()

        # Create or use preprocessor
        if fit or self.preprocessor is None:
            # Create preprocessor pipeline
            from sklearn.pipeline import Pipeline
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import RobustScaler

            self.preprocessor = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', RobustScaler())  # RobustScaler is good for data with outliers
            ])
            X_processed = self.preprocessor.fit_transform(df_encoded)
        else:
            X_processed = self.preprocessor.transform(df_encoded)

        return X_processed

    def build_autoencoder(self, input_dim):
        """
        Build autoencoder model architecture

        Parameters:
        -----------
        input_dim : int
            Dimension of input features

        Returns:
        --------
        tuple
            (encoder model, autoencoder model)
        """
        autoencoder_config = self.config['training']['autoencoder']
        encoding_dim = autoencoder_config['encoding_dim']
        hidden_layers = autoencoder_config['hidden_layers']
        activation = autoencoder_config['activation']

        # Input layer
        input_layer = Input(shape=(input_dim,))

        # Encoder
        encoder = input_layer
        for units in hidden_layers:
            encoder = Dense(units, activation=activation)(encoder)
            encoder = Dropout(0.2)(encoder)

        # Bottleneck layer
        bottleneck = Dense(encoding_dim, activation=activation, name='bottleneck')(encoder)

        # Decoder
        decoder = bottleneck
        for units in reversed(hidden_layers):
            decoder = Dense(units, activation=activation)(decoder)
            decoder = Dropout(0.2)(decoder)

        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoder)

        # Models
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        encoder_model = Model(inputs=input_layer, outputs=bottleneck)

        # Compile
        autoencoder.compile(
            optimizer=autoencoder_config['optimizer'],
            loss=autoencoder_config['loss']
        )

        return encoder_model, autoencoder

    def train_autoencoder(self, X_train, X_val=None):
        """
        Train the autoencoder model

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        X_val : numpy.ndarray, optional
            Validation data

        Returns:
        --------
        tuple
            (encoder model, autoencoder model, history)
        """
        autoencoder_config = self.config['training']['autoencoder']

        # Build autoencoder
        encoder, autoencoder = self.build_autoencoder(X_train.shape[1])

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=autoencoder_config['patience'],
                restore_best_weights=True
            )
        ]

        # Model path for saving
        model_dir = os.path.dirname(self.config['model_paths']['autoencoder'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        callbacks.append(
            ModelCheckpoint(
                self.config['model_paths']['autoencoder'],
                monitor='val_loss',
                save_best_only=True
            )
        )

        # Train the model
        history = autoencoder.fit(
            X_train, X_train,
            epochs=autoencoder_config['epochs'],
            batch_size=autoencoder_config['batch_size'],
            validation_data=(X_val, X_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )

        # Save the encoder separately
        encoder_path = self.config['model_paths']['autoencoder'].replace('.h5', '_encoder.h5')
        encoder.save(encoder_path)

        # Store the model
        self.autoencoder = autoencoder

        return encoder, autoencoder, history

    def load_autoencoder(self):
        """Load the trained autoencoder model"""
        try:
            model_path = self.config['model_paths']['autoencoder']
            if os.path.exists(model_path):
                self.autoencoder = load_model(model_path)
                logger.info(f"Loaded autoencoder from {model_path}")
                return True
            else:
                logger.warning(f"Autoencoder model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading autoencoder: {str(e)}")
            return False

    def compute_reconstruction_error(self, X):
        """
        Compute reconstruction error from autoencoder

        Parameters:
        -----------
        X : numpy.ndarray
            Input data

        Returns:
        --------
        numpy.ndarray
            Reconstruction error for each sample
        """
        if self.autoencoder is None:
            if not self.load_autoencoder():
                logger.error("Autoencoder not available")
                return None

        # Get reconstructions
        X_reconstructed = self.autoencoder.predict(X)

        # Calculate MSE for each sample
        mse = np.mean(np.power(X - X_reconstructed, 2), axis=1)

        return mse

    def get_encoder_features(self, X):
        """
        Get encoded features from the autoencoder

        Parameters:
        -----------
        X : numpy.ndarray
            Input data

        Returns:
        --------
        numpy.ndarray
            Encoded features
        """
        if self.autoencoder is None:
            if not self.load_autoencoder():
                logger.error("Autoencoder not available")
                return None

        # Extract encoder part from autoencoder
        encoder_layer_name = 'bottleneck'
        encoder_output = self.autoencoder.get_layer(encoder_layer_name).output
        encoder_model = Model(inputs=self.autoencoder.input, outputs=encoder_output)

        # Get encoded features
        encoded_features = encoder_model.predict(X)

        return encoded_features

    def prepare_xgboost_features(self, X, y=None, include_reconstruction=True, include_encoded=True):
        """
        Prepare features for XGBoost, including autoencoder-derived features

        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray, optional
            Target labels
        include_reconstruction : bool, default=True
            Whether to include reconstruction error as a feature
        include_encoded : bool, default=True
            Whether to include encoded features from autoencoder

        Returns:
        --------
        tuple
            (X_features, y), where X_features includes original and derived features
        """
        features_list = [X]  # Start with original features

        # Add reconstruction error if requested
        if include_reconstruction:
            reconstruction_errors = self.compute_reconstruction_error(X)
            if reconstruction_errors is not None:
                features_list.append(reconstruction_errors.reshape(-1, 1))

        # Add encoded features if requested
        if include_encoded:
            encoded_features = self.get_encoder_features(X)
            if encoded_features is not None:
                features_list.append(encoded_features)

        # Combine all features
        if len(features_list) > 1:
            X_features = np.hstack(features_list)
        else:
            X_features = features_list[0]

        return X_features, y

    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """
        Optimize XGBoost hyperparameters using Optuna

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation labels
        n_trials : int, default=50
            Number of optimization trials

        Returns:
        --------
        dict
            Best hyperparameters
        """
        try:
            import optuna

            def objective(trial):
                # Define the hyperparameters to optimize
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'hist',
                    'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                    'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                    'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                }

                # Create and train the model
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )

                # Return the best validation score
                return model.best_score

            # Create the Optuna study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_trial.value}")
            logger.info(f"Best params: {study.best_trial.params}")

            # Add the best number of estimators
            best_params = study.best_trial.params
            best_params['n_estimators'] = 1000  # We'll use early stopping

            return best_params

        except ImportError:
            logger.warning("Optuna not installed. Using default parameters.")
            return self.config['training']['xgboost']

    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, optimize=False):
        """
        Train the XGBoost model

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray, optional
            Validation features
        y_val : numpy.ndarray, optional
            Validation labels
        optimize : bool, default=False
            Whether to optimize hyperparameters

        Returns:
        --------
        xgb.Booster
            Trained XGBoost model
        """
        # Handle imbalanced data if needed
        if self.config['training']['use_smote'] and len(np.unique(y_train)) > 1:
            try:
                # Only apply SMOTE if we have both classes
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(
                    sampling_strategy=self.config['training']['smote_ratio'],
                    random_state=self.config['training']['random_state']
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(
                    f"Applied SMOTE: {np.sum(y_train == 0)} negative samples, {np.sum(y_train == 1)} positive samples")
            except ImportError:
                logger.warning("imbalanced-learn not installed. Skipping SMOTE.")

        # Get model parameters
        if optimize and X_val is not None and y_val is not None:
            logger.info("Optimizing XGBoost hyperparameters...")
            params = self.optimize_xgboost(X_train, y_train, X_val, y_val)
        else:
            params = self.config['training']['xgboost']

        # Convert data to DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'val')]
        else:
            watchlist = [(dtrain, 'train')]

        # Train the model
        num_round = params.pop('n_estimators', 100)
        early_stopping_rounds = params.pop('early_stopping_rounds', 10)

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )

        # Save the model
        model_dir = os.path.dirname(self.config['model_paths']['xgboost'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        bst.save_model(self.config['model_paths']['xgboost'])

        # Save the preprocessor
        with open(self.config['model_paths']['preprocessor'], 'wb') as f:
            pickle.dump(self.preprocessor, f)

        # Store the model
        self.xgb_model = bst

        return bst

    def load_xgboost(self):
        """Load the trained XGBoost model"""
        try:
            model_path = self.config['model_paths']['xgboost']
            if os.path.exists(model_path):
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(model_path)
                logger.info(f"Loaded XGBoost model from {model_path}")

                # Load preprocessor
                preprocessor_path = self.config['model_paths']['preprocessor']
                if os.path.exists(preprocessor_path):
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessor = pickle.load(f)
                    logger.info(f"Loaded preprocessor from {preprocessor_path}")

                return True
            else:
                logger.warning(f"XGBoost model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            return False

    def train(self, df, target_col, test_size=None, optimize_xgb=False):
        """
        Train both autoencoder and XGBoost models

        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        target_col : str
            Name of the target column
        test_size : float, optional
            Test size for train-test split, uses config value if not provided
        optimize_xgb : bool, default=False
            Whether to optimize XGBoost hyperparameters

        Returns:
        --------
        dict
            Training src and metrics
        """
        # Get parameters from config
        if test_size is None:
            test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']

        # Extract target
        y = df[target_col].values
        X_df = df.drop(columns=[target_col])

        # Split data
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Preprocess data
        logger.info("Preprocessing training data...")
        X_train = self.preprocess_data(X_train_df, fit=True)
        X_test = self.preprocess_data(X_test_df, fit=False)

        # Train autoencoder
        logger.info("Training autoencoder...")
        encoder, autoencoder, ae_history = self.train_autoencoder(X_train, X_test)

        # Prepare XGBoost features
        logger.info("Preparing XGBoost features...")
        X_train_xgb, y_train_xgb = self.prepare_xgboost_features(
            X_train, y_train,
            include_reconstruction=self.config['feature_engineering']['include_autoencoder_features'],
            include_encoded=self.config['feature_engineering']['include_autoencoder_features']
        )

        X_test_xgb, y_test_xgb = self.prepare_xgboost_features(
            X_test, y_test,
            include_reconstruction=self.config['feature_engineering']['include_autoencoder_features'],
            include_encoded=self.config['feature_engineering']['include_autoencoder_features']
        )

        # Train XGBoost
        logger.info("Training XGBoost model...")
        bst = self.train_xgboost(
            X_train_xgb, y_train_xgb,
            X_test_xgb, y_test_xgb,
            optimize=optimize_xgb
        )

        # Evaluate XGBoost
        dtest = xgb.DMatrix(X_test_xgb)
        y_pred_proba = bst.predict(dtest)
        y_pred = (y_pred_proba > self.config['inference']['threshold']).astype(int)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(
            f"XGBoost performance - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Return src
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'autoencoder_history': {k: [float(val) for val in v] for k, v in ae_history.history.items()},
            'feature_importance': self.get_feature_importance()
        }

        return results

    def get_feature_importance(self, plot=False):
        """
        Get feature importance from the XGBoost model

        Parameters:
        -----------
        plot : bool, default=False
            Whether to plot feature importance

        Returns:
        --------
        dict
            Feature importance scores
        """
        if self.xgb_model is None:
            if not self.load_xgboost():
                logger.error("XGBoost model not available")
                return None

        # Get feature importance
        importance_type = 'gain'
        scores = self.xgb_model.get_score(importance_type=importance_type)

        # Get feature names in a dict format
        feature_names = [f'f{i}' for i in range(len(scores))]

        if self.feature_names:
            # Map feature indices to actual feature names
            # Handling potentially different lengths
            orig_features_count = len(self.feature_names)
            feature_names = []

            for i in range(len(scores)):
                if i < orig_features_count:
                    feature_names.append(self.feature_names[i])
                elif i == orig_features_count:
                    feature_names.append('reconstruction_error')
                else:
                    feature_names.append(f'encoded_{i - orig_features_count - 1}')

        # Convert to a sorted list of (feature, importance) tuples
        importance_tuples = [(feature_names[int(k.replace('f', ''))], v) for k, v in scores.items()]
        importance_tuples.sort(key=lambda x: x[1], reverse=True)

        # Convert to dict
        importance_dict = {feat: float(score) for feat, score in importance_tuples}

        # Plot if requested
        if plot:
            plt.figure(figsize=(12, 8))
            sorted_idx = np.argsort([v for _, v in importance_tuples])
            plt.barh(
                [importance_tuples[i][0] for i in sorted_idx[-30:]],
                [importance_tuples[i][1] for i in sorted_idx[-30:]]
            )
            plt.xlabel('Feature Importance (Gain)')
            plt.title('Top 30 Features by Importance')
            plt.tight_layout()
            plt.show()

        return importance_dict

    def predict(self, df, threshold=None):
        """
        Make predictions using the APT detection model

        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        threshold : float, optional
            Classification threshold, uses config value if not provided

        Returns:
        --------
        tuple
            (predictions, probabilities, anomaly_scores)
        """
        if threshold is None:
            threshold = self.config['inference']['threshold']

        # Preprocess data
        X = self.preprocess_data(df)

        # Prepare features for XGBoost
        X_xgb, _ = self.prepare_xgboost_features(X)

        # Make predictions
        # Make predictions
        if self.xgb_model is None:
            if not self.load_xgboost():
                logger.error("XGBoost model not available")
                return None, None, None

        dtest = xgb.DMatrix(X_xgb)
        probabilities = self.xgb_model.predict(dtest)
        predictions = (probabilities > threshold).astype(int)

        # Compute reconstruction errors as another anomaly signal
        reconstruction_errors = self.compute_reconstruction_error(X)

        return predictions, probabilities, reconstruction_errors

    def explain_prediction(self, df, index=0):
        """
        Explain prediction using SHAP values

        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        index : int, default=0
            Index of the sample to explain

        Returns:
        --------
        dict
            SHAP explanation
        """
        # Preprocess data
        X = self.preprocess_data(df)

        # Prepare features for XGBoost
        X_xgb, _ = self.prepare_xgboost_features(X)

        # Get SHAP values
        try:
            import shap
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer.shap_values(X_xgb)
        except ImportError:
            logger.warning("SHAP not installed. Cannot explain prediction.")
            return None

        # Get feature names
        feature_names = self.feature_names.copy() if self.feature_names else [f'feature_{i}' for i in
                                                                              range(X_xgb.shape[1])]

        # Add autoencoder feature names if used
        if self.config['feature_engineering']['include_autoencoder_features']:
            orig_feature_count = len(self.feature_names) if self.feature_names else X.shape[1]
            encoding_dim = self.config['training']['autoencoder']['encoding_dim']

            # Add reconstruction error feature
            feature_names.append('reconstruction_error')

            # Add encoded features
            for i in range(encoding_dim):
                feature_names.append(f'encoded_{i}')

        # Handle case where actual features differ from expected
        if len(feature_names) != X_xgb.shape[1]:
            feature_names = [f'feature_{i}' for i in range(X_xgb.shape[1])]

        # Get explanation for a specific sample
        sample_idx = min(index, X_xgb.shape[0] - 1)  # Ensure valid index

        # Build explanation dictionary
        explanation = {
            'base_value': float(explainer.expected_value),
            'features': []
        }

        # For each feature, add its contribution
        for i, name in enumerate(feature_names):
            explanation['features'].append({
                'name': name,
                'value': float(X_xgb[sample_idx, i]),
                'contribution': float(shap_values[sample_idx, i])
            })

        # Sort by absolute contribution
        explanation['features'].sort(key=lambda x: abs(x['contribution']), reverse=True)

        return explanation

    def save_to_mongodb(self, predictions, df, collection_name='detections'):
        """
        Save predictions to MongoDB

        Parameters:
        -----------
        predictions : numpy.ndarray
            Prediction src (0 or 1)
        df : pandas.DataFrame
            Original data
        collection_name : str, default='detections'
            Name of the MongoDB collection to save to

        Returns:
        --------
        bool
            Whether the operation was successful
        """
        if self.mongo_client is None:
            logger.error("MongoDB connection not available")
            return False

        try:
            # Prepare data for saving
            df = df.copy()
            df['is_apt'] = predictions
            df['detection_time'] = pd.Timestamp.now()

            # Convert to records
            records = df.to_dict(orient='records')

            # Save to MongoDB
            collection = self.mongodb[collection_name]
            result = collection.insert_many(records)

            logger.info(f"Saved {len(result.inserted_ids)} predictions to MongoDB")
            return True

        except Exception as e:
            logger.error(f"Error saving to MongoDB: {str(e)}")
            return False

    def visualize_results(self, y_true, y_pred, y_proba):
        """
        Visualize model performance

        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels
        y_proba : numpy.ndarray
            Prediction probabilities

        Returns:
        --------
        dict
            Dictionary with plot data
        """
        # Initialize src dict
        results = {}

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'APT'],
            yticklabels=['Normal', 'APT']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        results['confusion_matrix'] = cm.tolist()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        results['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        results['pr'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'auc': float(pr_auc)
        }

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        results['metrics'] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred))
        }

        return results

    def compare_with_siem(self, df_siem, target_col, df_ml=None):
        """
        Compare ML approach with SIEM rules

        Parameters:
        -----------
        df_siem : pandas.DataFrame
            Data with SIEM rule detection src
        target_col : str
            Name of the ground truth target column
        df_ml : pandas.DataFrame, optional
            Data for ML detection, if None, uses same as df_siem

        Returns:
        --------
        dict
            Comparison src
        """
        if df_ml is None:
            df_ml = df_siem.copy()

        # Check that both datasets have the target column
        if target_col not in df_siem.columns or target_col not in df_ml.columns:
            logger.error(f"Target column {target_col} not found in data")
            return None

        # Get ground truth
        y_true = df_siem[target_col].values

        # Get SIEM predictions (assuming column 'siem_detection')
        siem_col = 'siem_detection'
        if siem_col not in df_siem.columns:
            logger.error(f"SIEM detection column {siem_col} not found in data")
            return None

        y_siem = df_siem[siem_col].values

        # Get ML predictions
        _, y_ml_proba, _ = self.predict(df_ml.drop(columns=[target_col]))
        threshold = self.config['inference']['threshold']
        y_ml = (y_ml_proba > threshold).astype(int)

        # Calculate metrics for both approaches
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        siem_metrics = {
            'accuracy': float(accuracy_score(y_true, y_siem)),
            'precision': float(precision_score(y_true, y_siem)),
            'recall': float(recall_score(y_true, y_siem)),
            'f1': float(f1_score(y_true, y_siem))
        }

        ml_metrics = {
            'accuracy': float(accuracy_score(y_true, y_ml)),
            'precision': float(precision_score(y_true, y_ml)),
            'recall': float(recall_score(y_true, y_ml)),
            'f1': float(f1_score(y_true, y_ml))
        }

        # Confusion matrices
        siem_cm = confusion_matrix(y_true, y_siem)
        ml_cm = confusion_matrix(y_true, y_ml)

        # Calculate detection time difference if available
        time_diff = None
        if 'siem_detection_time' in df_siem.columns and 'ml_detection_time' in df_ml.columns:
            # Only compare times for true positives
            true_pos_idx = (y_true == 1) & (y_siem == 1) & (y_ml == 1)

            if np.any(true_pos_idx):
                siem_times = pd.to_datetime(df_siem.loc[true_pos_idx, 'siem_detection_time'])
                ml_times = pd.to_datetime(df_ml.loc[true_pos_idx, 'ml_detection_time'])

                # Calculate time differences in seconds
                diff_seconds = (ml_times - siem_times).dt.total_seconds()

                time_diff = {
                    'mean': float(diff_seconds.mean()),
                    'median': float(diff_seconds.median()),
                    'min': float(diff_seconds.min()),
                    'max': float(diff_seconds.max())
                }

        # Combine src
        comparison = {
            'siem': {
                'metrics': siem_metrics,
                'confusion_matrix': siem_cm.tolist()
            },
            'ml': {
                'metrics': ml_metrics,
                'confusion_matrix': ml_cm.tolist()
            },
            'detection_time_diff': time_diff
        }

        return comparison

    def monitor_adaptive_performance(self, data_stream, target_col, batch_size=1000, adaptation_threshold=0.1):
        """
        Monitor model performance over time and adapt if needed

        Parameters:
        -----------
        data_stream : generator
            Generator yielding batches of data
        target_col : str
            Name of the target column
        batch_size : int, default=1000
            Number of samples to process at once
        adaptation_threshold : float, default=0.1
            Performance drop threshold to trigger adaptation

        Returns:
        --------
        dict
            Monitoring src
        """
        performance_history = []
        baseline_f1 = None
        total_samples = 0
        adaptation_needed = False

        for batch_idx, batch_df in enumerate(data_stream):
            # Check that batch has expected columns
            if target_col not in batch_df.columns:
                logger.error(f"Target column {target_col} not found in batch {batch_idx}")
                continue

            # Extract ground truth
            y_true = batch_df[target_col].values
            X_df = batch_df.drop(columns=[target_col])

            # Make predictions
            y_pred, y_proba, _ = self.predict(X_df)

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            batch_metrics = {
                'batch': batch_idx,
                'samples': len(batch_df),
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1)
            }

            # Update performance history
            performance_history.append(batch_metrics)
            total_samples += len(batch_df)

            # Set baseline F1 if not set
            if baseline_f1 is None:
                baseline_f1 = f1
                logger.info(f"Baseline F1 score: {baseline_f1:.4f}")

            # Check for performance degradation
            if f1 < baseline_f1 - adaptation_threshold:
                logger.warning(
                    f"Performance drop detected in batch {batch_idx}: F1 {f1:.4f} vs baseline {baseline_f1:.4f}")
                adaptation_needed = True

                # Store data for retraining
                if not hasattr(self, 'adaptation_data'):
                    self.adaptation_data = batch_df
                else:
                    self.adaptation_data = pd.concat([self.adaptation_data, batch_df])

                # Retrain if enough data collected
                if len(self.adaptation_data) >= batch_size:
                    logger.info(f"Retraining model with {len(self.adaptation_data)} new samples")

                    # Retrain
                    self.train(self.adaptation_data, target_col, test_size=0.3)

                    # Reset
                    self.adaptation_data = None
                    adaptation_needed = False
                    baseline_f1 = None  # Will be reset on next batch

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {total_samples} samples in {batch_idx + 1} batches")
                logger.info(f"Current F1 score: {f1:.4f}")

        # Prepare monitoring src
        monitoring_results = {
            'performance_history': performance_history,
            'total_samples': total_samples,
            'adaptation_needed': adaptation_needed
        }

        return monitoring_results