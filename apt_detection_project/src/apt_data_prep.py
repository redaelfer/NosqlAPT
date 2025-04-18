#!/usr/bin/env python3
"""
APT Detection - Data Preparation Utilities
These utilities help prepare and clean data for APT detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('apt_data_prep')


class APTDataPreprocessor:
    """
    Class for preprocessing network traffic data for APT detection
    """

    def __init__(self, config=None):
        """
        Initialize the preprocessor with configuration

        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}
        self.scaler = None
        self.imputer = None
        self.feature_names = None

    def handle_missing_values(self, df, numeric_strategy='knn', categorical_strategy='constant'):
        """
        Handle missing values in the dataframe

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        numeric_strategy : str, default='knn'
            Strategy for numeric columns ('mean', 'median', 'knn')
        categorical_strategy : str, default='constant'
            Strategy for categorical columns ('mode', 'constant')

        Returns:
        --------
        pandas.DataFrame
            Dataframe with missing values handled
        """
        df = df.copy()

        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns

        if numeric_strategy == 'knn':
            if self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5)
                numeric_data = self.imputer.fit_transform(df[numeric_cols])
            else:
                numeric_data = self.imputer.transform(df[numeric_cols])

            df[numeric_cols] = numeric_data

        elif numeric_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())

        elif numeric_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())

        # Handle categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if categorical_strategy == 'mode':
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

        elif categorical_strategy == 'constant':
            for col in cat_cols:
                df[col] = df[col].fillna('unknown')

        return df

    def handle_timestamps(self, df, time_cols=None):
        """
        Extract features from timestamp columns

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        time_cols : list, optional
            List of timestamp columns, if None, auto-detect

        Returns:
        --------
        pandas.DataFrame
            Dataframe with timestamp features
        """
        df = df.copy()

        # Auto-detect timestamp columns if not provided
        if time_cols is None:
            # Look for columns with common time-related names
            time_keywords = ['time', 'date', 'timestamp', 'created', 'modified']
            time_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in time_keywords)]

        # Process each timestamp column
        for col in time_cols:
            if col in df.columns:
                try:
                    # Convert to datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')

                    # Extract time-based features
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_weekofyear'] = df[col].dt.isocalendar().week

                    # Calculate time since a reference date
                    reference_date = datetime.datetime(2020, 1, 1)
                    df[f'{col}_days_since_ref'] = (df[col] - reference_date).dt.total_seconds() / (24 * 3600)

                    # Time of day category (morning, afternoon, evening, night)
                    time_cats = pd.cut(
                        df[col].dt.hour,
                        bins=[-1, 5, 11, 17, 23],
                        labels=['night', 'morning', 'afternoon', 'evening']
                    )
                    df[f'{col}_time_category'] = time_cats

                    # Drop original timestamp column as it's not usable for ML
                    df = df.drop(columns=[col])

                    logger.info(f"Processed timestamp column: {col}")
                except Exception as e:
                    logger.warning(f"Could not process timestamp column {col}: {str(e)}")

        return df

    def handle_ip_addresses(self, df, ip_cols=None):
        """
        Extract features from IP address columns

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        ip_cols : list, optional
            List of IP address columns, if None, auto-detect

        Returns:
        --------
        pandas.DataFrame
            Dataframe with IP address features
        """
        df = df.copy()

        # Auto-detect IP columns if not provided
        if ip_cols is None:
            # Look for columns with common IP-related names
            ip_keywords = ['ip', 'addr', 'host', 'source', 'destination', 'src', 'dst']
            ip_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ip_keywords)]

        # Process each IP column
        for col in ip_cols:
            if col in df.columns:
                try:
                    # Check if column contains IP addresses
                    sample_values = df[col].dropna().sample(min(100, len(df))).astype(str)
                    is_ip_col = any('.' in str(val) and sum(c.isdigit() for c in str(val)) > 3 for val in sample_values)

                    if is_ip_col:
                        # Extract the first octet (for basic network categorization)
                        df[f'{col}_first_octet'] = df[col].astype(str).str.extract(r'(\d+)\.').astype(float)

                        # Categorize into private/public IPs
                        def ip_category(ip_str):
                            try:
                                if pd.isna(ip_str):
                                    return 'unknown'

                                # Simple check for common private IP ranges
                                ip_str = str(ip_str)
                                if ip_str.startswith('10.') or ip_str.startswith('192.168.') or ip_str.startswith(
                                        '172.'):
                                    return 'private'
                                else:
                                    return 'public'
                            except:
                                return 'unknown'

                        df[f'{col}_category'] = df[col].apply(ip_category)

                        # Convert to categorical
                        df[f'{col}_category'] = df[f'{col}_category'].astype('category')

                        logger.info(f"Processed IP address column: {col}")
                except Exception as e:
                    logger.warning(f"Could not process IP column {col}: {str(e)}")

        return df

    def handle_ports_and_protocols(self, df, port_cols=None, protocol_cols=None):
        """
        Process port and protocol columns

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        port_cols : list, optional
            List of port columns, if None, auto-detect
        protocol_cols : list, optional
            List of protocol columns, if None, auto-detect

        Returns:
        --------
        pandas.DataFrame
            Dataframe with processed port and protocol features
        """
        df = df.copy()

        # Auto-detect port columns if not provided
        if port_cols is None:
            port_keywords = ['port', 'src_port', 'dst_port', 'source_port', 'destination_port']
            port_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in port_keywords)]

        # Auto-detect protocol columns if not provided
        if protocol_cols is None:
            protocol_keywords = ['protocol', 'proto']
            protocol_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in protocol_keywords)]

        # Process port columns
        for col in port_cols:
            if col in df.columns:
                try:
                    # Convert to numeric if possible
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Create port category (well-known, registered, dynamic)
                    port_cats = pd.cut(
                        df[col],
                        bins=[-1, 1023, 49151, 65535],
                        labels=['well-known', 'registered', 'dynamic']
                    )
                    df[f'{col}_category'] = port_cats

                    # Flag common service ports
                    common_ports = {
                        '22': 'SSH',
                        '23': 'Telnet',
                        '53': 'DNS',
                        '80': 'HTTP',
                        '443': 'HTTPS',
                        '445': 'SMB',
                        '3389': 'RDP'
                    }

                    df[f'{col}_is_service'] = df[col].astype(str).isin(common_ports.keys())

                    logger.info(f"Processed port column: {col}")
                except Exception as e:
                    logger.warning(f"Could not process port column {col}: {str(e)}")

        # Process protocol columns
        for col in protocol_cols:
            if col in df.columns:
                try:
                    # Standardize protocol names to uppercase
                    df[col] = df[col].astype(str).str.upper()

                    # Create binary flags for common protocols
                    common_protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS']
                    for proto in common_protocols:
                        df[f'{col}_is_{proto}'] = df[col].str.contains(proto, case=False)

                    logger.info(f"Processed protocol column: {col}")
                except Exception as e:
                    logger.warning(f"Could not process protocol column {col}: {str(e)}")

        return df

    def normalize_features(self, df, method='robust', fit=False):
        """
        Normalize numeric features

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        method : str, default='robust'
            Normalization method ('standard', 'robust')
        fit : bool, default=False
            Whether to fit the scaler or just transform

        Returns:
        --------
        pandas.DataFrame
            Dataframe with normalized features
        """
        df = df.copy()

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) == 0:
            logger.warning("No numeric columns to normalize")
            return df

        # Create or use scaler
        if fit or self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:  # robust
                self.scaler = RobustScaler()

            # Fit and transform
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            # Just transform
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def encode_categorical(self, df, method='onehot', max_categories=10):
        """
        Encode categorical variables

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        method : str, default='onehot'
            Encoding method ('onehot', 'label', 'binary')
        max_categories : int, default=10
            Maximum number of categories to use for one-hot encoding

        Returns:
        --------
        pandas.DataFrame
            Dataframe with encoded categories
        """
        df = df.copy()

        # Get categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(cat_cols) == 0:
            logger.warning("No categorical columns to encode")
            return df

        # Process each categorical column
        for col in cat_cols:
            # Get value counts
            value_counts = df[col].value_counts()

            # For high-cardinality columns, keep only top categories
            if len(value_counts) > max_categories:
                top_categories = value_counts.index[:max_categories].tolist()
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'other')

            # Apply encoding method
            if method == 'onehot':
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])

            elif method == 'label':
                # Label encode
                categories = df[col].astype('category').cat.categories
                df[col] = df[col].astype('category').cat.codes

            elif method == 'binary':
                # Only for binary categories
                if len(df[col].unique()) <= 2:
                    df[col] = (df[col] == df[col].value_counts().index[0]).astype(int)

        return df

    def generate_flow_features(self, df, time_window=60, src_col='source_ip', dst_col='destination_ip',
                               time_col='timestamp'):
        """
        Generate network flow-based features

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        time_window : int, default=60
            Time window in seconds for flow aggregation
        src_col : str, default='source_ip'
            Source IP column name
        dst_col : str, default='destination_ip'
            Destination IP column name
        time_col : str, default='timestamp'
            Timestamp column name

        Returns:
        --------
        pandas.DataFrame
            Dataframe with flow features
        """
        if not all(col in df.columns for col in [src_col, dst_col, time_col]):
            logger.warning(f"Missing required columns for flow feature generation")
            return df

        df = df.copy()

        try:
            # Ensure timestamp is datetime
            df[time_col] = pd.to_datetime(df[time_col])

            # Sort by timestamp
            df = df.sort_values(by=time_col)

            # Create flow identifier (src_ip + dst_ip)
            df['flow_id'] = df[src_col].astype(str) + '_' + df[dst_col].astype(str)

            # Group by flow ID and calculate features within time windows
            flows = []
            for flow_id, group in df.groupby('flow_id'):
                # Skip flows with only one packet
                if len(group) <= 1:
                    flows.append(group)
                    continue

                # Calculate time deltas within the flow
                group = group.copy()
                group['time_delta'] = group[time_col].diff().dt.total_seconds()

                # Calculate flow features
                # - Packets count in the flow
                group['flow_packets_count'] = len(group)

                # - Flow duration
                flow_duration = (group[time_col].max() - group[time_col].min()).total_seconds()
                group['flow_duration'] = flow_duration

                # - Packets per second
                if flow_duration > 0:
                    group['flow_packets_per_sec'] = len(group) / flow_duration
                else:
                    group['flow_packets_per_sec'] = 0

                # - Mean time between packets
                group['flow_mean_time_delta'] = group['time_delta'].mean()

                # - Standard deviation of time between packets
                group['flow_std_time_delta'] = group['time_delta'].std()

                # - Flow direction ratio (src to dst vs dst to src)
                if 'protocol' in df.columns:
                    src_to_dst = group[group['protocol'] == 'TCP'].shape[0]
                    dst_to_src = group[group['protocol'] == 'UDP'].shape[0]
                    total = src_to_dst + dst_to_src
                    if total > 0:
                        group['flow_direction_ratio'] = src_to_dst / total
                    else:
                        group['flow_direction_ratio'] = 0.5

                flows.append(group)

            # Combine flows back into a dataframe
            if flows:
                df = pd.concat(flows)

            logger.info(f"Generated flow features for {len(df)} records")

        except Exception as e:
            logger.error(f"Error generating flow features: {str(e)}")

        return df

        def engineer_apt_specific_features(self, df):
            """
            Engineer features specifically for APT detection

            Parameters:
            -----------
            df : pandas.DataFrame
                Input dataframe

            Returns:
            --------
            pandas.DataFrame
                Dataframe with APT-specific features
            """
            df = df.copy()

            try:
                # 1. Connection duration features (APTs often have long-lived connections)
                if 'Flow Duration' in df.columns or 'flow_duration' in df.columns:
                    flow_duration_col = 'Flow Duration' if 'Flow Duration' in df.columns else 'flow_duration'
                    # Flag long connections (> 10 minutes)
                    df['is_long_connection'] = df[flow_duration_col] > 600

                    # Flag very short connections (< 1 second, potential scan)
                    df['is_scan_connection'] = df[flow_duration_col] < 1

                # 2. Beaconing detection (regular communication patterns)
                if ('flow_std_time_delta' in df.columns and 'flow_mean_time_delta' in df.columns) or \
                        ('Packet Length Std' in df.columns and 'Packet Length Mean' in df.columns):
                    # Use flow features if available, otherwise use packet length features
                    std_col = 'flow_std_time_delta' if 'flow_std_time_delta' in df.columns else 'Packet Length Std'
                    mean_col = 'flow_mean_time_delta' if 'flow_mean_time_delta' in df.columns else 'Packet Length Mean'

                    # Calculate coefficient of variation (low value suggests regular beaconing)
                    df['flow_time_regularity'] = df[std_col] / df[mean_col].replace(0, np.nan)
                    df['flow_time_regularity'] = df['flow_time_regularity'].fillna(0)

                    # Flag potential beaconing behavior
                    df['potential_beaconing'] = df['flow_time_regularity'] < 0.1

                # 3. Data exfiltration patterns
                if 'Flow Bytes/s' in df.columns:
                    # Flag large outbound data flows
                    bytes_threshold = df['Flow Bytes/s'].quantile(0.95)
                    df['potential_data_exfil'] = df['Flow Bytes/s'] > bytes_threshold

                # 4. C2 communication features
                if 'dst_port' in df.columns or any('port' in col.lower() for col in df.columns):
                    # Find port column
                    port_col = 'dst_port' if 'dst_port' in df.columns else \
                        next((col for col in df.columns if 'port' in col.lower()), None)

                    if port_col:
                        # Flag uncommon ports
                        common_ports = [22, 53, 80, 443, 8080, 8443]
                        df['uncommon_port'] = ~df[port_col].isin(common_ports)

                # 5. Flag counts analysis
                flag_columns = [col for col in df.columns if 'Flag Count' in col]
                if flag_columns:
                    # Calculate total flags
                    df['total_flags'] = df[flag_columns].sum(axis=1)

                    # Detect unusual flag patterns (high RST, FIN with low ACK, etc.)
                    if 'RST Flag Count' in df.columns and 'ACK Flag Count' in df.columns:
                        df['unusual_flags'] = (df['RST Flag Count'] > 0) & (df['ACK Flag Count'] == 0)

                # 6. Packet size variation
                if 'Packet Length Std' in df.columns and 'Packet Length Mean' in df.columns:
                    # Calculate coefficient of variation for packet size
                    df['packet_size_variation'] = df['Packet Length Std'] / df['Packet Length Mean'].replace(0, np.nan)
                    df['packet_size_variation'] = df['packet_size_variation'].fillna(0)

                    # Flag high variation (potentially encrypted payload)
                    df['high_size_variation'] = df['packet_size_variation'] > 0.5

                # 7. Look for exfiltration stage
                if 'Stage' in df.columns:
                    df['is_exfiltration'] = df['Stage'].str.contains('exfiltration', case=False, na=False).astype(int)

                logger.info(f"Engineered APT-specific features")

            except Exception as e:
                logger.error(f"Error engineering APT-specific features: {str(e)}")

            return df

        def preprocess_apt_data(self, df, target_col=None):
            """
            Complete preprocessing pipeline for APT detection

            Parameters:
            -----------
            df : pandas.DataFrame
                Input dataframe
            target_col : str, optional
                Name of target column to preserve

            Returns:
            --------
            pandas.DataFrame
                Fully preprocessed dataframe ready for modeling
            """
            # Make a copy to avoid modifying the original
            df = df.copy()

            # Store target column if provided
            target = None
            if target_col is not None and target_col in df.columns:
                target = df[target_col].copy()
                df = df.drop(columns=[target_col])

            # Apply preprocessing steps
            logger.info("Starting data preprocessing pipeline...")

            # 1. Handle missing values
            df = self.handle_missing_values(df)

            # 2. Process timestamps
            timestamp_cols = [col for col in df.columns if
                              any(keyword in str(col).lower() for keyword in ['time', 'date', 'timestamp'])]
            if timestamp_cols:
                df = self.handle_timestamps(df, timestamp_cols)

            # 3. Process IP addresses
            ip_cols = [col for col in df.columns if
                       any(keyword in str(col).lower() for keyword in ['ip', 'src', 'dst', 'source', 'destination'])]
            if ip_cols:
                df = self.handle_ip_addresses(df, ip_cols)

            # 4. Process ports and protocols
            port_cols = [col for col in df.columns if 'port' in str(col).lower()]
            protocol_cols = [col for col in df.columns if
                             any(keyword in str(col).lower() for keyword in ['protocol', 'proto'])]
            if port_cols or protocol_cols:
                df = self.handle_ports_and_protocols(df, port_cols, protocol_cols)

            # 5. Engineer APT-specific features
            df = self.engineer_apt_specific_features(df)

            # 6. Encode categorical variables
            df = self.encode_categorical(df)

            # 7. Normalize numeric features
            df = self.normalize_features(df, fit=True)

            # Add back target column if it was provided
            if target is not None:
                df[target_col] = target

            # Store feature names for future reference
            self.feature_names = df.columns.tolist()
            if target_col in self.feature_names:
                self.feature_names.remove(target_col)

            logger.info(f"Preprocessing complete. Final dataframe has {df.shape[0]} rows and {df.shape[1]} columns")

            return df

        def generate_synthetic_apt_data(self, n_samples=10000, contamination=0.05, n_features=20):
            """
            Generate synthetic data for APT detection testing

            Parameters:
            -----------
            n_samples : int, default=10000
                Number of samples to generate
            contamination : float, default=0.05
                Proportion of APT samples
            n_features : int, default=20
                Number of features to generate

            Returns:
            --------
            pandas.DataFrame
                Synthetic data for APT detection
            """
            try:
                from sklearn.datasets import make_classification
                import random

                # Generate the base synthetic data
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=int(n_features * 0.7),
                    n_redundant=int(n_features * 0.1),
                    n_repeated=0,
                    n_classes=2,
                    weights=[1 - contamination, contamination],
                    random_state=42
                )

                # Convert to DataFrame
                feature_names = [f'feature_{i}' for i in range(n_features)]
                df = pd.DataFrame(X, columns=feature_names)
                df['is_apt'] = y

                # Generate network-related features to match MongoDB schema

                # 1. Flow Duration
                flow_durations = []

                for is_apt in y:
                    if is_apt == 1:
                        # APT: longer sessions
                        flow_durations.append(random.uniform(300, 7200))  # 5 min to 2 hours
                    else:
                        # Normal: shorter sessions
                        flow_durations.append(random.uniform(10, 300))  # 10 sec to 5 min

                df['Flow Duration'] = flow_durations

                # 2. Packet counts
                df['Total Fwd Packet'] = np.where(y == 1,
                                                  np.random.randint(100, 1000, size=n_samples),
                                                  np.random.randint(10, 100, size=n_samples))

                df['Total Bwd packets'] = np.where(y == 1,
                                                   np.random.randint(50, 500, size=n_samples),
                                                   np.random.randint(5, 50, size=n_samples))

                # 3. Flow rates
                df['Flow Bytes/s'] = np.where(y == 1,
                                              np.random.uniform(1000, 10000, size=n_samples),
                                              np.random.uniform(100, 5000, size=n_samples))

                df['Flow Packets/s'] = np.where(y == 1,
                                                np.random.uniform(10, 100, size=n_samples),
                                                np.random.uniform(1, 50, size=n_samples))

                # 4. Packet length stats
                df['Packet Length Mean'] = np.where(y == 1,
                                                    np.random.uniform(800, 1500, size=n_samples),
                                                    np.random.uniform(200, 1000, size=n_samples))

                df['Packet Length Std'] = np.where(y == 1,
                                                   np.random.uniform(100, 500, size=n_samples),
                                                   np.random.uniform(10, 200, size=n_samples))

                df['Packet Length Variance'] = df['Packet Length Std'] ** 2

                # 5. Flag counts
                for flag in ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG']:
                    # APTs often have unusual flag patterns
                    if flag in ['RST', 'FIN']:
                        df[f'{flag} Flag Count'] = np.where(y == 1,
                                                            np.random.choice([0, 1, 2], size=n_samples,
                                                                             p=[0.5, 0.3, 0.2]),
                                                            np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]))
                    elif flag in ['ACK']:
                        df[f'{flag} Flag Count'] = np.where(y == 1,
                                                            np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
                                                            np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9]))
                    else:
                        df[f'{flag} Flag Count'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

                # 6. Packet size
                df['Average Packet Size'] = df['Packet Length Mean']

                df['Fwd Segment Size Avg'] = np.where(y == 1,
                                                      np.random.uniform(800, 1500, size=n_samples),
                                                      np.random.uniform(200, 1000, size=n_samples))

                df['Bwd Segment Size Avg'] = np.where(y == 1,
                                                      np.random.uniform(300, 1000, size=n_samples),
                                                      np.random.uniform(100, 500, size=n_samples))

                # 7. Add Stage for some APT samples (exfiltration)
                df['Stage'] = "normal"
                exfil_mask = (y == 1) & (np.random.random(size=n_samples) < 0.3)
                df.loc[exfil_mask, 'Stage'] = "exfiltration"

                logger.info(f"Generated synthetic APT data with {n_samples} samples")
                logger.info(f"APT class count: {sum(y == 1)} ({sum(y == 1) / n_samples:.2%})")

                return df

            except Exception as e:
                logger.error(f"Error generating synthetic data: {str(e)}")
                # Return a minimal dataframe
                df = pd.DataFrame({
                    'feature_1': np.random.randn(n_samples),
                    'feature_2': np.random.randn(n_samples),
                    'is_apt': np.random.choice([0, 1], size=n_samples, p=[1 - contamination, contamination])
                })
                return df
