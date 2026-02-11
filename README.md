# NosqlAPT - APT Detection via NoSQL Log Analysis 🛡️

**NosqlAPT** is an Advanced Persistent Threat (APT) detection system designed to monitor NoSQL environments. The project uses a hybrid architecture combining machine learning (XGBoost) and deep learning (Autoencoders) to identify malicious behavior through system logs.

## 🌟 Key Features

* **Multi-level Log Analysis**: Processing and normalization of system logs for relevant feature extraction.
* **Hybrid AI Detection**:
* **Autoencoder**: Used for unsupervised anomaly detection.
* **XGBoost**: Used for supervised classification of APT attack stages.


* **Preprocessing Pipeline**: Automatic data cleaning and encoding via a serialized Scikit-learn pipeline.
* **Containerized Architecture**: Simplified deployment of the ELK stack (Logstash) and detection services via Docker Compose.

## 🛠️ Technologies Used

* **Language**: Python 3.9.
* **Artificial Intelligence**: TensorFlow/Keras (Autoencoders), XGBoost, Scikit-learn.
* **Infrastructure**: Docker, Docker Compose, Logstash.
* **Data Science**: Pandas, NumPy, Joblib.

## Installation and Launch

### 1. Prerequisites

* Docker and Docker Compose installed.
* Python 3.9 (for local script execution).

### 2. Deployment via Docker

The project uses Docker Compose to orchestrate collection and processing services:

```bash
# Launch the stack (Logstash and associated services)
docker-compose up --build

```

### 3. Using Detection Scripts

You can test the detector using the scripts provided in the `scripts/` folder or at the root:

```bash
# Run the detection demo
python scripts/apt_detection_demo.py

# Integrate and process logs
python process_logs.py

```

## 📂 Project Structure

* `apt_detection_project/models/`: Contains trained models (`.h5`, `.json`) and the preprocessor (`.pkl`).
* `apt_detection_project/src/`: Core logic for detection and data preparation.
* `logstash/config/`: Configuration for log ingestion via Logstash.
* `apt_detector.py`: Main script for the detection interface.

## Detection Pipeline

1. **Ingestion**: Logs are collected and sent to the processing pipeline.
2. **Preprocessing**: Data is cleaned and transformed according to the expected features (`expected_features.json`).
3. **Analysis**: The hybrid model evaluates whether the behavior corresponds to an APT attack stage.
4. **Alerting**: Results are recorded in the detection log files.

---
