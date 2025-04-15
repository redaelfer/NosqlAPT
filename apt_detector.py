import pandas as pd
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch, exceptions
from pymongo import MongoClient
import logging
import random
from sklearn.ensemble import IsolationForest

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. Connexion aux bases de données
def connect_databases():
    """Établit les connexions à Elasticsearch et MongoDB avec gestion d'erreurs"""
    try:
        # Connexion à Elasticsearch
        es = Elasticsearch(['http://localhost:9200'], request_timeout=30)
        if not es.ping():
            raise ConnectionError("Impossible de se connecter à Elasticsearch")
        
        # Connexion à MongoDB
        mongo_client = MongoClient('mongodb://mongodb:27017/', serverSelectionTimeoutMS=5000)
        mongo_db = mongo_client['apt_detection']
        
        logger.info("Connexions aux bases établies avec succès")
        return es, mongo_db
    
    except Exception as e:
        logger.error(f"Erreur de connexion : {str(e)}")
        exit(1)

# 2. Génération de logs synthétiques (pour tester)
def generate_sample_logs(filename="logs.txt", num_entries=100):
    """Génère un fichier de logs de test avec des patterns normaux et suspects"""
    normal_events = [
        "user_login", 
        "file_access", 
        "system_update"
    ]
    
    attack_events = [
        "APT_attack_port_scan",
        "APT_attack_data_exfiltration",
        "APT_attack_bruteforce"
    ]
    
    with open(filename, 'w') as f:
        for _ in range(num_entries):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            source = f"192.168.1.{random.randint(1, 50)}"
            dest = f"10.0.0.{random.randint(1, 10)}"
            
            # 10% de chance d'être une attaque
            if random.random() < 0.1:
                event = random.choice(attack_events)
            else:
                event = random.choice(normal_events)
            
            f.write(f"{timestamp},{source},{dest},{event}\n")
    
    logger.info(f"Fichier {filename} généré avec {num_entries} entrées")

# 3. Classification avec Machine Learning basique
class APTDetector:
    """Classe pour la détection d'anomalies avec Isolation Forest"""
    
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
    def extract_features(self, event):
        """Transforme les logs en features pour le modèle ML"""
        return [
            len(event),                  # Longueur de l'événement
            event.count('APT_attack'),   # Nombre de mots-clés suspects
            event.count('_')             # Complexité de l'événement
        ]
    
    def train(self, events):
        """Entraîne le modèle sur des données historiques"""
        features = [self.extract_features(e) for e in events]
        self.model.fit(features)
        logger.info("Modèle ML entraîné avec succès")
    
    def predict(self, event):
        """Prédit si un événement est anormal"""
        features = self.extract_features(event)
        return self.model.predict([features])[0] == -1  # -1 = anomalie

def afficher_attaques_mongodb():
    """Affiche les attaques stockées dans MongoDB"""
    try:
        client = MongoClient('mongodb://mongodb:27017/', serverSelectionTimeoutMS=5000)
        db = client['apt_detection']
        collection = db['confirmed_attacks']
        
        print("\nAttaques stockées dans MongoDB:")
        for attack in collection.find():
            print(f"- {attack['event']} ({attack['timestamp']})")
            
        print(f"\nTotal: {collection.count_documents({})} attaques")
        
    except Exception as e:
        print(f"Erreur MongoDB: {str(e)}")


# 4. Pipeline principal
def main():
    # Génération de logs de test
    generate_sample_logs()
    
    # Initialisation
    es, mongo_db = connect_databases()
    detector = APTDetector()
    index_name = "apt-logs"
    collection = mongo_db['confirmed_attacks']
    
    # Chargement des données
    try:
        df = pd.read_csv(
            "logs.txt", 
            header=None, 
            names=['timestamp', 'source', 'destination', 'event']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Entraînement du modèle ML
        detector.train(df['event'].tolist())
        
        # Prédictions
        df['is_anomaly'] = df['event'].apply(detector.predict)
        df['classification'] = np.where(
            df['is_anomaly'], 
            'attaque', 
            'normal'
        )
        
    except Exception as e:
        logger.error(f"Erreur de traitement : {str(e)}")
        exit(1)
    
    # Création de l'index Elasticsearch
    if not es.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "source": {"type": "ip"},
                    "destination": {"type": "ip"},
                    "event": {"type": "keyword"},
                    "classification": {"type": "keyword"},
                    "is_anomaly": {"type": "boolean"}
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Index {index_name} créé")
    
    # Indexation et stockage dans MongoDB
    attack_count = 0
    for _, row in df.iterrows():
        doc = {
            "timestamp": row['timestamp'].isoformat(),
            "source": row['source'],
            "destination": row['destination'],
            "event": row['event'],
            "classification": row['classification'],
            "is_anomaly": row['is_anomaly']
        }
        
        try:
            # Indexation dans Elasticsearch
            es.index(index=index_name, document=doc)
            
            # Stockage dans MongoDB si attaque
            if row['classification'] == 'attaque':
                collection.insert_one(doc)
                attack_count += 1
                
        except Exception as e:
            logger.error(f"Erreur d'indexation : {str(e)}")
    
    logger.info(f"""
    Traitement terminé !
    - Logs indexés : {len(df)}
    - Attaques détectées : {attack_count}
    """)
    afficher_attaques_mongodb()

    # TEST : Vérifier si la base "apt_detection" existe
    print("\nListe des bases de données:", mongo_db.client.list_database_names())
    print("Collections dans apt_detection:", mongo_db.list_collection_names())



if __name__ == '__main__':
    main()

