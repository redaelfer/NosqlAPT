import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch

# 1. Connexion à Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
index_name = "logs-simples"

# 2. Lecture du fichier de logs
def charger_logs(fichier):
    # On suppose que le fichier est au format CSV simple avec séparateur virgule
    df = pd.read_csv(fichier, header=None, names=['timestamp', 'source', 'destination', 'event'])
    # Conversion de la colonne timestamp en datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# 3. Classification simple des logs
def classifier_log(event):
    # Pour cet exemple : si "APT_attack" est présent, c'est une attaque, sinon, c'est du trafic normal
    if "APT_attack" in event:
        return "attaque"
    else:
        return "normal"

def ajouter_classification(df):
    df['classification'] = df['event'].apply(classifier_log)
    return df

# 4. Indexation des logs dans Elasticsearch
def indexer_logs(df):
    # Pour chaque ligne, on crée un document JSON
    for index, row in df.iterrows():
        doc = {
            "timestamp": row['timestamp'].isoformat(),
            "source": row['source'],
            "destination": row['destination'],
            "event": row['event'],
            "classification": row['classification']
        }
        # Indexation dans Elasticsearch
        response = es.index(index=index_name, document=doc)
        print("Indexation :", response['result'])

def main():
    fichier = "logs.txt"
    logs_df = charger_logs(fichier)
    logs_df = ajouter_classification(logs_df)
    
    # Création de l'index dans Elasticsearch si non existant (avec un mapping simple)
    if not es.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "source": {"type": "ip"},
                    "destination": {"type": "ip"},
                    "event": {"type": "keyword"},
                    "classification": {"type": "keyword"}
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
        print("Index créé:", index_name)
    
    indexer_logs(logs_df)
    print("Traitement terminé.")

if __name__ == '__main__':
    main()
