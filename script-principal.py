import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import tensorflow_hub as hub
import tensorflow as tf

# Charger les données
file_path = '/Users/simonazoulay/Similar-Content/List-articles-title-meta.csv'
print("Chargement des données depuis :", file_path)
data = pd.read_csv(file_path, delimiter=';')
print("Données chargées avec succès.")

# Fusionner le Titre et la Meta Description
data['combined'] = data['Title 1'] + " " + data['Meta Description 1']
print("Colonnes fusionnées en une seule colonne 'combined'.")

# Méthode 1 : SBERT 
def find_similar_sbert(data):
    print("SBERT : Démarrage du calcul des similarités.")
    start_time = time.time()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    embeddings_cpu = embeddings.cpu()
    cosine_similarities = util.pytorch_cos_sim(embeddings_cpu, embeddings_cpu).numpy()
    end_time = time.time()
    print(f"SBERT : Calcul des similarités terminé en {end_time - start_time:.2f} secondes.")
    return cosine_similarities

# Méthode 2 : DistilBERT
def find_similar_distilbert(data):
    print("DistilBERT : Démarrage du calcul des similarités.")
    start_time = time.time()
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    embeddings_cpu = embeddings.cpu()
    cosine_similarities = util.pytorch_cos_sim(embeddings_cpu, embeddings_cpu).numpy()
    end_time = time.time()
    print(f"DistilBERT : Calcul des similarités terminé en {end_time - start_time:.2f} secondes.")
    return cosine_similarities

# Méthode 3 : RoBERTa
def find_similar_roberta(data):
    print("RoBERTa : Démarrage du calcul des similarités.")
    start_time = time.time()
    model = SentenceTransformer('stsb-roberta-large')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    embeddings_cpu = embeddings.cpu()
    cosine_similarities = util.pytorch_cos_sim(embeddings_cpu, embeddings_cpu).numpy()
    end_time = time.time()
    print(f"RoBERTa : Calcul des similarités terminé en {end_time - start_time:.2f} secondes.")
    return cosine_similarities

# Méthode 4 : BERT
def find_similar_bert(data):
    print("BERT : Démarrage du calcul des similarités.")
    start_time = time.time()
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    embeddings_cpu = embeddings.cpu()
    cosine_similarities = util.pytorch_cos_sim(embeddings_cpu, embeddings_cpu).numpy()
    end_time = time.time()
    print(f"BERT : Calcul des similarités terminé en {end_time - start_time:.2f} secondes.")
    return cosine_similarities

# Méthode 5 : USE (Universal Sentence Encoder)
def find_similar_use(data):
    print("USE : Démarrage du calcul des similarités.")
    start_time = time.time()
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = model(data['combined'].tolist())
    cosine_similarities = cosine_similarity(embeddings)
    end_time = time.time()
    print(f"USE : Calcul des similarités terminé en {end_time - start_time:.2f} secondes.")
    return cosine_similarities

# Générer les résultats
def generate_results(data, cosine_similarities, method_name):
    print(f"Génération des résultats pour la méthode {method_name}.")
    results_list = []

    def get_similar_articles(idx, sim_matrix, top_n=5):
        similar_indices = sim_matrix[idx].argsort()[-top_n-1:-1][::-1]
        return similar_indices

    for idx in range(len(data)):
        similar_indices = get_similar_articles(idx, cosine_similarities)
        similar_urls = data.iloc[similar_indices]['Original Url'].tolist()
        results_list.append({
            'Original Url': data.iloc[idx]['Original Url'],
            'Similar Urls': similar_urls
        })

    results = pd.DataFrame(results_list)
    output_file_path = f'/Users/simonazoulay/Similar-Content/similar_articles_{method_name}.csv'
    results.to_csv(output_file_path, index=False)
    print(f"Les articles similaires ont été trouvés et sauvegardés avec succès pour la méthode {method_name}.")
    return results

if __name__ == '__main__':
    # Exécuter les méthodes et sauvegarder les résultats
    print("Exécution de la méthode SBERT.")
    cosine_similarities_sbert = find_similar_sbert(data)
    results_sbert = generate_results(data, cosine_similarities_sbert, 'sbert')

    print("Exécution de la méthode DistilBERT.")
    cosine_similarities_distilbert = find_similar_distilbert(data)
    results_distilbert = generate_results(data, cosine_similarities_distilbert, 'distilbert')

    print("Exécution de la méthode RoBERTa.")
    cosine_similarities_roberta = find_similar_roberta(data)
    results_roberta = generate_results(data, cosine_similarities_roberta, 'roberta')

    print("Exécution de la méthode BERT.")
    cosine_similarities_bert = find_similar_bert(data)
    results_bert = generate_results(data, cosine_similarities_bert, 'bert')

    print("Exécution de la méthode USE.")
    cosine_similarities_use = find_similar_use(data)
    results_use = generate_results(data, cosine_similarities_use, 'use')

    print("Toutes les méthodes ont été exécutées avec succès.")
