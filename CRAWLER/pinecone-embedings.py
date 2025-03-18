# pinecone_indexation.py
import os
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np
from sklearn.decomposition import PCA
import joblib  # Sauvegarde du modèle PCA

load_dotenv()

# Configuration PCA
USE_PCA = True
TARGET_DIMENSION = 384  # Dimension cible après PCA si possible
PCA_MODEL_PATH = "pca_model.pkl"  # Fichier de sauvegarde du modèle PCA

# Nom de l'index Pinecone
index_name = "education-index"

# Initialisation de la connexion Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Paramètres de connexion à PostgreSQL
DB_PARAMS = {
    "database": os.environ.get('database'),
    "user": os.environ.get('user'),
    "password": os.environ.get('password'),
    "host": os.environ.get('host'),
}

# -------------------------------------------------------------------
# Initialisation du modèle d'embeddings
# -------------------------------------------------------------------
try:
    model = SentenceTransformer('all-mpnet-base-v2', device="cpu")
    model = model.half()  # Réduction mémoire
except Exception as e:
    print("Erreur lors de la conversion du modèle en float16 sur CPU:", e)
    model = SentenceTransformer('all-mpnet-base-v2', device="cpu")

def compute_embedding(text):
    """Calcule l'embedding pour un texte donné"""
    return model.encode(text)

def normalize_embeddings(embeddings):
    """Normalise les embeddings pour garantir une bonne comparaison"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Pour éviter la division par zéro
    return embeddings / norms

# -------------------------------------------------------------------
# Vérification et création de l'index Pinecone
# -------------------------------------------------------------------
def create_or_recreate_index(target_dim):
    if index_name in pc.list_indexes():
        desc = pc.describe_index(index_name)
        existing_dim = desc['dimension']
        if existing_dim != target_dim:
            print(f"⚠️ Dimension mismatch: index {existing_dim} ≠ {target_dim}. Recréation...")
            pc.delete_index(index_name)
            pc.create_index(
                name=index_name,
                dimension=target_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            print(f"✅ Index existant avec la bonne dimension ({existing_dim}).")
    else:
        print(f"🚀 Création de l'index avec la dimension {target_dim}...")
        pc.create_index(
            name=index_name,
            dimension=target_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

# -------------------------------------------------------------------
# Indexation des documents
# -------------------------------------------------------------------
def index_documents():
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("SELECT video_id, content, title, url, language FROM Documents;")
    rows = cursor.fetchall()
    conn.close()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    
    embeddings_list, metadata_list, vector_ids = [], [], []

    for video_id, content, title, url, language in tqdm(rows, desc="Traitement des documents"):
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            embedding = compute_embedding(chunk)
            embeddings_list.append(embedding)
            metadata_list.append({
                "title": title, "url": url, "language": language, "chunk_index": i, "content": chunk
            })
            vector_ids.append(f"{video_id}_chunk_{i}")

    embeddings_array = np.array(embeddings_list)
    n_samples, n_features = embeddings_array.shape
    print(f"📊 Nombre d'embeddings : {n_samples}, Dimension originale : {n_features}")

    if USE_PCA and n_samples >= TARGET_DIMENSION:
        print("✅ PCA appliqué...")
        pca = PCA(n_components=TARGET_DIMENSION)
        embeddings_array = pca.fit_transform(embeddings_array)
        joblib.dump(pca, PCA_MODEL_PATH)  # Sauvegarde du modèle PCA
        final_dim = TARGET_DIMENSION
    else:
        print(f"⚠️ PCA non appliqué (échantillons insuffisants). Dimension utilisée : {n_features}")
        final_dim = n_features

    embeddings_array = normalize_embeddings(embeddings_array)

    index = create_or_recreate_index(final_dim)

    # Upsert dans Pinecone
    vectors = [(vector_ids[i], embeddings_array[i].tolist(), metadata_list[i]) for i in range(len(vector_ids))]
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])
    print("✅ Indexation terminée.")

if __name__ == '__main__':
    index_documents()