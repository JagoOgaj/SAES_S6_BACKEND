# pinecone_indexation.py
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "education-index"
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# -------------------------------------------------------------------
# Paramètres de connexion à PostgreSQL
# -------------------------------------------------------------------
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
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    model = model.half()
except Exception as e:
    print("Erreur lors de la conversion du modèle en float16 sur CPU:", e)
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

def compute_embedding(text):
    return model.encode(text).tolist()

# -------------------------------------------------------------------
# Lecture des documents depuis PostgreSQL et indexation dans Pinecone avec chunking
# -------------------------------------------------------------------
def index_documents():
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("SELECT video_id, content, title, url, language FROM Documents;")
    rows = cursor.fetchall()
    conn.close()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    vectors = []

    for video_id, content, title, url, language in tqdm(rows, desc="Traitement des documents"):
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            embedding = compute_embedding(chunk)
            metadata = {
                "title": title,
                "url": url,
                "language": language,
                "chunk_index": i,
                "content": chunk 
            }
            vector_id = f"{video_id}_chunk_{i}"
            vectors.append((vector_id, embedding, metadata))

    print(f"Upserting {len(vectors)} embeddings dans Pinecone...")
    index.upsert(vectors=vectors)
    print("Indexation terminée.")

if __name__ == '__main__':
    index_documents()
