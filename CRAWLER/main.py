import os
import numpy as np
import joblib
from typing import List, Optional, Any
from pydantic import Field
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableLambda

load_dotenv()

# 🔹 Chargement des clés API
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

PCA_MODEL_PATH = "pca_model.pkl"

# -------------------------------
# 📌 Classification thématique (Zero-Shot)
# -------------------------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def predict_category(question: str) -> str:
    candidate_labels = ["histoire", "mathématiques", "sciences", "philosophie", "économie", "éducation"]
    result = classifier(question, candidate_labels)
    return f"éducation, {result['labels'][0]}"

# -------------------------------
# 📌 Expansion de requête (Amélioration de la recherche)
# -------------------------------
query_expansion_model = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def expand_query(question: str) -> str:
    reformulated_query = query_expansion_model(question, max_length=64, do_sample=False)
    return reformulated_query[0]["generated_text"]

# -------------------------------
# 📌 Normalisation des embeddings (pour la cohérence)
# -------------------------------
def normalize_embeddings(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

# -------------------------------
# 📌 Embeddings personnalisés avec PCA si disponible
# -------------------------------
class CustomEmbeddings(HuggingFaceEmbeddings):
    pca_model_path: Optional[str] = Field(default=None)
    pca: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self, model_name: str, pca_model_path: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        object.__setattr__(self, "pca_model_path", pca_model_path)
        if pca_model_path and os.path.exists(pca_model_path):
            self.pca = joblib.load(pca_model_path)
            print(f"✅ PCA model chargé depuis {pca_model_path}.")
        else:
            print("⚠️ Aucun modèle PCA trouvé. Utilisation de la dimension d'origine.")

    def embed_query(self, text: str) -> List[float]:
        # Obtenir l'embedding de base via le modèle
        base_embedding = super().embed_query(text)
        base_embedding = np.array(base_embedding)
        # Squeeze pour obtenir un vecteur plat (éliminer les dimensions superflues)
        base_embedding = np.squeeze(base_embedding)
        # Si PCA est appliqué, on transforme l'embedding (reshape nécessaire pour PCA)
        if self.pca:
            base_embedding = self.pca.transform(base_embedding.reshape(1, -1))[0]
        # Normalisation finale
        normalized = normalize_embeddings(base_embedding)
        return normalized.tolist()

# 🔹 Initialisation des embeddings
embeddings = CustomEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", pca_model_path=PCA_MODEL_PATH)

# -------------------------------
# 📌 Configuration de Pinecone
# -------------------------------
index_name = "education-index"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, text_key='content')

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 7, "score_threshold": 0.5, "include_metadata": True, "include_values": True}  # ✅ Ajout des options
)

# -------------------------------
# 📌 Initialisation du modèle de langage (LLM)
# -------------------------------
hf_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

def call_llm(inputs):
    prompt_text = str(inputs)  # Conversion explicite en chaîne
    response = hf_client.chat_completion(
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.5,
    )
    return response["choices"][0]["message"]["content"]

llm = RunnableLambda(call_llm)  # Rend InferenceClient compatible avec LangChain

# -------------------------------
# 📌 Définition du PromptTemplate
# -------------------------------
prompt_template = PromptTemplate(
    template="""
    Tu es un expert en {expertise}.
    Réponds précisément à la question posée en utilisant uniquement les extraits fournis ci-dessous.
    Explique chaque événement en détaillant ses causes et conséquences immédiates.
    Corrige les erreurs et reformule les réponses de manière claire et exacte.
    Ne fais aucune supposition si les informations ne sont pas dans les extraits.

    Contexte :
    {context}

    Question :
    {question}

    Réponse détaillée et sourcée :
    """,
    input_variables=["expertise", "context", "question"]
)

# -------------------------------
# Utilisation de RunnableSequence pour créer la chaîne
# -------------------------------
llm_chain = prompt_template | llm

# -------------------------------
# 📌 Fonction principale pour poser une question
# -------------------------------
def ask_question(query: str):
    expertise = predict_category(query)
    print(f"🔍 Catégorie détectée : {expertise}")

    expanded_query = expand_query(query)
    docs = retriever.invoke(expanded_query)

    if not docs:
        print("⚠️ Aucun document trouvé correspondant à la requête.")
        return

    context = "\n\n".join([doc.page_content for doc in docs])
    result = llm_chain.invoke({"expertise": expertise, "context": context, "question": expanded_query})

    print("\n📝 Réponse détaillée :\n", result)
    print("\n📌 Sources utilisées :")
    for doc in docs:
        metadata = doc.metadata
        title = metadata.get('title', 'Titre inconnu')
        url = metadata.get('url', 'URL inconnue')
        print(f"- **{title}** ({url})")  
        print(f"  **Extrait :** {doc.page_content[:300]}...\n")

# -------------------------------
# 📌 Interface CLI pour poser une question
# -------------------------------
if __name__ == '__main__':
    question = input("\n🔎 Quelle est votre question ? ")
    ask_question(question)