import os
from typing import Self, List
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
from transformers import pipeline 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.app.core.utility import create_json_response
from backend.app.exeptions import ModelTypeNotFoundError
from dotenv import load_dotenv

load_dotenv()

class Service_MODEL:
    def __init__(self: Self, typeModel: str) -> None:
        self.typeModel = typeModel
        self.target_dimension = 768  # Taille cible fixée à 768
        
        self.hf_client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = PineconeVectorStore(
            index_name="education-index", 
            embedding=self.embeddings, 
            text_key='content'
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",  # Recherche rapide par similarité approximative
            search_kwargs={
                "k": 5,  # Réduit le nombre de documents retournés pour améliorer la vitesse
                "score_threshold": 0.5  # Filtre les documents moins pertinents pour optimiser la recherche
            }
        )
        
        # Initialisation de l'historique de conversation avec un message système dédié
        self.messages = [
            {"role": "system", "content": "Tu es un chatbot expert dans le domaine de l'éducation. Tu fournis des réponses précises, détaillées et sourcées aux questions posées, en t'appuyant sur des extraits fiables et des connaissances approfondies dans ce domaine."}
        ]
        
        # Prompt template inchangé
        self.prompt_template = PromptTemplate(
            template="""
            Tu es un expert en {expertise}.
            Réponds précisément à la question posée en utilisant uniquement les extraits fiables fournis ci-dessous.
            Explique chaque événement en détaillant ses causes et conséquences immédiates.
            Corrige les erreurs et reformule les réponses de manière claire et exacte.
            Évite d'inventer des informations ou de faire des suppositions.

            Contexte :
            {context}

            Question :
            {question}

            Réponse détaillée, précise et sourcée :
            """,
            input_variables=["expertise", "context", "question"]
        )
    
    def extract_text_from_document(self, file) -> str:
        """Extrait le texte d'un document (PDF, TXT, DOCX, ou image)."""
        try:
            if file.filename.endswith(".pdf"):
                reader = PdfReader(file)
                return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif file.filename.endswith(".txt"):
                return file.read().decode("utf-8")
            elif file.filename.endswith(".docx"):
                doc = Document(file)
                return " ".join([paragraph.text for paragraph in doc.paragraphs])
            elif file.filename.endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(file)
                return pytesseract.image_to_string(image)
            else:
                raise ValueError("Format de fichier non pris en charge.")
        except Exception as e:
            raise Exception(f"Erreur lors de l'extraction du texte : {str(e)}")
    
    def predict_category(self, question: str) -> str:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = ["histoire", "mathématiques", "sciences", "philosophie", "économie", "éducation"]
        result = classifier(question, candidate_labels)
        return f"éducation, {result['labels'][0]}"
    
    def handle_prediction(self: Self, user_input: str, documents: List = None) -> str:
        # Extraction et concaténation du texte provenant des documents (si présents)
        combined_document_text = ""
        if documents:
            document_texts = [self.extract_text_from_document(doc) for doc in documents]
            combined_document_text = " ".join(document_texts)

        # Récupération des documents pertinents via le retriever
        docs = self.retriever.invoke(user_input)
        context = combined_document_text + "\n\n" if combined_document_text else ""
        context += "\n\n".join([doc.page_content for doc in docs])

        # Identification de l'expertise via le classificateur
        expertise = self.predict_category(user_input)

        # Préparation du prompt utilisateur en incluant l'expertise et le contexte
        if context:
            user_input = f"Expertise : {expertise}\n\nContexte fourni : {context}\n\nQuestion : {user_input}"
        else:
            user_input = f"Expertise : {expertise}\n\nQuestion : {user_input}"
        
        # On ajoute l'entrée utilisateur comme message "user"
        self.messages.append({"role": "user", "content": user_input})
        
        # Appel au modèle en utilisant l'historique complet
        try:
            response = self.hf_client.chat_completion(
                messages=self.messages,
                temperature=0.5,
                max_tokens=1000
            )
            assistant_reply = response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Erreur lors de l'appel au modèle : {str(e)}"

        # Ajout de la réponse du modèle à l'historique
        self.messages.append({"role": "assistant", "content": assistant_reply})

        # Récupération des liens sources issus des documents, s'ils existent
        video_links = []
        for doc in docs:
            if 'url' in doc.metadata:
                video_links.append(doc.metadata['url'])
        if video_links:
            assistant_reply += "\n\nSources utilisées :\n" + "\n".join(video_links)

        return assistant_reply