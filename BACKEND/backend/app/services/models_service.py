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
        self.target_dimension = 768 
        
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
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self.messages = [
            {
                "role": "system", 
                "content": (
                    "Tu es un chatbot expert dans le domaine de l'éducation. Ta mission est de répondre aux demandes "
                    "des utilisateurs de manière naturelle et conversationnelle. Adapte ton style en fonction de la nature de la demande :\n"
                    " - Si l'utilisateur pose une question pédagogique, appuie-toi sur le contexte externe (base Pinecone, documents) pour fournir une réponse détaillée, sourcée et précise.\n"
                    " - Sinon, répond de manière conviviale tout en précisant que tes réponses se concentrent principalement sur l'éducation."
                )
            }
        ]
        self.max_history_messages = 10


        self.prompt_template = PromptTemplate(
            template="""
            Tu es un chatbot expert dans le domaine de l'éducation.
            Tâche : adapte ton style de réponse en fonction de la nature de l'entrée de l'utilisateur.
            - Si l'entrée est une question pédagogique, utilise le contexte suivant pour fournir une réponse détaillée, sourcée et précise.
            - Sinon, répond de manière naturelle et conviviale sans intégrer le contexte externe.
 
            Contexte (optionnel) :
            {context}

            Message de l'utilisateur :
            {question}

            Réponse :
            """,
            input_variables=["context", "question"]
        )
    
    def extract_text_from_document(self, file) -> str:
        """Extrait le texte d'un document (PDF, TXT, DOCX ou image)."""
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
    
    def is_greeting(self, message: str) -> bool:
        """Vérifie si le message est une salutation ou très court."""
        greetings = {"bonjour", "salut", "hello", "coucou", "bonsoir"}
        tokens = set(message.strip().lower().split())
        return len(tokens.intersection(greetings)) > 0 or len(message.strip().split()) <= 2

    def is_recall_query(self, message: str) -> bool:
        """Détecte si le message demande un rappel (ex. : 'Quelle était ma dernière question ?')."""
        lowered = message.lower()
        return "dernière question" in lowered or "précédente" in lowered

    def predict_category(self, question: str) -> bool:
        """
        Utilise un classificateur zéro-shot pour déterminer si la question porte sur l'éducation.
        Renvoie True si le label "éducation" obtient un score significatif.
        Vous pouvez ajuster le seuil ici si besoin.
        """
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = ["éducation", "non-éducation"]
        result = classifier(question, candidate_labels)
        education_score = 0.0
        non_education_score = 0.0
        for label, score in zip(result["labels"], result["scores"]):
            if label.lower() in ["éducation", "education"]:
                education_score = score
            else:
                non_education_score = score
    
        return education_score > non_education_score and education_score > 0.65

    def update_summary_buffer(self) -> None:
        """
        Si l'historique dépasse le seuil, génère un résumé synthétique et réinitialise l'historique
        en conservant le message système initial.
        """
        conversation_text = "\n".join(
            [f"{msg['role']} : {msg['content']}" for msg in self.messages if msg['role'] != "system"]
        )
        summary_prompt = (
            "Synthétise de manière concise la conversation suivante en extrayant les points clés :\n\n"
            + conversation_text
        )
        try:
            summary_response = self.hf_client.chat_completion(
                messages=[{"role": "system", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=300
            )
            summary = summary_response["choices"][0]["message"]["content"]
        except Exception as e:
            summary = "Impossible de générer un résumé de la conversation en raison d'une erreur."
            print(f"Erreur lors du résumé : {str(e)}")
        
        system_message = self.messages[0]
        self.messages = [
            system_message,
            {"role": "assistant", "content": f"Résumé de la conversation précédente : {summary}"}
        ]
    
    def handle_prediction(self: Self, user_input: str, documents: List = None) -> str:
        if self.is_greeting(user_input):
            greeting_response = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "assistant", "content": greeting_response})
            return greeting_response


        if self.is_recall_query(user_input):
            last_question = None
            for msg in reversed(self.messages):
                if msg["role"] == "user" and not self.is_recall_query(msg["content"]):
                    last_question = msg["content"]
                    break
            if last_question:
                recall_response = f"Votre dernière question était : {last_question}"
            else:
                recall_response = "Je n'ai pas de question précédente enregistrée."
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "assistant", "content": recall_response})
            return recall_response

        if self.predict_category(user_input):
            extra_instructions = (
                "Réponds de manière naturelle, conversationnelle et experte. "
                "Utilise le contexte s'il est pertinent et cite les sources le cas échéant."
            )
        else:
            extra_instructions = (
                "Note : Bien que votre question ne semble pas directement liée à l'éducation, "
                "je vais tenter d'y répondre en m'appuyant sur mes connaissances en éducation."
            )
    
        combined_document_text = ""
        if documents:
            document_texts = [self.extract_text_from_document(doc) for doc in documents]
            combined_document_text = " ".join(document_texts)

        docs = self.retriever.invoke(user_input)
        pinecone_context = "\n\n".join([doc.page_content for doc in docs])
        

        if combined_document_text:
            context = combined_document_text + "\n\n" + pinecone_context
        else:
            context = pinecone_context
        if not context.strip():
            context = "Aucun contexte supplémentaire n'est disponible."
        
        formatted_prompt = self.prompt_template.format(
            context=context,
            question=user_input,
            extra_instructions=extra_instructions  
                                                
        )
        self.messages.append({"role": "user", "content": formatted_prompt})
        
        if len(self.messages) > self.max_history_messages:
            self.update_summary_buffer()

        try:
            response = self.hf_client.chat_completion(
                messages=self.messages,
                temperature=0.5,
                max_tokens=1000
            )
            assistant_reply = response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Erreur lors de l'appel au modèle : {str(e)}"

        self.messages.append({"role": "assistant", "content": assistant_reply})

        video_links = []
        docs = self.retriever.invoke(user_input)
        for doc in docs:
            if 'url' in doc.metadata:
                video_links.append(doc.metadata['url'])
        if video_links:
            assistant_reply += "\n\nSources utilisées :\n" + "\n".join(video_links)

        return assistant_reply