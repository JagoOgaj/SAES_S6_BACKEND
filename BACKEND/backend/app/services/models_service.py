import os
from typing import Self, List
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
from transformers import pipeline 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.app.core.utility import create_json_response
from backend.app.exeptions import ModelTypeNotFoundError
from dotenv import load_dotenv
import faiss 
import json
import numpy as np
from langchain.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_pinecone import PineconeVectorStore  # Import de Pinecone

load_dotenv()

class Service_MODEL:
    def __init__(self: Self, typeModel: str) -> None:
        self.typeModel = typeModel
        self.target_dimension = 768 
        
        # Initialisation des embedders pour la recherche
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.embeddings2 = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
        
        # Initialisation de la recherche pour la base YouTube via Pinecone
        # Assure-toi que l'index Pinecone est créé et que la variable d'environnement PINECONE_API_KEY est définie.
        index_name = "education-index"
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=self.embeddings, text_key='content')
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 7, "score_threshold": 0.5, "include_metadata": True, "include_values": True}
        )
        
        # Recherche pour les PDF (nous continuons d'utiliser FAISS pour les PDF)
        # Ces fichiers (index.faiss, etc.) sont générés séparément pour les PDF.
    
        # Message système initial
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
        
        # Mémoire de conversation (fenêtre de 6 échanges)
        self.memory = ConversationBufferWindowMemory(k=6, memory_key="chat_history", return_messages=True)
    
        self.prompt_template = PromptTemplate(
            template="""
            Tu es un chatbot expert dans le domaine de l'éducation.
            Tâche : adapte ton style de réponse en fonction de la nature de l'entrée de l'utilisateur.
            - Si l'entrée est une question pédagogique, utilise le contexte ci-dessous pour fournir une réponse détaillée, sourcée et précise.
            - Sinon, répond de manière naturelle et conviviale.
 
            Historique (limité) :
            {chat_history}

            Contexte supplémentaire :
            {context}

            Message de l'utilisateur :
            {question}

            Réponse :
            """,
            input_variables=["chat_history", "context", "question"]
        )
        
        self.max_history_messages = 2

    def extract_text_from_document(self, file) -> str:
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
        greetings = {"bonjour", "salut", "hello", "coucou", "bonsoir"}
        tokens = set(message.strip().lower().split())
        return len(tokens.intersection(greetings)) > 0 or len(message.strip().split()) <= 2

    def is_recall_query(self, message: str) -> bool:
        lowered = message.lower()
        return "dernière question" in lowered or "précédente" in lowered

    def predict_category(self, question: str) -> bool:
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

    def clean_text(self, text: str) -> str:
        import re
        text = re.sub(r'\.{2,}', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def search_with_pinecone(self, query: str, k: int = 5) -> List[str]:
        # Utilise le retriever Pinecone pour récupérer les documents de la base YouTube
        # Le retriever renvoie une liste d'objets ayant la propriété page_content
        docs = self.retriever.invoke(query)
        return [self.clean_text(doc.page_content) for doc in docs]

    def search_pdf_faiss(self, query: str, k: int = 5) -> List[str]:
        try:
            pdf_vectorstore = FAISS.load_local("BACKEND/db-pdf", self.embeddings2, allow_dangerous_deserialization=True)
            docs = pdf_vectorstore.similarity_search(query, k=k)
            return [self.clean_text(doc.page_content) for doc in docs]
        except Exception as e:
            print(f"Erreur lors de la recherche PDF FAISS: {str(e)}")
            return []
    
    def handle_prediction(self: Self, user_input: str, documents: List = None) -> str:
        if self.is_greeting(user_input):
            greeting_response = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
            self.memory.save_context({"input": user_input}, {"output": greeting_response})
            return greeting_response

        if self.is_recall_query(user_input):
            mem_variables = self.memory.load_memory_variables({})
            last_history = mem_variables.get("chat_history", "")
            recall_response = f"Voici ce que je me souviens : {last_history}" if last_history else "Je n'ai aucun historique enregistré."
            self.memory.save_context({"input": user_input}, {"output": recall_response})
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

        # Préparer le contexte
        if documents:
            # Utiliser uniquement les documents fournis par l'utilisateur
            document_texts = [self.extract_text_from_document(doc) for doc in documents]
            context = self.clean_text(" ".join(document_texts))
        else:
            # Fusionner le contexte provenant de la base Pinecone (vidéo youtube) et des PDF stockés dans la base
            retrieved_texts = self.search_with_pinecone(user_input)
            current_context = "\n\n".join(retrieved_texts)
            current_context = self.clean_text(current_context)

            pdf_texts = self.search_pdf_faiss(user_input)
            pdf_context = "\n\n".join(pdf_texts)
            pdf_context = self.clean_text(pdf_context)

            if current_context and pdf_context:
                context = current_context + "\n\n---\n\n" + pdf_context
            elif current_context:
                context = current_context
            elif pdf_context:
                context = pdf_context
            else:
                context = "Aucun contexte supplémentaire n'est disponible."

        # Charger l'historique de conversation depuis la mémoire
        mem_variables = self.memory.load_memory_variables({})
        chat_history = mem_variables.get("chat_history", "")

        formatted_prompt = self.prompt_template.format(
            chat_history=chat_history,
            context=context,
            question=user_input
        )

        try:
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": self.messages[0]["content"]},
                    {"role": "user", "content": formatted_prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 1000
            }
            headers = {
                "Authorization": f"Bearer {os.environ['DEEPINFRA_API_KEY']}",
                "Content-Type": "application/json"
            }
            url = "https://api.deepinfra.com/v1/openai/chat/completions"
            res = requests.post(url, headers=headers, json=payload)
            res_json = res.json()
            if "choices" in res_json and len(res_json["choices"]) > 0:
                assistant_reply = res_json["choices"][0]["message"]["content"]
            else:
                print("Réponse inattendue de DeepInfra :", res_json)
                assistant_reply = "Le modèle n'a pas pu générer de réponse. Veuillez réessayer plus tard."
        except Exception as e:
            return f"Erreur lors de l'appel au modèle : {str(e)}"
        self.memory.save_context({"input": user_input}, {"output": assistant_reply})
        return assistant_reply