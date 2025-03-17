# query_with_langchain.py
import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration des clés API
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Configuration des embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configuration de Pinecone
index_name = "education-index"
vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings,
    text_key='content' 
)

# Chargement du modèle Mistral depuis HuggingFace
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    max_new_tokens=1024
)

prompt_template = """
Tu es un enseignant spécialiste en éducation. Réponds à la question posée en t'appuyant uniquement et strictement sur les extraits fournis dans le contexte ci-dessous. Sois extrêmement précis et détaillé, en citant systématiquement les sources utilisées (titre et URL). Si la réponse ne se trouve pas explicitement dans les extraits fournis, dis clairement que tu ne peux pas répondre. N'invente aucune information qui ne figure pas dans le contexte donné.

Contexte disponible :
{context}

Question posée :
{question}

Réponse précise, détaillée et sourcée à partir du contexte uniquement :
"""
PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# Configuration du RetrieverQA avec LangChain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT} 
)

# Fonction pour poser une question et obtenir la réponse du modèle avec le contexte

def ask_question(query):
    result = qa_chain({"query": query})
    print("Réponse:", result["result"])
    print("\nSources utilisées:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['title']} ({doc.metadata['url']})\nContenu : {doc.page_content}\n")

# Exemple d'utilisation
if __name__ == '__main__':
    question = input("Quelle est votre question ? ")
    ask_question(question)
