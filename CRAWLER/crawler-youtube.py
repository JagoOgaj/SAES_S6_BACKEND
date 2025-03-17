import argparse
import re
import psycopg2
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, tempfile
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
import yt_dlp
import whisper
from threading import local
thread_local = local()

DetectorFactory.seed = 0  # Pour rendre la détection déterministe
load_dotenv()


def get_model():
    if not hasattr(thread_local, "model"):
        # Charge le modèle une seule fois par thread
        thread_local.model = whisper.load_model("small", device="cpu")
    return thread_local.model

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
# Web scraping pour récupérer le titre d'une vidéo
# -------------------------------------------------------------------
def scrape_video_title(video_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(video_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                return title_tag.text.replace(" - YouTube", "").strip()
        return ""
    except Exception as e:
        print(f"Erreur lors du scraping du titre pour {video_url}: {e}")
        return ""

# -------------------------------------------------------------------
# Création de la table Documents dans PostgreSQL
# -------------------------------------------------------------------
def create_db():
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Documents (
            document_id SERIAL PRIMARY KEY,
            video_id VARCHAR(255) UNIQUE NOT NULL,
            url TEXT,
            title TEXT,
            language VARCHAR(10),
            content TEXT
        );
    """)
    conn.commit()
    return conn, cursor

# -------------------------------------------------------------------
# Insertion d'un document dans PostgreSQL
# -------------------------------------------------------------------
def insert_document(cursor, video_id, url, title, language, content):
    cursor.execute("""
        INSERT INTO Documents (video_id, url, title, language, content)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO NOTHING
        RETURNING document_id;
    """, (video_id, url, title, language, content))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("SELECT document_id FROM Documents WHERE video_id = %s;", (video_id,))
        return cursor.fetchone()[0]

# -------------------------------------------------------------------
# Transcription via Whisper (installation depuis GitHub)
# -------------------------------------------------------------------
def transcribe_video_asr(video_url):
    """
    Télécharge l'audio de la vidéo avec yt-dlp dans un dossier temporaire
    et transcrit l'audio en utilisant Whisper.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'postprocessors': [{
                 'key': 'FFmpegExtractAudio',
                 'preferredcodec': 'wav',
                 'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_id = info_dict.get("id", None)
            ydl.download([video_url])
        
        downloaded_file = None
        for filename in os.listdir(temp_dir):
            if filename.startswith(video_id):
                downloaded_file = os.path.join(temp_dir, filename)
                break
        
        if not downloaded_file:
            raise Exception("Fichier audio non trouvé dans le dossier temporaire")
        
        if os.path.getsize(downloaded_file) == 0:
            raise Exception("Le fichier audio téléchargé est vide")
        
        # Utilisation de l'instance du modèle propre au thread
        model_instance = get_model()
        result = model_instance.transcribe(downloaded_file, language="fr", verbose=False)
        
        transcript = result["text"]
        return transcript
# -------------------------------------------------------------------
# Récupération du transcript via ASR
# -------------------------------------------------------------------
def get_full_transcript(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    print("Utilisation de Whisper pour transcrire la vidéo...")
    try:
        return transcribe_video_asr(video_url)
    except Exception as e:
        print(f"Erreur lors de la transcription pour {video_url}: {e}")
        return ""

# -------------------------------------------------------------------
# Extraction de l'ID d'une URL vidéo (garde uniquement la partie avant le premier '&')
# -------------------------------------------------------------------
def extract_id_from_url(url):
    base_url = url.split('&')[0]
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([^&/?]+)", base_url)
    if m:
        return m.group(1)
    return None

# -------------------------------------------------------------------
# Traitement d'une vidéo (sans calcul d'embedding)
# -------------------------------------------------------------------
def process_video(video_url):
    video_id = extract_id_from_url(video_url)
    if not video_id:
        print(f"Impossible d'extraire l'ID de {video_url}")
        return
    conn, cursor = create_db()
    title = scrape_video_title(video_url)
    full_content = get_full_transcript(video_id)
    if not full_content:
        print(f"Aucun contenu transcript pour la vidéo {video_url}")
        conn.close()
        return
    try:
        language = detect(full_content)
    except Exception:
        language = ""
    document_id = insert_document(cursor, video_id, video_url, title, language, full_content)
    conn.commit()
    conn.close()

# -------------------------------------------------------------------
# Fonction principale : lecture d'un fichier d'URL et lancement des crawlers
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Population de la base de données Documents")
    parser.add_argument("--input-file", "-i", help="Fichier contenant une liste d'URL vidéo (une par ligne)", required=True)
    args = parser.parse_args()
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    cleaned_urls = []
    for url in urls:
        video_id = extract_id_from_url(url)
        if video_id:
            cleaned_urls.append(f"https://www.youtube.com/watch?v={video_id}")
        else:
            print(f"URL ignorée (non valide) : {url}")
    
    print(f"Population de la DB pour {len(cleaned_urls)} vidéos...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_video, url): url for url in cleaned_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Traitement global des vidéos"):
            try:
                future.result()
            except Exception as e:
                print(f"Erreur lors du traitement de {futures[future]}: {e}")

if __name__ == '__main__':
    main()