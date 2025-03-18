import argparse
import os
import re
import psycopg2
import subprocess
import tempfile
import yt_dlp
import whisper
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from threading import local
import stanza

load_dotenv()

stanza.download('fr')
nlp = stanza.Pipeline('fr', processors='tokenize')

DetectorFactory.seed = 0  
thread_local = local()   

DB_PARAMS = {
    "database": os.environ.get("database"),
    "user": os.environ.get("user"),
    "password": os.environ.get("password"),
    "host": os.environ.get("host"),
}

def get_model():
    """Charge le modèle Whisper une seule fois par thread"""
    if not hasattr(thread_local, "model"):
        thread_local.model = whisper.load_model("small", device="cpu")
    return thread_local.model

def scrape_video_title(video_url):
    """Scrape le titre de la vidéo YouTube"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(video_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                return title_tag.text.replace(" - YouTube", "").strip()
        return "Titre inconnu"
    except Exception as e:
        print(f"Erreur lors du scraping du titre : {e}")
        return "Erreur titre"

def create_db():
    """Crée la table Documents dans PostgreSQL si elle n'existe pas"""
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

def insert_document(cursor, video_id, url, title, language, content):
    """Insère un document dans la base PostgreSQL"""
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

def split_audio(input_file, output_folder, segment_duration=30):
    """Découpe un fichier audio en segments de 30s avec ffmpeg"""
    os.makedirs(output_folder, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", input_file, "-f", "segment", "-segment_time", str(segment_duration),
        "-c", "copy", os.path.join(output_folder, "segment_%02d.wav")
    ], check=True)

def transcribe_video_asr(video_url):
    """
    Télécharge la vidéo, découpe en segments de 30s,
    transcrit chaque segment avec Whisper, puis recolle les morceaux.
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
        
        audio_path = os.path.join(temp_dir, f"{video_id}.wav")
        output_folder = os.path.join(temp_dir, "segments")
        
        split_audio(audio_path, output_folder) 
        
        transcript_segments = []
        model_instance = get_model()

        for segment in sorted(os.listdir(output_folder)):
            segment_path = os.path.join(output_folder, segment)
            result = model_instance.transcribe(segment_path, language="fr", verbose=False)
            transcript_segments.append(result["text"])

        return " ".join(transcript_segments).strip()

def clean_text(text):
    """
    Nettoie le texte en supprimant les caractères spéciaux inutiles,
    les espaces multiples et les répétitions de phrases.
    """
    text = re.sub(r"[^A-Za-z0-9À-ÿ .,;!?'-]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    return " ".join(unique_sentences)

def post_process_output(text):
    """
    Applique un post-traitement pour améliorer l'output :
    - Segmentation du texte avec Stanza
    - Correction de la capitalisation (première lettre de chaque phrase en majuscule)
    """
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    processed_sentences = [s.capitalize() for s in sentences]
    return " ".join(processed_sentences)

def extract_id_from_url(url):
    """Extrait l'ID d'une vidéo YouTube depuis l'URL"""
    base_url = url.split("&")[0]
    match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([^&/?]+)", base_url)
    return match.group(1) if match else None

def process_video(video_url):
    """Télécharge, transcrit, nettoie, post-traite et stocke une vidéo YouTube"""
    video_id = extract_id_from_url(video_url)
    if not video_id:
        print(f"Impossible d'extraire l'ID de {video_url}")
        return

    conn, cursor = create_db()
    title = scrape_video_title(video_url)
    full_content = transcribe_video_asr(video_url)

    if not full_content:
        print(f"Aucun contenu transcript pour {video_url}")
        conn.close()
        return

    cleaned_content = clean_text(full_content)
    final_content = post_process_output(cleaned_content)

    try:
        language = detect(final_content)
    except Exception:
        language = "inconnu"

    insert_document(cursor, video_id, video_url, title, language, final_content)
    conn.commit()
    conn.close()
    print(f"Vidéo {video_id} traitée et enregistrée.")

def main():
    parser = argparse.ArgumentParser(description="Crawl et transcription de vidéos YouTube")
    parser.add_argument("--input-file", "-i", required=True, help="Fichier contenant les URLs")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_video, urls), total=len(urls), desc="Traitement des vidéos"))

if __name__ == "__main__":
    main()