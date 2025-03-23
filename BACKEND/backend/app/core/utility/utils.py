import pytz
from datetime import datetime, timedelta
from backend.app.core import ENUM_TIMEZONE
from flask import jsonify, request
import base64
import numpy as np
import requests
from user_agents import parse
from backend.app.log import logger


def get_paris_time():
    """
    Retourne l'heure actuelle dans le fuseau horaire de Paris.

    Returns:
        datetime: L'heure actuelle dans le fuseau horaire de Paris.
    """
    paris_tz = pytz.timezone(ENUM_TIMEZONE.TIMEZONE_PARIS.value)
    return datetime.now(paris_tz)


def create_json_response(status_code=200, **kwargs):
    """
    Crée une réponse JSON avec un code de statut HTTP.

    Args:
        status_code (int, optional): Le code de statut HTTP de la réponse. Par défaut à 200.
        **kwargs: Les données à inclure dans la réponse JSON.

    Returns:
        Response: Une réponse Flask contenant les données JSON et le code de statut.
    """
    response = jsonify(kwargs)
    response.status_code = status_code
    return response


def get_client_info():
    """
    Récupère les informations du client à partir de la requête en cours.

    Cette fonction extrait l'adresse IP du client, analyse l'en-tête User-Agent pour obtenir des informations sur l'appareil,
    et utilise un service de géolocalisation pour déterminer la région du client.

    Returns:
        tuple: Un tuple contenant l'adresse IP du client (str), la région (str) et les informations sur l'appareil (str).
    """
    client_ip = request.remote_addr
    user_agent = parse(request.headers.get("User-Agent"))
    try:
        response = requests.get(f"http://ipinfo.io/{client_ip}/json")
        region = response.json().get("region", "Unknown")
    except Exception as e:
        logger.error(f"Error fetching region info: {str(e)}")
        region = "Unknown"

    device = f"{user_agent.device.family} {user_agent.device.brand} {user_agent.device.model}"
    return client_ip, region, device


def send_reset_password_email(to: str, html_content: str) -> None:
    """
    Envoie un email de réinitialisation de mot de passe avec un contenu HTML.

    Args:
        to (str): L'adresse email du destinataire.
        html_content (str): Le contenu HTML de l'email.

    Raises:
        smtplib.SMTPAuthenticationError: Si une erreur d'authentification SMTP se produit.
        smtplib.SMTPServerDisconnected: Si la connexion au serveur SMTP est fermée.
        Exception: Pour toute autre erreur qui se produit lors de l'envoi de l'email.

    Cette fonction utilise les variables d'environnement suivantes :
        - EMAIL_SAES: L'adresse email de l'expéditeur.
        - PWD_EMAIL: Le mot de passe de l'expéditeur.
        - SMTP_SERVER: L'adresse du serveur SMTP.
        - SMTP_PORT: Le port du serveur SMTP.
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from dotenv import load_dotenv
    import os

    load_dotenv()

    sender_email = os.environ.get("EMAIL_SAES")
    sender_password = os.environ.get("PWD_EMAIL")
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = int(os.environ.get("SMTP_PORT"))

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Réinitialisation de mot de passe"
    msg["From"] = sender_email
    msg["To"] = to

    part = MIMEText(html_content, "html")
    msg.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to, msg.as_string())
    except smtplib.SMTPAuthenticationError as e:
        raise smtplib.SMTPAuthenticationError(f"Erreur d'authentification : {e}")
    except smtplib.SMTPServerDisconnected as e:
        raise smtplib.SMTPServerDisconnected(
            f"Erreur : Connexion au serveur SMTP fermée. {e}"
        )
    except Exception as e:
        raise Exception(f"Erreur {e}")
