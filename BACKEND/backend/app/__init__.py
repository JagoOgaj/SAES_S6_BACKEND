import os
from flask import Flask, request, send_from_directory
from dotenv import load_dotenv
from backend.app.core import (
    ENUM_DB_ENV,
    ENUM_FLASK_ENV,
    ENUM_JWT_ENV,
    ENUM_CORS,
    ENUM_URL_PREFIX,
    ENUM_CONFIG_DB_KEY,
)
from flask_cors import CORS
from backend.app.extension import ext
from mongoengine import connect
from backend.app.controllers import bp_user, bp_auth, bp_model
from backend.app.log import logger
from werkzeug.exceptions import (
    HTTPException,
    NotFound,
    BadRequest,
    InternalServerError,
    Forbidden,
    Unauthorized,
    MethodNotAllowed,
)
from flask_socketio import SocketIO
from backend.app.core import create_json_response, get_client_info

load_dotenv()


class Config:
    # Flask #
    FLASK_HOST: str = os.environ.get(ENUM_FLASK_ENV.HOST.value)
    FLASK_DEBUG: str = os.environ.get(ENUM_FLASK_ENV.DEBUG.value)
    FLASK_ENV: str = os.environ.get(ENUM_FLASK_ENV.ENV.value)
    FLASK_PORT: str = os.environ.get(ENUM_FLASK_ENV.PORT.value)

    # DB #
    MONGODB_CONFIG: dict[str, str] = {
        ENUM_CONFIG_DB_KEY.DB.value: os.environ.get(ENUM_DB_ENV.DB_NAME.value),
        ENUM_CONFIG_DB_KEY.HOST.value: os.environ.get(ENUM_DB_ENV.DB_HOST.value),
        ENUM_CONFIG_DB_KEY.PORT.value: os.environ.get(ENUM_DB_ENV.DB_PORT.value),
    }

    # JWT #
    JWT_SECRET_KEY: str = os.environ.get(ENUM_JWT_ENV.SECRET_KEY.value)
    JWT_IDENTITY_CLAIM: str = os.environ.get(ENUM_JWT_ENV.IDENTITY_CLAIM.value)
    JWT_TOKEN_LOCATION: list[str] = [os.environ.get(ENUM_JWT_ENV.TOKEN_LOCATION.value)]


class App:
    @staticmethod
    def create_app() -> Flask:
        app = Flask(__name__)

        # Load Config object #
        app.config.from_object(Config)

        # Connect to MongoDB #
        connect(
            db=app.config["MONGODB_CONFIG"][ENUM_CONFIG_DB_KEY.DB.value],
            host=app.config["MONGODB_CONFIG"][ENUM_CONFIG_DB_KEY.HOST.value],
            port=int(app.config["MONGODB_CONFIG"][ENUM_CONFIG_DB_KEY.PORT.value]),
        )

        # Cors #
        CORS(app, resources=ENUM_CORS.RESSOURCE.value)

        # Extensions #
        ext.ma_ext.init_app(app)
        ext.jwt_ext.init_app(app)

        # Routes with blueprint #
        app.register_blueprint(bp_auth, url_prefix=ENUM_URL_PREFIX.AUTH.value)
        app.register_blueprint(bp_user, url_prefix=ENUM_URL_PREFIX.USER.value)
        app.register_blueprint(bp_model, url_prefix=ENUM_URL_PREFIX.MODEL.value)

        # Middleware Request Handler #
        @app.before_request
        def log_request_info():
            client_ip, region, device = get_client_info()
            logger.info(
                f"Request: {request.method} {request.url}  - IP: {client_ip} - Region: {region} - Device: {device}"
            )

        @app.after_request
        def log_response_info(response):
            client_ip, region, device = get_client_info()
            logger.info(
                f"Response: {response.status}  - IP: {client_ip} - Region: {region} - Device: {device}"
            )
            return response

        @app.errorhandler(NotFound)
        def handle_404(e):
            client_ip, region, device = get_client_info()
            logger.warning(
                f"404 Not Found: {request.url} - IP: {client_ip} - Region: {region} - Device: {device}"
            )
            return create_json_response(
                error="Not Found",
                message="La ressource demandée est introuvable.",
                status_code=404,
            )

        @app.errorhandler(BadRequest)
        def handle_400(e):
            client_ip, region, device = get_client_info()
            logger.warning(
                f"400 Bad Request: {request.url} - IP: {client_ip} - Region: {region} - Device: {device}"
            )
            return create_json_response(
                error="Bad Request",
                message="La requête est invalide ou mal formée.",
                status_code=400,
            )

        @app.errorhandler(Unauthorized)
        def handle_401(e):
            client_ip, region, device = get_client_info()
            logger.warning(
                f"401 Unauthorized: {request.url} - IP: {client_ip} - Region: {region} - Device: {device}"
            )
            return create_json_response(
                error="Unauthorized",
                message="Authentification requise pour accéder à cette ressource.",
                status_code=401,
            )

        @app.errorhandler(Forbidden)
        def handle_403(e):
            client_ip, region, device = get_client_info()
            logger.warning(
                f"403 Forbidden: {request.url} - IP: {client_ip} - Region: {region} - Device: {device}"
            )
            return create_json_response(
                error="Forbidden",
                message="Vous n'avez pas les permissions pour accéder à cette ressource.",
                status_code=403,
            )

        @app.errorhandler(MethodNotAllowed)
        def handle_405(e):
            client_ip, region, device = get_client_info()
            logger.warning(
                f"405 Method Not Allowed: {request.url} - IP: {client_ip} - Region: {region} - Device: {device}"
            )
            return create_json_response(
                error="Method Not Allowed",
                message="La méthode HTTP utilisée n'est pas autorisée pour cette route.",
                status_code=405,
            )

        @app.errorhandler(InternalServerError)
        def handle_500(e):
            client_ip, region, device = get_client_info()
            logger.error(
                f"500 Internal Server Error: {request.url} - IP: {client_ip} - Region: {region} - Device: {device}",
                exc_info=True,
            )
            return create_json_response(
                error="Internal Server Error",
                message="Une erreur interne est survenue sur le serveur.",
                status_code=500,
            )

        @app.errorhandler(HTTPException)
        def handle_http_exception(e):
            client_ip, region, device = get_client_info()
            logger.error(
                f"{e.code} {e.name}: {request.url} - IP: {client_ip} - Region: {region} - Device: {device}"
            )
            return create_json_response(
                error="Http Exception error",
                mesasge="Une erreur est survenue sur le serveur",
                status_code=500,
            )

        @app.errorhandler(Exception)
        def handle_exception(e):
            client_ip, region, device = get_client_info()
            logger.error(
                f"Unhandled Exception: {str(e)} - IP: {client_ip} - Region: {region} - Device: {device}",
                exc_info=True,
            )
            return create_json_response(
                erorr="Internal Server Error",
                message="Une erreur inattendue est survenue.",
                status_code=500,
            )

        return app


app = App.create_app()
