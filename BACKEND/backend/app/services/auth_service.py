from backend.app.extension import ext
from flask import render_template
from flask_jwt_extended import create_access_token, create_refresh_token
from backend.app.exeptions.custom_exeptions import (
    UserNotFound,
    UserPasswordNotFound,
    PayloadError,
    EmailAlreadyUsed,
)
from backend.app.core.const import ENUM_RESET_URL
from typing import Tuple
from backend.app.services.db_service import service_db
from backend.app.services.jwt_service import service_jwt
from backend.app.core.utility import send_reset_password_email
from datetime import timedelta


class Service_AUTH:
    def __init__(self) -> None:
        pass

    def login(self, data: dict[...:...]) -> Tuple[str, str]:
        if (email := data.get("email")) and (password := data.get("password")):
            user = service_db.find_user_by_filters(email=email)

            if not user:
                raise UserNotFound(email=email)

            if not ext.pwd_context_ext.verify(password, user.password_hash):
                raise UserPasswordNotFound(user.user_id)

            service_jwt.revoke_all_tokens(user_id=user.user_id)

            access_token = create_access_token(
                identity=user.user_id, expires_delta=timedelta(hours=1)
            )
            refresh_token = create_refresh_token(identity=user.user_id)

            service_jwt.add_token_to_database(access_token)
            service_jwt.add_token_to_database(refresh_token)

            return access_token, refresh_token

        raise PayloadError("email", "password")

    def registry(self, data: dict[...:...]) -> None:
        if email := data.get("email"):
            user = service_db.find_user_by_filters(email=email)

            if user:
                raise EmailAlreadyUsed(email)

            user = service_db.create_user(data)

            access_token = create_access_token(
                identity=user.user_id, expires_delta=timedelta(hours=1)
            )
            refresh_token = create_refresh_token(identity=user.user_id)

            service_jwt.add_token_to_database(access_token)
            service_jwt.add_token_to_database(refresh_token)

            return access_token, refresh_token

    def getNewAccessToken(self, user_id: int) -> str:
        newTokenAccess = create_access_token(
            identity=user_id, expires_delta=timedelta(hours=1)
        )

        service_jwt.add_token_to_database(newTokenAccess)

        return newTokenAccess

    def reset_password(self, user_id: int, new_password: str) -> None:
        try:
            user = service_db.find_user_by_filters(user_id=user_id)
            if not user:
                raise UserNotFound(user_id=user_id)

            user.password_hash = ext.pwd_context_ext.hash(new_password)
            service_db.add_to_db(user)
        except Exception as e:
            raise Exception(
                f"Une erreur est survenue lors de la réinitialisation du mot de passe : {str(e)}"
            )

    def send_reset_password_email(self, email: str) -> None:
        user = service_db.find_user_by_filters(email=email)
        if not user:
            raise UserNotFound(email=email)
        service_jwt.revoke_all_tokens(user.user_id)
        reset_token = create_access_token(
            identity=user.user_id,
            expires_delta=timedelta(minutes=15),
            additional_claims={"type": "reset_password"},
        )
        reset_url = f"{ENUM_RESET_URL.LOCAL.value}{reset_token}"
        html_content = render_template(
            "reset_password_email.html", reset_url=reset_url, username=user.username
        )
        send_reset_password_email(to=email, html_content=html_content)

    def is_reset_password_token(self, token) -> bool:
        return token.get("type") == "reset_password"


service_auth: Service_AUTH = Service_AUTH()
