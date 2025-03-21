from flask import current_app as app
from flask_jwt_extended import decode_token
import pytz
from backend.app.models.models import MODEL_TokenBlockList
from backend.app.core.const.enum import (
    ENUM_DECODED_TOKEN_KEY,
    ENUM_JWT_ENV,
    ENUM_TIMEZONE,
)
from backend.app.services.db_service import service_db
from backend.app.core.utility.utils import get_paris_time
from mongoengine.errors import DoesNotExist
from backend.app.extension.extensions import ext
import datetime


class Service_JWT:
    def __init__(self) -> None:
        pass

    def add_token_to_database(self, encoded_token: str) -> None:
        decoded_token = decode_token(encoded_token)

        db_token = MODEL_TokenBlockList(
            token_id=service_db.get_next_sequence_value("token_id"),
            jti=decoded_token[ENUM_DECODED_TOKEN_KEY.JTI.value],
            token_type=decoded_token[ENUM_DECODED_TOKEN_KEY.TYPE.value],
            user_id=decoded_token[app.config.get(ENUM_JWT_ENV.IDENTITY_CLAIM.value)],
            expires=datetime.datetime.fromtimestamp(decoded_token["exp"]),
        )

        service_db.add_to_db(db_token)

    def revoke_token(self, token_jti, user_id) -> None:
        try:
            token = service_db.find_token_by_filters(jti=token_jti, user_id=user_id)
            token.is_revoked = True
            service_db.add_to_db(token)

        except DoesNotExist as e:
            raise DoesNotExist(
                f"Aucun token trouver avec token_jti = {token_jti}, user_id = {user_id} - \n {e}"
            )

        except Exception as e:
            raise Exception(f"Une erreur est survenu - \n {e}")

    def revoke_all_tokens(self, user_id: int) -> None:
        tokens = service_db.find_token_by_filters(multiple=True, user_id=user_id)
        for token in tokens:
            token.is_revoked = True
            service_db.add_to_db(token)

    def is_token_revoked(self, jwt_payload) -> bool:
        jti = jwt_payload[ENUM_DECODED_TOKEN_KEY.JTI.value]
        user_id = jwt_payload[app.config.get(ENUM_JWT_ENV.IDENTITY_CLAIM.value)]

        try:
            token = service_db.find_token_by_filters(jti=jti, user_id=user_id)
            return token.is_revoked

        except DoesNotExist as e:
            DoesNotExist(
                f"Aucun token trouver avec jti = {jti}, _id = {user_id} - \n {e}"
            )

        except Exception as e:
            Exception(f"Une erreur est survenu - \n {e}")

    def is_token_expired(self, expiration_timestamp) -> bool:
        if expiration_timestamp is None:
            return False

        expiration_time = datetime.datetime.fromtimestamp(
            expiration_timestamp, tz=pytz.timezone(ENUM_TIMEZONE.TIMEZONE_PARIS.value)
        )
        curent_time = get_paris_time()

        return expiration_time < curent_time


service_jwt: Service_JWT = Service_JWT()


@ext.jwt_ext.token_in_blocklist_loader
def check_if_token_revoked(jwt_headers, jwt_payload):
    try:
        if service_jwt.is_token_revoked(jwt_payload):
            return True

        if service_jwt.is_token_expired(jwt_payload[ENUM_DECODED_TOKEN_KEY.EXP.value]):
            service_jwt.revoke_token(
                jti=jwt_payload[ENUM_DECODED_TOKEN_KEY.JTI.value],
                user_id=jwt_payload[app.config.get(ENUM_JWT_ENV.IDENTITY_CLAIM.value)],
            )
            return True

        return False
    except Exception:
        return True


@ext.jwt_ext.user_lookup_loader
def user_loader_callback(jwt_header, jwt_payload):
    return service_db.find_user_by_filters(
        user_id=jwt_payload[app.config.get(ENUM_JWT_ENV.IDENTITY_CLAIM.value)]
    )
