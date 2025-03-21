from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity
from backend.app.core import ENUM_BLUEPRINT_ID
from backend.app.core.const.enum import ENUM_ENDPOINT_AUTH, ENUM_METHODS
from backend.app.schemas.auth_schemas import (
    LoginSchema,
    ResgistrySchema,
    ResetPasswordRequestSchema,
    ResetPasswordSchema,
)
from marshmallow import ValidationError
from backend.app.services.auth_service import service_auth
from backend.app.services.jwt_service import service_jwt
from backend.app.exeptions import (
    UserNotFound,
    PayloadError,
    UserPasswordNotFound,
    EmailAlreadyUsed,
)
from backend.app.core.utility import create_json_response

bp_auth = Blueprint(ENUM_BLUEPRINT_ID.AUTH.value, __name__)


@bp_auth.route(ENUM_ENDPOINT_AUTH.LOGIN.value, methods=[ENUM_METHODS.POST.value])
def login_endpoint():
    try:
        data = LoginSchema().load(request.get_json())
        access_token, refresh_token = service_auth.login(data)
        return create_json_response(
            status="success",
            message="Connexion accepter",
            access_token=access_token,
            refresh_token=refresh_token,
        )
    except ValidationError as e:
        return create_json_response(
            status_code=400,
            status="fail",
            message="Erreur dans la validation des données fournis",
            details=f"{str(e)}",
        )
    except UserNotFound as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="Utilisateur pas trouver",
            details=f"{str(e)}",
        )
    except UserPasswordNotFound as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="Le mot de passe ne correspond pas",
            details=f"{str(e)}",
        )
    except PayloadError as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="Erreur dans la payload",
            details=f"{str(e)}",
        )
    except Exception as e:
        return create_json_response(
            status_code=400,
            status="fail",
            message="Une erreur est survenue",
            details=f"{str(e)}",
        )


@bp_auth.route(ENUM_ENDPOINT_AUTH.REGISTRY.value, methods=[ENUM_METHODS.POST.value])
def registry_endpoint():
    try:
        data = ResgistrySchema().load(request.get_json())
        access_token, refresh_token = service_auth.registry(data)
        return create_json_response(
            status_code=201,
            status="sucess",
            message="Enregistrement réaliser",
            access_token=access_token,
            refresh_token=refresh_token,
        )
    except ValidationError as e:
        return create_json_response(
            status_code=400,
            status="fail",
            message="Erreur dans la validation des données fournis",
            details=f"{str(e)}",
        )
    except EmailAlreadyUsed as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="L'email est déjà utiliser",
            details=f"{str(e)}",
        )
    except Exception as e:
        return create_json_response(
            status_code=400,
            status="fail",
            message="Une erreur est survenue",
            details=f"{str(e)}",
        )


@bp_auth.route(ENUM_ENDPOINT_AUTH.LOGOUT.value, methods=[ENUM_METHODS.POST.value])
@jwt_required()
def logout_endpoint():
    try:
        service_jwt.revoke_token(get_jwt()["jti"], get_jwt_identity())
        return create_json_response(
            status_code=201, status="success", message="Deconnexion reussite"
        )

    except Exception as e:
        return create_json_response(
            status_code=400,
            status="fail",
            message="Une erreur est survenue",
            details=f"{str(e)}",
        )


@bp_auth.route(
    ENUM_ENDPOINT_AUTH.REQUEST_RESET_PASSWORD.value, methods=[ENUM_METHODS.POST.value]
)
def request_reset_password():
    try:
        data = ResetPasswordRequestSchema().load(request.get_json())
        email = data["email"]
        service_auth.send_reset_password_email(email)
        return create_json_response(
            status_code=200,
            status="success",
            message="Un email de réinitialisation a été envoyé.",
        )
    except ValidationError as e:
        return create_json_response(
            status_code=400,
            status="fail",
            message="Erreur dans la validation des données fournies",
            details=str(e),
        )
    except UserNotFound as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="Utilisateur non trouvé",
            details=str(e),
        )
    except Exception as e:
        return create_json_response(
            status_code=500,
            status="fail",
            message="Une erreur est survenue",
            details=str(e),
        )


@bp_auth.route(
    ENUM_ENDPOINT_AUTH.RESET_PASSWORD.value, methods=[ENUM_METHODS.POST.value]
)
@jwt_required()
def reset_password():
    try:
        data = ResetPasswordSchema().load(request.get_json())
        new_password = data["password"]
        user_id = get_jwt_identity()
        service_auth.reset_password(user_id, new_password)
        return create_json_response(
            status_code=200,
            status="success",
            message="Le mot de passe a été réinitialisé avec succès.",
        )
    except ValidationError as e:
        return create_json_response(
            status_code=400,
            status="fail",
            message="Erreur dans la validation des données fournies",
            details=str(e),
        )
    except Exception as e:
        return create_json_response(
            status_code=500,
            status="fail",
            message="Une erreur est survenue",
            details=str(e),
        )


@bp_auth.route(
    ENUM_ENDPOINT_AUTH.CHECK_RESET_PASSWORD_TOKEN.value,
    methods=[ENUM_METHODS.GET.value],
)
@jwt_required()
def validate_reset_token():
    try:
        if service_auth.is_reset_password_token(get_jwt()):
            return create_json_response(
                status_code=200,
                status="success",
                message="Ceci est un token de reset password",
            )
        return create_json_response(
            status_code=403,
            status="fail",
            message="Token invalide pour réinitialisation du mot de passe.",
        )
    except Exception as e:
        return create_json_response(
            status_code=500,
            status="fail",
            message="Une erreur est survenue",
            details=str(e),
        )


@bp_auth.route(
    ENUM_ENDPOINT_AUTH.REFRESH_TOKEN.value, methods=[ENUM_METHODS.POST.value]
)
@jwt_required(refresh=True)
def refresh_endpoint():
    try:
        access_token = service_auth.getNewAccessToken(get_jwt_identity())
        return create_json_response(
            status_code=201,
            status="success",
            message="Le token a bien été refresh",
            access_token=access_token,
        )
    except Exception as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="Une erreur est survenue",
            details=f"{str(e)}",
        )


@bp_auth.route(
    ENUM_ENDPOINT_AUTH.REVOKE_ACCESS_TOKEN.value, methods=[ENUM_METHODS.DELETE.value]
)
@jwt_required()
def revoke_access_endpoint():
    try:
        service_jwt.revoke_token(get_jwt()["jti"], get_jwt_identity())
        return create_json_response(
            status_code=201, status="success", message="L'access token a été revoker"
        )
    except Exception as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="Une erreur est survenue",
            details=f"{str(e)}",
        )


@bp_auth.route(
    ENUM_ENDPOINT_AUTH.REVOKE_REFRESH_TOKEN.value, methods=[ENUM_METHODS.DELETE.value]
)
@jwt_required(refresh=True)
def revoke_refresh_token_endpoint():
    try:
        service_jwt.revoke_token(get_jwt()["jti"], get_jwt_identity())
        return create_json_response(
            status_code=201, status="success", message="Le refresh token a été revoker"
        )
    except Exception as e:
        return create_json_response(
            status_code=404,
            status="fail",
            message="Une erreur est survenue",
            details=f"{str(e)}",
        )
