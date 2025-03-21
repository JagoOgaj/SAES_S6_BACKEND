from flask import Blueprint, request
from backend.app.core.utility import create_json_response
from backend.app.exeptions import ModelTypeNotFoundError
from backend.app.core.const.enum import (
    ENUM_ENDPOINT_MODEL,
    ENUM_METHODS,
    ENUM_BLUEPRINT_ID,
)
from backend.app.services.models_service import Service_MODEL
from marshmallow import ValidationError


bp_model = Blueprint(ENUM_BLUEPRINT_ID.MODEL.value, __name__)


@bp_model.route(ENUM_ENDPOINT_MODEL.PREDICT.value, methods=[ENUM_METHODS.POST.value])
def predict(typeModel: str):
    try:
        if "image" not in request.files:
            return create_json_response(
                status_code=400,
                status="fail",
                message="Aucun fichier image ou base64 n'a été fourni.",
            )

        image_file = request.files["image"]
        model = Service_MODEL(typeModel)
        message = model.handle_prediction(image_file)
        return create_json_response(status="success", message=message)
    except ValidationError as e:
        create_json_response(
            status_code=400,
            status="fail",
            message="Erreur dans la validation des données fournis",
            details=f"{str(e)}",
        )
    except TypeError as e:
        create_json_response(
            status_code=404,
            status="fail",
            message="Erreur de type de donnée",
            details=f"{str(e)}",
        )
    except ModelTypeNotFoundError as e:
        create_json_response(
            status_code=404,
            status="fail",
            message="Aucun model trouvée",
            details=f"{str(e)}",
        )
    except Exception as e:
        create_json_response(
            status_code=404,
            status="fail",
            message="Une erreur est survenue",
            details=f"{str(e)}",
        )
