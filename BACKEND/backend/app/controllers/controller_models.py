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
        user_input = request.form.get("userInput", "").strip()
        if not user_input:
            return create_json_response(
                status_code=400,
                status="fail",
                message="Le champ 'userInput' est requis.",
            )

        all_documents = []
        for key in request.files.keys():
            if key.startswith("documents"):
                all_documents.extend(request.files.getlist(key))
        document_names = [doc.filename for doc in all_documents]

        model = Service_MODEL(typeModel)

        message = model.handle_prediction(user_input, all_documents)

        return create_json_response(
            status="success",
            message=message,
            details={"userInput": user_input, "documents": document_names},
        )
    
    except Exception as e:
        print(e)
        return create_json_response(
            status_code=500,
            status="fail",
            message="Une erreur est survenue.",
            details=str(e),
        )