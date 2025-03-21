from marshmallow import Schema, fields, validates, ValidationError
from werkzeug.utils import secure_filename


class PredictRequestSchema(Schema):
    image = fields.Field(
        required=True,
        error_messages={
            "required": "L'image est un champ requis",
        },
    )

    @validates("image")
    def validate_image(self, value):
        """
        Validation pour s'assurer que le fichier est bien une image.
        Cette validation peut être améliorée en fonction de vos besoins spécifiques.
        """
        if not value or not hasattr(value, "filename"):
            raise ValidationError("Aucun fichier n'a été fourni")

        filename = secure_filename(value.filename)
        extension = filename.rsplit(".", 1)[-1].lower()

        if extension not in ["jpg", "png"]:
            raise ValidationError("Le fichier doit être une image de type JPG, PNG")
