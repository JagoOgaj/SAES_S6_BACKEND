from marshmallow import Schema, fields, validates, ValidationError, validate


class ConversationOverviewRequestSchema(Schema):

    page_fields = "page"
    per_page_fields = "per_page"

    page = fields.Int(
        missing=1, error_messages={"invalid": "Veuillez fournir la donnée en type int"}
    )
    per_page = fields.Int(
        missing=10, error_message={"invalid": "Veuillez fournir la donnée en type int"}
    )

    @validates(page_fields)
    def validate_page(self, value):
        errorMessages: list[str] = []
        if not isinstance(value, int):
            errorMessages.append(
                f"Le type {type(value)} n'est pas adapté pour cette request il faut en entier"
            )
        if value < 0:
            errorMessages.append("Le nombre de page doit etre supérieur a 0")

        if len(errorMessages) > 0:
            raise ValidationError(", ".join(errorMessages))

    @validates(per_page_fields)
    def validate_per_page(self, value):
        errorMessages: list[str] = []
        if not isinstance(value, int):
            errorMessages.append(
                f"Le type {type(value)} n'est pas adapté pour cette request il faut en entier"
            )
        if 0 < value > 100:
            errorMessages.append(
                "Le nombre d'item par page doit etre supérieur a 0 et inférieur a 100"
            )

        if len(errorMessages) > 0:
            raise ValidationError(", ".join(errorMessages))


class MessagesSchema(Schema):
    type = fields.String(
        required=True,
        validate=validate.OneOf(["user", "ia"]),
        description="Type de message",
    )
    content = fields.String(allow_none=True, description="Contenu du message")
    image = fields.String(allow_none=True, description="Image encodée en base64")
    created_at = fields.DateTime(
        required=True, description="Date de création du message"
    )


class ConversationSchema(Schema):
    name = fields.String(required=True, description="Nom de la conversation")
    created_at = fields.DateTime(
        required=True, description="Date de création de la conversation"
    )
    updated_at = fields.DateTime(
        required=True, description="Date de la dernière mise à jour de la conversation"
    )
    messages = fields.List(
        fields.Nested(MessagesSchema),
        required=True,
        description="Liste des messages de la conversation",
    )


class UpdatedConversation(Schema):
    messages = fields.List(
        fields.Nested(MessagesSchema),
        required=True,
        description="Liste des messages de la conversation",
    )
