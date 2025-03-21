from marshmallow import Schema, ValidationError, fields, validates
from backend.app.core.const.enum import ENUM_LOGIN_SCHEMA
import re


class LoginSchema(Schema):
    email_fields: str = "email"
    password_fields: str = "password"

    email = fields.Str(
        required=True,
        error_messages={
            "required": "L'email est un champs requis",
            "null": "L'email est un champs qui ne peux pas etre vide",
            "invalid": "Veuillez fournir une adresse email valide",
        },
    )
    password = fields.Str(
        required=True,
        error_messages={
            "required": "Le mot de passe est un champs requis",
            "null": "Le mot de passe ne peux pas etre un champs requis",
            "invalide": "Veuillez fournir un mot de passe valide",
        },
    )

    @validates(email_fields)
    def validator_email(self, value):
        errorMessage: list[str] = []
        if not value:
            errorMessage.append(ENUM_LOGIN_SCHEMA.EMAIL_EMPTY_ERROR_MESSAGE.value)

        if not re.match(ENUM_LOGIN_SCHEMA.EMAIL_PATERN.value, value):
            errorMessage.append(ENUM_LOGIN_SCHEMA.EMAIL_REGEX_ERROR_MESSAGE.value)

        if len(errorMessage) > 0:
            raise ValidationError(", ".join(errorMessage))

    @validates(password_fields)
    def validator_password(self, value):
        if not value:
            raise ValidationError(ENUM_LOGIN_SCHEMA.PASSWORD_ERROR.value)


class ResgistrySchema(Schema):

    email_fields: str = "email"
    username_fields: str = "username"
    password_fields: str = "password"

    email = fields.Str(
        required=True,
        error_messages={
            "required": "L'email est un champs requis",
            "null": "L'email est un champs qui ne peux pas etre vide",
            "invalid": "Veuillez fournir une adresse email valide",
        },
    )

    username = fields.Str(
        required=True,
        validate=[
            fields.Length(
                min=2,
                max=50,
                error="Le username doit contenir entre 2 et 50 caractéres",
            )
        ],
        error_messages={
            "required": "Le username est un champs requis",
            "null": "Le username est un champs qui ne peux pas etre vide",
            "invalid": "Veuillez fournir un username valide",
        },
    )

    password = fields.Str(
        required=True,
        load_only=True,
        error_messages={
            "required": "Le mot de passe est un champs requis",
            "null": "Le mot de passe ne peux pas etre un champs requis",
            "invalid": "Veuillez fournir un mot de passe valide",
        },
    )

    @validates(email_fields)
    def validate_email(self, value):
        errorMessages: list[str] = []
        if not value:
            errorMessages.append("L'email est requis et ne peut pas être vide")

        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$", value):
            errorMessages.append("Format d'email invalide")

        if len(errorMessages) > 0:
            raise ValidationError(", ".join(errorMessages))

    @validates(password_fields)
    def validate_password_strength(self, value):
        errorMessages: list[str] = []
        if len(value) < 8:
            errorMessages.append("Le mot de passe doit contenir au moins 8 caractères")
        if not any(char.islower() for char in value):
            errorMessages.append(
                "Le mot de passe doit contenir au moins une lettre minuscule"
            )
        if not any(char.isupper() for char in value):
            errorMessages.append(
                "Le mot de passe doit contenir au moins une lettre majuscule"
            )
        if not any(char.isdigit() for char in value):
            errorMessages.append("Le mot de passe doit contenir au moins un chiffre")
        if not any(char in "!@#$%^&*()_+-=[]{}|;:,.<>?/" for char in value):
            errorMessages.append(
                "Le mot de passe doit contenir au moins un caractère spécial"
            )

        if len(errorMessages) > 0:
            raise ValidationError(", ".join(errorMessages))

    @validates(username_fields)
    def validate_username(self, value):
        errorMessages: list[str] = []
        if not value.strip():
            errorMessages.append(
                "Le username ne peux pas etre composé uniquement d'espace"
            )
        if len(value) > 20:
            errorMessages.append("Le username ne peux pas avoir plus de 20 charactères")

        if len(errorMessages) > 0:
            raise ValidationError(", ".join(errorMessages))


class ResetPasswordRequestSchema(Schema):
    email_fields: str = "email"

    email = fields.Str(
        required=True,
        error_messages={
            "required": "L'email est un champs requis",
            "null": "L'email est un champs qui ne peux pas etre vide",
            "invalid": "Veuillez fournir une adresse email valide",
        },
    )

    @validates(email_fields)
    def validator_email(self, value):
        errorMessage: list[str] = []
        if not value:
            errorMessage.append(ENUM_LOGIN_SCHEMA.EMAIL_EMPTY_ERROR_MESSAGE.value)

        if not re.match(ENUM_LOGIN_SCHEMA.EMAIL_PATERN.value, value):
            errorMessage.append(ENUM_LOGIN_SCHEMA.EMAIL_REGEX_ERROR_MESSAGE.value)

        if len(errorMessage) > 0:
            raise ValidationError(", ".join(errorMessage))


class ResetPasswordSchema(Schema):
    password_fields: str = "password"

    password = fields.Str(
        required=True,
        load_only=True,
        error_messages={
            "required": "Le mot de passe est un champs requis",
            "null": "Le mot de passe ne peux pas etre un champs requis",
            "invalid": "Veuillez fournir un mot de passe valide",
        },
    )

    @validates(password_fields)
    def validate_password_strength(self, value):
        errorMessages: list[str] = []
        if len(value) < 8:
            errorMessages.append("Le mot de passe doit contenir au moins 8 caractères")
        if not any(char.islower() for char in value):
            errorMessages.append(
                "Le mot de passe doit contenir au moins une lettre minuscule"
            )
        if not any(char.isupper() for char in value):
            errorMessages.append(
                "Le mot de passe doit contenir au moins une lettre majuscule"
            )
        if not any(char.isdigit() for char in value):
            errorMessages.append("Le mot de passe doit contenir au moins un chiffre")
        if not any(char in "!@#$%^&*()_+-=[]{}|;:,.<>?/" for char in value):
            errorMessages.append(
                "Le mot de passe doit contenir au moins un caractère spécial"
            )

        if len(errorMessages) > 0:
            raise ValidationError(", ".join(errorMessages))
