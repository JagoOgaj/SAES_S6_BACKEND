from mongoengine import (
    Document,
    EmbeddedDocument,
    StringField,
    DateTimeField,
    IntField,
    EmbeddedDocumentListField,
    BooleanField,
    ListField,
)
from backend.app.core import ENUM_COLECTION_NAME
from backend.app.core import get_paris_time


class MODEL_USER(Document):
    user_id = IntField(required=True, unique=True)
    email = StringField(required=True, unique=True, max_length=255)
    username = StringField(required=True, unique=True, max_length=255)
    password_hash = StringField(required=True, max_length=255)
    created_at = DateTimeField(default=get_paris_time())

    meta = {"collection": ENUM_COLECTION_NAME.USERS.value}


class MODEL_MESSAGE(EmbeddedDocument):
    type = StringField(required=True, choices=["user", "ia"])
    content = StringField()
    documents = ListField(StringField(), default=None)  # Champ optionnel
    created_at = DateTimeField(default=get_paris_time())


class MODEL_CONVERSATION(Document):
    conversation_id = IntField(required=True, unique=True)
    user_id = IntField(required=True)
    name = StringField(required=True, max_length=255)
    created_at = DateTimeField(default=get_paris_time())
    updated_at = DateTimeField(default=get_paris_time())
    messages = EmbeddedDocumentListField(MODEL_MESSAGE)

    meta = {"collection": ENUM_COLECTION_NAME.CONVERSATIONS.value}


class MODEL_TokenBlockList(Document):
    token_id = IntField(required=True, unique=True)
    jti = StringField(required=True, unique=True)
    token_type = StringField(required=True, max_length=50)
    user_id = IntField(required=True)
    is_revoked = BooleanField(required=True, default=False)
    expires = DateTimeField(required=True)

    meta = {"collection": ENUM_COLECTION_NAME.TOKEN.value}


class MODEL_Sequence(Document):
    id = StringField(primary_key=True)
    sequence_value = IntField(default=0)

    meta = {"collection": ENUM_COLECTION_NAME.SEQUENCES.value}
