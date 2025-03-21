from mongoengine.errors import ValidationError, DoesNotExist
from backend.app.extension.extensions import ext
from typing import Self
from backend.app.core.const.enum import (
    ENUM_FILTERS_USER,
    ENUM_FILTERS_TOKEN,
    ENUM_FIELDS_USER,
)
from backend.app.exeptions.custom_exeptions import (
    FilterUserMissingError,
    FilterTokenMissingError,
    FieldsUserMissingError,
)
from backend.app.models.models import MODEL_USER, MODEL_Sequence, MODEL_TokenBlockList


class Service_DB:
    def __init__(self: Self) -> None:
        pass

    def get_next_sequence_value(self, sequence_name: str) -> int:
        sequence = MODEL_Sequence.objects(id=sequence_name).modify(
            upsert=True, new=True, inc__sequence_value=1
        )
        return sequence.sequence_value

    def find_user_by_filters(self, multiple: bool = False, **kwargs: dict):
        filters: dict[str:...] = {
            key: value
            for key, value in kwargs.items()
            if key in ENUM_FILTERS_USER.FILTERS.value
        }

        if not filters:
            raise FilterUserMissingError(ENUM_FILTERS_USER.FILTERS.value)

        query = MODEL_USER.objects(**filters)

        return query if multiple else query.first()

    def create_user(self, data: dict[...:...]) -> dict:
        missing_fields = [
            field for field in ENUM_FIELDS_USER.FIELDS.value if field not in data.keys()
        ]

        if missing_fields:
            raise FieldsUserMissingError(ENUM_FIELDS_USER.FIELDS.value)

        new_user = MODEL_USER(
            user_id=self.get_next_sequence_value("user_id"),
            email=data[ENUM_FIELDS_USER.FIELDS.value[0]],
            username=data[ENUM_FIELDS_USER.FIELDS.value[1]],
            password_hash=ext.pwd_context_ext.hash(
                data[ENUM_FIELDS_USER.FIELDS.value[2]]
            ),
        )

        new_user.save()
        return new_user

    def find_token_by_filters(self, multiple: bool = False, **kwargs: dict):
        filters: dict[str:...] = {
            key: value
            for key, value in kwargs.items()
            if key in ENUM_FILTERS_TOKEN.FILTERS.value
        }

        if not filters:
            raise FilterTokenMissingError(ENUM_FILTERS_TOKEN.FILTERS.value)

        query = MODEL_TokenBlockList.objects(**filters)

        return query if multiple else query.first()

    def add_to_db(self, data) -> None:
        try:
            data.save()
        except (Exception, ValidationError) as e:
            raise Exception(str(e))

    def delete_data(self, data_to_delete) -> None:
        try:
            data_to_delete.delete()
        except (DoesNotExist, ValidationError, Exception) as e:
            raise Exception(str(e))


service_db = Service_DB()
