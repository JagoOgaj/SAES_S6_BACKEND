from typing import Self
from backend.app.models.models import MODEL_CONVERSATION, MODEL_MESSAGE
from collections import defaultdict
from backend.app.core.utility import get_paris_time
from backend.app.services.db_service import service_db
from backend.app.exeptions.custom_exeptions import (
    ConversationNotFoundError,
    NotAllowedToAccessThisConversationError,
)
from backend.app.schemas.user_schemas import MessagesSchema
from marshmallow import ValidationError
from pytz import UTC, timezone
from backend.app.core.const import ENUM_TIMEZONE


class Service_USER:
    def __init__(self, userId) -> None:
        self._curentUserId = userId

    def getUserConversationsByPeriod(self: Self) -> dict:
        conversations = MODEL_CONVERSATION.objects(user_id=self._curentUserId).order_by(
            "-updated_at"
        )

        result = []
        now = get_paris_time()
        paris_tz = timezone(ENUM_TIMEZONE.TIMEZONE_PARIS.value)

        period_order = {}
        order_index = 0

        for conversation in conversations:
            updated_at_utc = conversation.updated_at.replace(tzinfo=UTC)
            created_at_paris = updated_at_utc.astimezone(paris_tz)
            delta = now - created_at_paris

            if delta.days == 0:
                period = "Aujourd'hui"
            elif delta.days == 1:
                period = "Hier"
            elif delta.days == 2:
                period = "Avant-hier"
            elif delta.days < 7:
                period = f"Il y a {delta.days} jours"
            elif delta.days < 30:
                weeks = delta.days // 7
                period = f"Il y a {weeks} semaine{'s' if weeks > 1 else ''}"
            elif delta.days < 365:
                months = delta.days // 30
                period = f"Il y a {months} mois"
            else:
                years = delta.days // 365
                period = f"Il y a {years} an{'s' if years > 1 else ''}"

            if period not in period_order:
                period_order[period] = order_index
                result.append({"period": period, "conversations": []})
                order_index += 1

            result[period_order[period]]["conversations"].append(
                {"id": conversation.conversation_id, "name": conversation.name}
            )

        return {i: result[i] for i in range(len(result))}

    def deleteConversation(self: Self, idConversation: int) -> None:
        if not isinstance(idConversation, int):
            raise TypeError("L'ID de la conversation doit être un entier")

        conversation = MODEL_CONVERSATION.objects(
            conversation_id=idConversation, user_id=self._curentUserId
        ).first()

        if not conversation:
            raise ConversationNotFoundError(idConversation)

        service_db.delete_data(conversation)

    def createConversation(self: Self, data: dict) -> int:
        conversation = MODEL_CONVERSATION(
            conversation_id=service_db.get_next_sequence_value("conversation_id"),
            user_id=self._curentUserId,
            name=data["name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

        for msg in data.get("messages", []):
            conversation.messages.append(
                MODEL_MESSAGE(
                    type=msg["type"],
                    content=msg.get("content"),
                    documents=msg.get("documents") if "documents" in msg else None,  # Vérification
                    created_at=msg["created_at"],
                )
            )

        service_db.add_to_db(conversation)
        return conversation.conversation_id

    def updateConversation(self: Self, messages: list, idConversation: int) -> None:
        try:
            conversation = MODEL_CONVERSATION.objects(
                conversation_id=idConversation
            ).first()
            if not conversation:
                raise ConversationNotFoundError(idConversation)

            if conversation.user_id != self._curentUserId:
                raise NotAllowedToAccessThisConversationError(idConversation)

            embedded_messages = [
                MODEL_MESSAGE(
                    type=msg["type"],
                    content=msg.get("content"),
                    documents=msg.get("documents") if "documents" in msg else None,  # Vérification
                    created_at=msg["created_at"],
                ).to_mongo()
                for msg in messages
            ]

            MODEL_CONVERSATION.objects(conversation_id=idConversation).update(
                push__messages={"$each": embedded_messages},
                set__updated_at=get_paris_time(),
            )

        except Exception as e:
            raise Exception(
                f"Une erreur inconnue est survenue lors de la mise à jour de la conversation : {str(e)}"
            )

    def get_conversation(self: Self, idConversation: int) -> dict[str:...]:
        try:
            if not isinstance(idConversation, int):
                raise TypeError("L'id de la conversation doit etre un entier")

            conversation = MODEL_CONVERSATION.objects(
                conversation_id=idConversation
            ).first()

            if not conversation:
                raise ConversationNotFoundError(idConversation)

            if conversation.user_id != self._curentUserId:
                raise NotAllowedToAccessThisConversationError(idConversation)

            messages = sorted(conversation.messages, key=lambda msg: msg.created_at)

            message_data = MessagesSchema(many=True).dump(messages)

            return {
                "id": conversation.conversation_id,
                "name": conversation.name,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "messages": message_data,
            }

        except ConversationNotFoundError as e:
            raise ConversationNotFoundError(
                f"Conversation avec l'ID {idConversation} non trouvée."
            ) from e

        except NotAllowedToAccessThisConversationError as e:
            raise NotAllowedToAccessThisConversationError(
                "Vous n'êtes pas autorisé à accéder à cette conversation."
            ) from e

        except TypeError as e:
            raise TypeError("L'id de la conversation doit être un entier") from e

        except ValidationError as e:
            raise ValidationError(str(e)) from e

        except Exception as e:
            raise Exception(f"Une erreur inconnue est survenue : {str(e)}") from e
