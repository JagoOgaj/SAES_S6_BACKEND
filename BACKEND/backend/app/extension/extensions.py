from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from passlib.context import CryptContext
from typing import Self


class Extensions:
    def __init__(self) -> None:
        self._ma_ext: Marshmallow = Marshmallow()
        self._jwt_ext: JWTManager = JWTManager()
        self._pwd_context_ext: CryptContext = CryptContext(
            schemes=["pbkdf2_sha256"], deprecated="auto"
        )

    @property
    def ma_ext(self: Self) -> Marshmallow:
        return self._ma_ext

    @property
    def jwt_ext(self: Self) -> JWTManager:
        return self._jwt_ext

    @property
    def pwd_context_ext(self: Self) -> CryptContext:
        return self._pwd_context_ext


ext = Extensions()
