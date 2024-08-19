from typing import Any, List, Optional
from pydantic import BaseModel
from enum import Enum, auto

class ErrorCodes(Enum):
    SUCCESS = 200
    INVALID_INPUT = 400
    NOT_FOUND = 404
    PERMISSION_DENIED = 403
    SERVER_ERROR = 500
    UNKNOWN_ERROR = 666

class KGConnectionArgs(BaseModel):
    host: str
    port: str
    db_name: Optional[str]=None
    user: Optional[str]=None
    password: Optional[str]=None

RAG_KG = "KG"
localhost = "127.0.0.1"