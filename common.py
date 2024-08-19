from typing import Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum, auto

class ErrorCodes(Enum):
    SUCCESS = 200
    INVALID_INPUT = 400
    NOT_FOUND = 404
    PERMISSION_DENIED = 403
    SERVER_ERROR = 500
    UNKNOWN_ERROR = 666

class DbConnectionArgs(BaseModel):
    host: str = Field("localhost", example="local")
    port: str = Field("7687", example="17687")
    db_name: Optional[str] = Field(None, example="neo4j")
    user: Optional[str]=Field(None, example="neo4j")
    password: Optional[str]=Field(None, example="neo4j")

class KGConnectionArgs(BaseModel):
    connectionArgs: DbConnectionArgs

class KGConfig(BaseModel):
    resultNum: int
    connectionArgs: DbConnectionArgs
    description: Optional[str] = Field(None, example="Drug interactions Knowledge Graph")

RAG_KG = "KG"
localhost = "127.0.0.1"