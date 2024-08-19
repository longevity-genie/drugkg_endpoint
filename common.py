
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum, auto
from biochatter.rag_agent import RagAgentModeEnum

######### LITERALS ###########

CONNECTION_ARGS_EXAMPLE = {
    "host": "localhost",
    "port": "7687",
    "db_name": "neo4j",
    "user": "neo4j",
    "password": "neo4j"
}

DEFAULT_MODEL_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2000,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "sendMemory": True,
    "historyMessageCount": 4,
    "compressMessageLengthThreshold": 2000,
}

RAG_KG = RagAgentModeEnum.KG
BIOCHATTER_ENV = ".bioserver.env"
PROMPTS_FN =  "prompts.yaml"
NEO4J_DEFAULT_DB = "neo4j"
ARGS_CONNECTION_ARGS = "connectionArgs"
ARGS_RESULT_NUM = "numberOfResults"
ENV_KG_HOST = "KGHOST"
ENV_KG_PORT = "KGPORT"
ENV_KG_DB = "KG_DB"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
localhost = "127.0.0.1"
DEFAULT_MODEL = "gpt-4o-mini"
RESULT_NUM_DEFAULT=10
MAX_AGE = 3 * 24 * 3600 * 1000  # 3 days

######### PROMPTS ###########

KG_RAG_PROMPT = "User provided additional context: {statements}"

######### ENUMS ###########

class ErrorCodes(Enum):
    SUCCESS = 200
    INVALID_INPUT = 400
    NOT_FOUND = 404
    PERMISSION_DENIED = 403
    SERVER_ERROR = 500
    UNKNOWN_ERROR = 666
    MODEL_NOT_SUPPORTED = 5004


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"

######### MODELS ###########

class DbConnectionArgs(BaseModel):
    host: str = Field("localhost", example="local")
    port: str = Field("7687", example="17687")
    db_name: Optional[str] = Field(None, example="neo4j")
    user: Optional[str]=Field(None, example="neo4j")
    password: Optional[str]=Field(None, example="neo4j")

# KGConnectionArgs Model
class KGConnectionArgs(BaseModel):
    connectionArgs: DbConnectionArgs = Field(..., example=CONNECTION_ARGS_EXAMPLE)

# KGConfig Model
class KGConfig(BaseModel):
    numberOfResults: int = Field(default=10, example=10)
    connectionArgs: DbConnectionArgs = Field(..., example=CONNECTION_ARGS_EXAMPLE)
    description: Optional[str] = Field(default=None, example="Drug interactions Knowledge Graph")

# Message Model
class Message(BaseModel):
    role: Role = Field(..., example=Role.assistant)
    content: str = Field(..., example="Hello, how can I assist you today?")

# ChatCompletionsPostModel Model
class ChatCompletionsPostModel(BaseModel):
    messages: List[Message] = Field(..., example=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What drug interactions of rapamycin are you aware of? What are these interactions ?"}
    ])
    model: str = Field(..., example="gpt-3.5-turbo")
    temperature: float = Field(default=0.0, example=0.7)
    presence_penalty: float = Field(default=0.0, example=0.6)
    frequency_penalty: float = Field(default=0.0, example=0.5)
    top_p: float = Field(default=1.0, example=0.9)
    useKG: bool = Field(default=False, example=True)
    kgConfig: KGConfig = Field(..., example={
        "numberOfResults": 10,
        "connectionArgs": CONNECTION_ARGS_EXAMPLE,
        "description": "Drug interactions Knowledge Graph"
    })
    stream: bool = Field(default=False, example=True)

