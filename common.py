

from enum import Enum, auto
from biochatter.rag_agent import RagAgentModeEnum

######### LITERALS ###########

APP_PORT_DEF = 50501

CONNECTION_ARGS_EXAMPLE = {
    "host": "local",
    "port": "17687",
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
BIOCHATTER_ENV = ".env.local"
PROMPTS_FN =  "prompts.yaml"
NEO4J_DEFAULT_DB = "neo4j"
ARGS_CONNECTION_ARGS = "connectionArgs"
ARGS_RESULT_NUM = "numberOfResults"
ENV_KG_HOST = "KGHOST"
ENV_KG_PORT = "KGPORT"
ENV_KG_DB = "KG_DB"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_API_BASE = "OPENAI_API_BASE"
ENV_APP_PORT = "APP_PORT"
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
    tool = "tool"

    # make it similar to Literal["system", "user", "assistant"] while retaining enum convenience

    #def __new__(cls, value, *args, **kwargs):
    #    obj = str.__new__(cls, value)
    #    obj._value_ = value
    #    return obj

    def __str__(self):
        return str(self.value)
