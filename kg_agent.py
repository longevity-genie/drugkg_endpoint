import json
import uuid
import openai
import yaml
from common import *
from typing import List, Optional, Dict
from datetime import datetime as dt
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
from biochatter.llm_connect import GptConversation, OllamaConversation
from loguru import logger
import neo4j_utils as nu

# Safe import from YAML
def load_prompts(file_path: str = PROMPTS_FN):
    try:
        logger.debug(f"Importing prompts from: {file_path}")
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        logger.debug(f"Prompts imported: {data}")
        return data
    except Exception as e:
        logger.error(e)
        return None

def process_connection_args(rag: str, connection_args: dict) -> dict:
    if rag == RAG_KG:
        if connection_args.get("host", "").lower() == "local":
            connection_args["host"] = os.getenv(ENV_KG_HOST, localhost)
    return connection_args

def get_kg_config(kg_config: dict):
    try:
        kg_config[ARGS_CONNECTION_ARGS] = vars(kg_config[ARGS_CONNECTION_ARGS])
        kg_config[ARGS_CONNECTION_ARGS] = process_connection_args(
            RAG_KG, kg_config[ARGS_CONNECTION_ARGS]
        )

    except Exception as e:
        logger.error(e)
        return None

def get_rag_agent_prompts(prompt: str = None) -> List[str]:
    if prompt:
        return [prompt]
    else:
        return [KG_RAG_PROMPT]

def find_schema_info_node(connection_args: dict):
    try:
        """
        Look for a schema info node in the connected BioCypher graph and load the
        schema info if present.
        """
        db_uri = "bolt://" + connection_args.get("host") + \
            ":" + connection_args.get("port")
        neodriver = nu.Driver(
            db_name=connection_args.get("db_name") or os.getenv(ENV_KG_DB, NEO4J_DEFAULT_DB) ,
            db_uri=db_uri,
        )
        result = neodriver.query("MATCH (n:Schema_info) RETURN n LIMIT 1")

        if result[0]:
            schema_info_node = result[0][0]["n"]
            schema_dict = json.loads(schema_info_node["schema_info"])
            return schema_dict

        return None
    except Exception as e:
        logger.error(e)
        return None

def get_kg_connection_status(connection_args: Optional[dict]):
    if not connection_args:
        return False
    try:
        connection_args = process_connection_args(RAG_KG, connection_args)
        schema_dict = find_schema_info_node(connection_args)
        rag_agent = RagAgent(
            mode=RagAgentModeEnum.KG,
            model_name=DEFAULT_MODEL,
            connection_args=connection_args,
            schema_config_or_info_dict=schema_dict,
        )
        logger.debug("Agent connected: {}", lambda: rag_agent.agent.is_connected())
        return rag_agent.agent.is_connected()
    except Exception as e:
        logger.error(e)
        return False


class BiochatterInstance:
    def __init__(
        self,
        session_id: str = uuid.uuid4(),
        model_config: Dict = DEFAULT_MODEL_CONFIG.copy(),
        rag_agent_prompts=None
    ):
        if rag_agent_prompts is None:
            rag_agent_prompts = get_rag_agent_prompts()
        self.modelConfig = model_config
        self.model_name = model_config.get("model_name", DEFAULT_MODEL)
        self.session_id = session_id
        self.rag_agent_prompts = rag_agent_prompts

        logger.debug(f"Session ID: {session_id}")
        logger.debug(f"Session ID: {model_config}")

        self.createdAt = int(dt.now().timestamp() * 1000)  # in milliseconds
        self.refreshedAt = self.createdAt
        self.maxAge = MAX_AGE
        self.chatter = self.create_chatter()

    def chat(
        self,
        messages: List[Dict[str, str]],
        use_kg: bool = False,
        kg_config: dict = None,
    ):
        if self.chatter is None:
            return
        if not messages or len(messages) == 0:
            return
        api_key = get_api_key()
        if not api_key:
            return
        if not openai.api_key or not hasattr(self.chatter, "chat"):
                # save api_key to os.environ to facilitate conversation_factory
                # to create conversation
                if isinstance(self.chatter, GptConversation):
                    os.environ["OPENAI_API_KEY"] = api_key
                self.chatter.set_api_key(api_key, self.session_id)

        if use_kg:
            self.update_kg(kg_config)

        text = messages[-1]["content"] # extract last message for query
        history = messages[:-1] # trim last
        logger.debug(f"history:{history}")
        # Convert the list of dictionaries to a list of Message objects
        messages_list: List[Message] = [Message(**message_dict) for message_dict in messages]
        self.setup_messages(messages_list)

        try:
            msg, usage, corr = self.chatter.query(text) #primary LLM call
            kg_context_injection = self.chatter.get_last_injected_context()
            logger.debug(f"msg:{msg}")
            logger.debug(f"usage:{usage}")
            logger.debug(f"correction:{corr}")
            logger.debug(f"injection:{kg_context_injection}")
            return msg, usage, kg_context_injection
        except Exception as e:
            logger.error(e)
            raise e

    def setup_messages(self, openai_msgs: List[Message]):
        if self.chatter is None:
            return False
        self.chatter.messages = []
        for msg in openai_msgs:
            if msg.role == Role.system:
                self.chatter.append_system_message(msg.content)
            elif msg.role == Role.assistant:
                self.chatter.append_ai_message(msg.content)
            elif msg.role == Role.user:
                self.chatter.append_user_message(msg.content)

    def create_chatter(self): # TODO: support OllamaConversation
        logger.info("create chatter from biochatter.GptConversation")
        chatter = GptConversation(
            self.model_name,
            prompts={"rag_agent_prompts": self.rag_agent_prompts}
        )
        chatter.set_api_key(get_api_key(), self.session_id)
        chatter.ca_model_name = self.model_name  # Override hardcode
        return chatter

    def update_kg(self, kg_config: Optional[dict]):
        # update kg
        if not kg_config or ARGS_CONNECTION_ARGS not in kg_config:
            logger.error(f"missing {ARGS_CONNECTION_ARGS} in {kg_config}")
            return ErrorCodes.INVALID_INPUT
        try:
            schema_info = find_schema_info_node(kg_config[ARGS_CONNECTION_ARGS])
            if not schema_info:
                logger.error(f"missing schema_info in the graph!!!")
                return ErrorCodes.NOT_FOUND
            kg_agent = RagAgent(
                mode=RagAgentModeEnum.KG,
                model_name=DEFAULT_MODEL,
                connection_args=kg_config[ARGS_CONNECTION_ARGS],
                use_prompt=True, #must be set for retrival to work
                schema_config_or_info_dict=schema_info,
                conversation_factory=self.create_chatter,  # chatter factory
                n_results=kg_config.get(ARGS_RESULT_NUM, RESULT_NUM_DEFAULT)  # number of results to return
            )
            self.chatter.set_rag_agent(kg_agent) #only one instance of kg_agent per chatter
        except Exception as e:
            logger.error(e)
            return ErrorCodes.UNKNOWN_ERROR






