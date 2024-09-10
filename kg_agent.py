import os
import json
import asyncio
from uuid import uuid4
import openai
from langchain_openai import ChatOpenAI
import yaml
import time
import copy
from models import *
from typing import List, Optional, Dict, Tuple
from datetime import datetime as dt
from openai import OpenAI, Stream
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
from biochatter.llm_connect import (
    GptConversation,
   # OllamaConversation
)
from loguru import logger
import neo4j_utils as nu
from dotenv import load_dotenv

_prompts : Optional[dict] = None

load_dotenv(dotenv_path=BIOCHATTER_ENV, override=True)

def get_app_port() -> int:
    app_port = int(os.getenv(ENV_APP_PORT, APP_PORT_DEF))
    logger.trace(f"Imported app_port: {str(app_port)}")
    return app_port

def get_api_base() -> str:
    base = os.getenv(ENV_OPENAI_API_BASE, None)
    logger.trace(f"Imported api base: {str(base)}")
    return base

def get_api_key() -> str:
    key = os.getenv(ENV_OPENAI_API_KEY, None)
    logger.trace(f"Imported key starting with: {str(key)[:10]}")
    return key

def get_oai_client() -> Optional[OpenAI]:
    client = OpenAI(
        api_key=get_api_key(),
        base_url=get_api_base()
    )
    logger.debug(f"Started OAI")
    try:
        models = client.models.list()
        logger.trace(f"Models avail: {str(models)}")
        return client
    except openai.AuthenticationError as e:
        logger.error(f"OAI auth error: {str(e)}")
        return None

def load_prompts(file_path: str = PROMPTS_FN) -> Optional[dict]:
    global _prompts
    if _prompts:
        return _prompts #Load once
    try:
        logger.debug(f"Importing prompts from: {str(file_path)}")
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        logger.debug(f"Prompts imported: {str(data)}")
        _prompts = data
        return data
    except Exception as e:
        logger.error(f"Error during loading prompts: {str(e)}")
        return None

def get_system_prompt(request: ChatCompletionRequest) -> Tuple[Optional[str],Optional[List[str]]]:
    #       GROQ not supported by biochatter.llmconnect
    try:
        prompts = load_prompts()
        model_name = request.model.lower()
        if model_name and prompts:
            for key in prompts.keys():
                if model_name.startswith(key.lower()):
                    system_prompt = prompts[key].get("system_prompt", None)
                    rag_agent_prompts = [prompts[key].get("kg_rag_prompt", KG_RAG_PROMPT)]
                    return system_prompt, rag_agent_prompts
        logger.warning("No prompts available!")
    except Exception as e:
        logger.error(f"Error during selecting prompts: {str(e)}")
    return None, None


def has_system_prompt(request: ChatCompletionRequest) -> Optional[bool]:
    if request.messages and len(request.messages) > 0:
        if request.messages[0] and request.messages[0].role == Role.system:
            text = message_to_string(request.messages[0])
            if text:
                logger.info(f"Called with external prompt, switching to OpenAI")
                logger.debug(f"External prompt: {text}")
                return True #True if external prompt
        return False # False if no prompt
    else:
        return #None if empty


def message_to_string(message:Message) -> str:
    if isinstance(message.content, str):
        # If the content is already a string, return it as is
        return message.content

    # If content is a list, process each item
    result_strings = []
    for item in message.content:
        if isinstance(item, TextContent):
            # If the item is TextContent, add the text
            result_strings.append(item.text)
        elif isinstance(item, ImageContent):
            # If the item is ImageContent, add the image URL as a string
            result_strings.append(f"[Image: {item.image_url}]")

    # Join all the pieces into a single string with appropriate separators
    return " ".join(result_strings)


def string_to_message(
        text: str,
        message: Optional[Message] = None,
        role: Optional[Role] = Role.assistant,
        legacy: bool = False
) -> Message:

    if legacy:
        # If message is provided, set role to message.role
        new_role = message.role if message else role
        return Message(role=new_role, content=text)

    if message is None:
        # No message provided, return a new Message with TextContent
        return Message(
            role=role,
            content=[
                TextContent(
                    type="text",
                    text=text
                )
            ]
        )
    else:
        # Create a copy of the message to avoid mutating the original object
        message = copy.deepcopy(message)

    # If message content is a simple string, replace it with the text
    if isinstance(message.content, str):
        message.content = text
        return message

    # If message content is a list, determine if TextContent is present
    text_content_found = False
    for item in message.content:
        if isinstance(item, TextContent):
            item.text = text  # Replace the text in the existing TextContent
            text_content_found = True
            break

    if not text_content_found:
        # If TextContent is not present, append a new TextContent with the provided text
        message.content.append(TextContent(text=text))

    return message


def inject_system_prompt(
        request: ChatCompletionRequest,
        system_prompt: str,
        legacy: bool = False
) -> Optional[List[Message]]:
    try:
        logger.trace(f"Changing system prompt to: {str(system_prompt)}")
        if request.messages and len(request.messages) > 0:
            logger.debug(f"First message: {str(request.messages[0])}")
            if request.messages[0] and request.messages[0].role == Role.system:
                #First message is empty system prompt
                request.messages[0] = string_to_message(system_prompt, message=request.messages[0], legacy=legacy)
            else:
                # Injecting a SystemMessage instance into the start of conversation
                system_message = string_to_message(system_prompt, role=Role.system, legacy=legacy)
                request.messages.insert(0, system_message)
            logger.debug(f"New first message: {str(request.messages[0])}")
            return request.messages
        else:
            logger.warning(f"Empty messages!")
            return None #empty messages
    except Exception as e:
        logger.error(f"Error during injection: {str(e)}")
        return None



def process_connection_args(connection_args: DbConnectionArgs) -> DbConnectionArgs:
    logger.debug(f"Processing args: {str(connection_args)}")
    if connection_args.host is not None and connection_args.host.lower() == "local":
        connection_args.host = os.getenv(ENV_KG_HOST, localhost)
    if connection_args.port is None or connection_args.port == "":
        connection_args.port = os.getenv(ENV_KG_PORT, 7687)
    logger.debug(f"Processing result: {str(connection_args)}")
    return connection_args

def process_kg_config(kg_config: KGConfig) -> KGConfig:
    try:
        logger.debug(f"Input kg_config: {str(kg_config)}")
        kg_config.connectionArgs = process_connection_args(kg_config.connectionArgs)
        logger.debug(f"Processing result of kg_config: {str(kg_config)}")
        return kg_config
    except Exception as e:
        logger.error(f"Error during updating of kg_config: {str(e)}")
        return kg_config


def get_rag_agent_prompts(prompt: str = None) -> List[str]:
    if prompt:
        return [prompt]
    else:
        return [KG_RAG_PROMPT]

def find_schema_info_node(connection_args: dict) -> Optional[dict]:
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

def get_kg_connection_status(connection_args: DbConnectionArgs) -> bool:
    if not connection_args:
        return False
    try:
        connection_args = vars(process_connection_args(connection_args))
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

def get_completion_response(
        model : Optional[str]=DEFAULT_MODEL,
        text : Optional[str]=None,
        usage: Optional[ChatCompletionUsage] = None
) -> ChatCompletionResponse:

    if not text:
        text = "Something went wrong with response!!"
    message = ResponseMessage(
        role=str(Role.assistant),
        content=text
    )
    choice = ChatCompletionChoice(
        index=0,
        finish_reason="stop",
        message=message,
        text=text
    )
    response = ChatCompletionResponse(
        id = "chatcmpl-"+str(uuid4()),
        object = "chat.completion",
        created=time.time(),
        model = model,
        choices = [choice],
        usage=usage
    )
    return response

#  generator function to yield ChatCompletionChunk chunks
async def generate_response_chunks(response: ChatCompletionResponse, stop: Optional[str] = "[DONE]" ) -> AsyncGenerator[str, None]:
    logger.info("Imitating generation")
    logger.trace(f"Given {str(response)}")
    for choice in response.choices:
        delta=ChatCompletionChoiceChunk(
            index=choice.index,
            delta=choice.message,
            finish_reason=None
        )
        logger.trace(f"choice {str(delta)}")
        chunk = ChatCompletionChunkResponse(
            object="chat.completion.chunk",
            id=response.id,
            choices=[delta],
            created=int(time.time()),
            model=response.model,
        )
        chunk = chunk.model_dump_json()
        logger.debug(f"chunk {str(chunk)}")
        yield f"data: {chunk}\n\n"

        await asyncio.sleep(1)
    final_chunk = ChatCompletionChunkResponse(
        object="chat.completion.chunk",
        id=response.id,
        created=int(time.time()),
        model=response.model,
        choices=[ChatCompletionChoiceChunk(
            index=0,
            delta=ResponseMessage(
                role=None,
                content=None
            ),
            finish_reason="stop"
        )],
        usage=response.usage,
    )
    final_chunk = final_chunk.model_dump_json()
    logger.debug(f"final_chunk {str(final_chunk)}")
    yield f"data: {final_chunk}\n\n"
    yield f"data: {stop}\n\n"
    await asyncio.sleep(1)

class BiochatterInstance:
    def __init__(
        self,
        session_id: str = uuid4(),
        model_config: Dict = DEFAULT_MODEL_CONFIG.copy(),
        rag_agent_prompts=None
    ):
        if rag_agent_prompts is None:
            rag_agent_prompts = get_rag_agent_prompts()
        self.modelConfig = model_config
        self.model_name = model_config.get("model_name", DEFAULT_MODEL)
        self.session_id = session_id
        self.rag_agent_prompts = rag_agent_prompts

        logger.debug(f"Session ID: {str(session_id)}")
        logger.debug(f"Session ID: {str(model_config)}")

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
        logger.debug(f"Chatter..ok")
        if not messages or len(messages) == 0:
            return
        logger.debug(f"Messages..ok")
        api_key = get_api_key()
        api_base = get_api_base()
        logger.debug(f"Using api_key : {str(api_key)[:10]}...")
        if not (api_key and api_base):
            return
        if not openai.api_key or not hasattr(self.chatter, "chat"):
            logger.error(f"Chat not initialized: {str(self.chatter)}")

        if use_kg:
            logger.debug(f"Using KG, config: {str(kg_config)}")
            self.update_kg(kg_config)

        text = messages[-1]["content"] # extract last message for query
        history = messages[:-1] # trim last
        logger.debug(f"history:{str(history)}")
        # Convert the list of dictionaries to a list of Message objects
        messages_list: List[Message] = [Message(**message_dict) for message_dict in messages]
        self.setup_messages(messages_list)

        try:
            msg, usage, corr = self.chatter.query(text)                                              #primary LLM call
            kg_context_injection = self.chatter.get_last_injected_context()
            logger.debug(f"msg:{str(msg)}")
            logger.debug(f"usage:{str(usage)}")
            logger.debug(f"correction:{str(corr)}")
            logger.debug(f"injection:{str(kg_context_injection)}")
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
        chatter.ca_model_name = self.model_name  # Override hardcode
        api_key = get_api_key()
        base_url = get_api_base()
        chatter.user = self.session_id
        chatter.chat = ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                openai_api_key=api_key,
                base_url=base_url,
            )

        chatter.ca_chat = ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                openai_api_key=api_key,
                base_url=base_url,
            )
        return chatter

    def update_kg(self, kg_config: Optional[dict]):
        # update kg
        logger.debug(f"Updating KG_RAG agent")
        if not kg_config or ARGS_CONNECTION_ARGS not in kg_config:
            logger.error(f"missing {ARGS_CONNECTION_ARGS} in {str(kg_config)}")
            return ErrorCodes.INVALID_INPUT
        try:
            conn_args = vars(kg_config[ARGS_CONNECTION_ARGS])
            logger.debug(f"Connecting using {str(conn_args)}")
            schema_info = find_schema_info_node(conn_args)
            if not schema_info:
                logger.error("missing schema_info in the graph!!!")
                return ErrorCodes.NOT_FOUND
            else:
                logger.info(f"Successfully got schema {str(schema_info)}")
            n_results = kg_config.get(ARGS_RESULT_NUM, RESULT_NUM_DEFAULT)
            logger.debug(f"Expecting {str(n_results)} results")
            kg_agent = RagAgent(
                mode=RagAgentModeEnum.KG,
                model_name=DEFAULT_MODEL,
                connection_args=conn_args,
                use_prompt=True, #must be set for retrival to work
                schema_config_or_info_dict=schema_info,
                conversation_factory=self.create_chatter,  # chatter factory
                n_results=n_results,  # number of results to return
                use_reflexion=True,
            )
            self.chatter.set_rag_agent(kg_agent) #only one instance of kg_agent per chatter
        except Exception as e:
            logger.error(e)
            return ErrorCodes.UNKNOWN_ERROR






