from typing import Any, List, Union, Optional, Literal

from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from pydantic import BaseModel, Field, HttpUrl
from common import *

######### Pydantic models ###########

class DbConnectionArgs(BaseModel):
    host: str = Field("localhost", example="local")
    port: Optional[str] = Field("57687", example="57687")
    db_name: Optional[str] = Field(None, example="neo4j")
    user: Optional[str]=Field(None, example="neo4j")
    password: Optional[str]=Field(None, example="neo4j")

class KGConnectionArgs(BaseModel):
    connectionArgs: DbConnectionArgs = Field(..., example=CONNECTION_ARGS_EXAMPLE)

class KGConfig(BaseModel):
    numberOfResults: Optional[int] = Field(default=10, example=10)
    connectionArgs: DbConnectionArgs = Field(..., example=CONNECTION_ARGS_EXAMPLE)
    description: Optional[str] = Field(default=None, example="Drug interactions Knowledge Graph")

# Content types
class TextContent(BaseModel):
    type: str = Field("text", example="text")
    text: str = Field(..., example="What are in these images? Is there any difference between them?")

class ImageContent(BaseModel):
    type: str = Field("image_url", example="image_url")
    image_url: HttpUrl = Field(..., example="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

# Message class - Simple string content or a list of text or image content for vision model
class Message(BaseModel):
    role: Role = Field(..., example=Role.assistant.value)
    content: Union[
        str,  # Simple string content
        List[Union[TextContent, ImageContent]]
    ] = Field(
        ...,
        description="Content can be a simple string, or a list of content items including text or image URLs."
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., example="gpt-4o-mini")
    messages: List[Message] = Field(..., example=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What drug interactions of rapamycin are you aware of? What are these interactions ?"}
    ])
    temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0, example=0.7)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, example=0.9)
    n: Optional[int] = Field(1, ge=1)
    stream: Optional[bool] = Field(default=False, example=True)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, example=0.6)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, example=0.5)
    logit_bias: Optional[dict] = Field(None, example=None)
    user: Optional[str] = Field(None, example=None)

class ChatCompletionsExtendedModel(ChatCompletionRequest):
    useKG: bool = Field(default=False, example=True)
    kgConfig: KGConfig = Field(..., example={
        "numberOfResults": 10,
        "connectionArgs": CONNECTION_ARGS_EXAMPLE,
        "description": "Drug interactions Knowledge Graph"
    })


class ChatCompletionChoice(Choice):
    text: Optional[str] = Field(default=None, alias="message.content")
    message : ChatCompletionMessage


class ChatCompletionUsage(CompletionUsage):
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    pass

#TODO: format responses
class Context(BaseModel):
    mode : str
    context : Any

class ChatCompletionResponse(ChatCompletion):
#    id: str
#    object: Literal["chat.completion"]
    created: Union[int,float]
#    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = Field(default=None)


class ChatCompletionResponseExt(ChatCompletionResponse):
    contexts: Optional[List[Context]]
    err_code: Optional[ErrorCodes]