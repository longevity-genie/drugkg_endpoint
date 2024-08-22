

from models import *
from pathlib import Path
from fastapi import FastAPI, HTTPException
from openai import OpenAI

from fastapi.middleware.cors import CORSMiddleware
from kg_agent import (
    BiochatterInstance,
    get_kg_connection_status,
    process_kg_config,
    get_api_key,
    load_prompts,
    get_app_port,
    get_completion_response,
    content_to_string
)
from loguru import logger

log_path = Path(__file__)
log_path = Path(log_path.parent, "logs", "biochatter_endpoint.log")
logger.add(log_path.absolute(), rotation="10 MB")

prompts = load_prompts()

app = FastAPI(title="Biochatter Knowledge Graph API endpoint.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def openai_completion(messages: List[Message], model : str = DEFAULT_MODEL) -> ChatCompletionResponse:
    # Set your API key here
    client = OpenAI()
    client.api_key = get_api_key()
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    logger.debug(f"OAI response: {str(response)}")
    return response


@app.get("/", description="Default message", response_model=str)
async def default():
    logger.debug("GET /")
    return "BioChatter KG API Endpoint v1.0"

@app.post("/biochatter_api/kg_status", description="Tests knowledge graph connection for a specified database")
async def kg_connection_status(
    request: KGConnectionArgs,
):
    try:
        logger.debug("POST biochatter_api/kg_status")
        logger.debug(f"Input: {str(request)}")
        connected = get_kg_connection_status(request.connectionArgs)
        return {
            "status": "connected" if connected else "disconnected",
            "code": ErrorCodes.SUCCESS,
        }
    except Exception as e:
        logger.error(e)
        return {"error": str(e), "code": ErrorCodes.SERVER_ERROR}


@app.post("/biochatter_api/chat/completions_ext", description="chat completions")
async def kg_chat_completions(
    request: ChatCompletionsExtendedModel
) -> ChatCompletionResponseExt:
    logger.debug("POST /biochatter_api/chat/completions")
    logger.debug(f"Input: {str(request)}")
    usage = ChatCompletionUsage()
    kg_context_injection = ""
    resp_content = ""
    response = get_completion_response().model_dump()
    try:
        # session_id = request.session_id
        messages = [vars(msg) for msg in request.messages]
        model = request.model
        temperature = request.temperature
        presence_penalty = request.presence_penalty
        frequency_penalty = request.frequency_penalty
        top_p = request.top_p
        kg_config = vars(process_kg_config(request.kgConfig))
        use_kg = request.useKG

        system_prompt = None
        rag_agent_prompts = None

        current_llm: dict = {
            "model": model,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "top_p": top_p,
            "api_key": get_api_key(),
        }

        if request.model.startswith("gpt-"):
            system_prompt = prompts["gpt"].get("system_prompt", None)
            rag_agent_prompts =  [prompts["gpt"].get("kg_rag_prompt", KG_RAG_PROMPT)]

        #       GROQ not supported by biochatter.llmconnect

        if system_prompt:
            biochatter = BiochatterInstance( #instantiate
            #    session_id=session_id,
                model_config=current_llm,
                rag_agent_prompts= rag_agent_prompts
            )

            if (len(request.messages) > 0) and (request.messages[0].role == Role.system):
                request.messages[0].content = system_prompt
            else:
                # Creating a Message instance
                system_message = Message(role=Role.system, content=system_prompt)
                request.messages.insert(0, system_message)
            logger.debug("Starting biocahtter.chat")
            resp_content, usage, kg_context_injection = biochatter.chat(
                messages=messages,
                use_kg=use_kg,
                kg_config=kg_config,
            )
            logger.debug(f"response: {str(resp_content)}")
            logger.debug(f"usage: {str(usage)}")
            logger.debug(f"kg_context: {str(kg_context_injection)}")
            err_code = ErrorCodes.SUCCESS
        else:
            err_code = ErrorCodes.MODEL_NOT_SUPPORTED

        response = get_completion_response(
            model=model,
            text=resp_content,
            usage=ChatCompletionUsage(**usage)
        ).model_dump()

    except Exception as e:
        logger.error(str(e))
        err_code = ErrorCodes.SERVER_ERROR
        resp_content = str(e)

    return ChatCompletionResponseExt(
        **response,
        contexts=kg_context_injection,
        err_code=err_code
    )


@app.post("/biochatter_api/chat/completions", description="chat completions")
async def chat_completions(
    request: ChatCompletionRequest
) -> ChatCompletionResponse:
    try:
        logger.trace("Entering completions")
        logger.debug(request)
        if request.messages and len(request.messages) > 0:
            if request.messages[0] and request.messages[0].role == Role.system:
                text = content_to_string(request.messages[0])
                if text:
                    logger.info(f"Called with external prompt, switching to OpenAI")
                    logger.debug(f"External prompt: {text}")
                    result = await openai_completion(request.messages)
                    return result
        else:
            return get_completion_response(text="No messages provided!")
        model_dump = request.model_dump()
        logger.trace(f"Config {str(model_dump)}")
        config=KGConfig(
                numberOfResults=RESULT_NUM_DEFAULT,
                connectionArgs=DbConnectionArgs(
                    host="local", # empty, populated from environment in process_connection_args
                ),
                description="Drug interactions Knowledge Graph",
            )
        logger.trace(f"KGConfig: {str(config)}")
        updated_request = ChatCompletionsExtendedModel(
            **model_dump,
            useKG=True,
            kgConfig=config,
        )
        logger.debug(f"Updated_request: {str(updated_request)}")
        result = await kg_chat_completions(updated_request)
        logger.debug(f"Extended context: {str(result)}")
        return result
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error]")


@app.post("/biochatter_api/chat/debug", description="chat completions")
async def chat_completions_debug(request: dict)-> ChatCompletionResponse:
    resp_content = "Test!!!"
    try:
        logger.debug(request)
    except Exception as e:
        logger.error(str(e))
        resp_content = str(e)
    return get_completion_response(text=resp_content)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_app_port())
