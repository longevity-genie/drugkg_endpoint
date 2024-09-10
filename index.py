#!/usr/bin/env python3
import asyncio
from models import *
from pathlib import Path
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from just_agents.llm_session import LLMSession
from fastapi.middleware.cors import CORSMiddleware
from kg_agent import (
    BiochatterInstance,
    get_kg_connection_status,
    process_kg_config,
    get_api_key,
    get_app_port,
    get_completion_response,
    get_api_base,
    message_to_string,
    has_system_prompt,
    get_system_prompt,
    inject_system_prompt,
    generate_response_chunks, string_to_message, get_oai_client
)
from loguru import logger

log_path = Path(__file__)
log_path = Path(log_path.parent, "logs", "biochatter_endpoint.log")
logger.add(log_path.absolute(), rotation="10 MB")

app = FastAPI(title="Biochatter Knowledge Graph API endpoint.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def openai_completion(
        request: ChatCompletionRequest,
        model : str = DEFAULT_MODEL
) -> Optional[ChatCompletionResponse]:
    if request.stream:
        return
    client = get_oai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=request.temperature,
        stream=False,
        messages=request.messages,
        stop=request.stop,
    )
    logger.info(f"Finished OAI stream")
    return response

async def openai_completion_stream(
        request: ChatCompletionRequest,
        model : str = DEFAULT_MODEL
) -> AsyncGenerator[str, None]:
    if not request.stream:
        return
    client = get_oai_client()
    response =  client.chat.completions.create(
        model=model,
        temperature=request.temperature,
        stream=request.stream,
        messages=request.messages,
        stop=request.stop,
    )
    stop = request.stop[0] or "[DONE]"
    for chunk in response:
       # if not chunk.choices[0].delta.content:
       #     chunk.choices[0].delta.content = "[DONE]"
        chunk = chunk.model_dump_json()
        logger.debug(f"chunk : {str(chunk)}")
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(1)
    yield f"data: {stop}\n\n"
    logger.info(f"Finished OAI stream")



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
) -> Union[ChatCompletionResponseExt,Any]:
    logger.debug("POST /biochatter_api/chat/completions_ext")
    logger.debug(f"Input: {str(request)}")
    usage = ChatCompletionUsage()
    kg_context_injection : List[Context] = []
    resp_content = ""
    try:
        # session_id = request.session_id
        kg_config = vars(process_kg_config(request.kgConfig))
        use_kg = request.useKG

        system_prompt, rag_agent_prompts = get_system_prompt(request)
        has_prompt = has_system_prompt(request)
        model_options_fields = extract_common_fields(ModelOptions,request).model_dump()

        current_llm = ModelOptionsExt(
            **model_options_fields,
            api_key = get_api_key(),
            api_base = get_api_base()
        )
        # session: LLMSession = LLMSession(
        #     llm_options=current_llm.model_dump(),
        #     tools=None
        # )

        if has_prompt is not None:
            if has_prompt: #non-empty prompt
                if request.stream:
                     return StreamingResponse(
                        # session.stream_all([vars(msg) for msg in request.messages], run_callbacks=False),
                         openai_completion_stream(request),
                         media_type="application/x-ndjson"
                     )
                else:
                    result = await openai_completion(request)
                return result
        else:
            return get_completion_response(text="No messages provided!")

        if system_prompt:
            request.messages = inject_system_prompt(request, system_prompt)
            if  request.messages:
                messages = [vars(msg) for msg in request.messages]
                biochatter = BiochatterInstance( #instantiate
                #    session_id=request.session_id,
                    model_config=current_llm.model_dump(),
                    rag_agent_prompts= rag_agent_prompts
                )

                logger.debug("Starting biocahtter.chat")
                resp_content, usage, kg_context_injection = biochatter.chat(
                    messages=messages,
                    use_kg=use_kg,
                    kg_config=kg_config,
                )

                logger.debug(f"response: {str(resp_content)}")
                logger.debug(f"usage: {str(usage)}")
                logger.debug(f"kg_context: {str(kg_context_injection)}")
                usage = ChatCompletionUsage(**usage)
                err_code = ErrorCodes.SUCCESS
            else:
                err_code = ErrorCodes.INVALID_INPUT
                resp_content = "Something goes wrong, request did not contain messages!!!"
        else:
            err_code = ErrorCodes.MODEL_NOT_SUPPORTED

    except Exception as e:
        logger.error(str(e))
        err_code = ErrorCodes.SERVER_ERROR
        resp_content = str(e)

    response = get_completion_response(
        model=request.model,
        text=resp_content,
        usage=usage
    )

    if request.stream:
    #    messages.append(vars(string_to_message(resp_content)))
        return StreamingResponse(
            generate_response_chunks(
                response,
                request.stop[0] or "[DONE]"
            ),
#            session.stream_all(messages, run_callbacks=False),
            media_type="application/x-ndjson"
        )
    else:
        return ChatCompletionResponseExt(
            **response.model_dump(),
            contexts=kg_context_injection,
            err_code=err_code
        )


@app.post("/biochatter_api/chat/completions", description="chat completions")
async def chat_completions(
    request: ChatCompletionRequest
) -> Union[ChatCompletionResponse,Any]: #TODO: streaming model
    try:
        logger.trace("POST /biochatter_api/chat/completions")
        logger.debug(f"Input: {str(request)}")
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
    try:
        logger.debug(request)
        client = get_oai_client()
        messages=[string_to_message("Explain in less than 12 words why is the sky blue", role=Role.user)]
        logger.debug(f"Mes {str(messages)}")
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0,
            stream=False,
            messages=messages,
        )

        return response
    except Exception as e:
        logger.error(str(e))
        resp_content = str(e)
    return get_completion_response(text=resp_content)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_app_port())
