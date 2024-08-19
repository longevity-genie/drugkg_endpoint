import time
from common import *
from pathlib import Path
from fastapi import FastAPI
from just_agents.llm_session import LLMSession
from starlette.responses import StreamingResponse
from dotenv import load_dotenv
from just_agents.utils import RotateKeys
from fastapi.middleware.cors import CORSMiddleware
from kg_agent import get_kg_connection_status
import loguru
# import litellm
# litellm.set_verbose=True
log_path = Path(__file__)
log_path = Path(log_path.parent, "logs", "symptom_checker.log")
loguru.logger.add(log_path.absolute(), rotation="10 MB")

load_dotenv(override=True)
# What drug interactions of rapamycin are you aware of? What are these interactions
app = FastAPI(title="Biochatter Knowledge Graph API endpoint.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", description="Default message", response_model=str)
async def default():
    return "BioChatter KG API Endpoint v1.0"

@app.post("/biochatter_api/kg_status", description="returns knowledge graph connection status")
async def get_kg_connection_status(
    item: KGConnectionArgs,
):
    try:
        connection_args = item.connectionArgs
        connection_args = vars(connection_args)
        connected = get_kg_connection_status(connection_args)
        return {
            "status": "connected" if connected else "disconnected",
            "code": ErrorCodes.SUCCESS,
        }
    except Exception as e:
        return {"error": str(e), "code": ErrorCodes.SERVER_ERROR}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50501)