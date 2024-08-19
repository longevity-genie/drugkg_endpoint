import json
import os
from common import *
from typing import Any, List, Optional
from pydantic import BaseModel
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
from loguru import logger
import neo4j_utils as nu



def process_connection_args(rag: str, connection_args: dict) -> dict:
    if rag == RAG_KG:
        if connection_args.get("host", "").lower() == "local":
            connection_args["host"] = os.getenv("KGHOST", localhost)
    return connection_args

def find_schema_info_node(connection_args: dict):
    try:
        """
        Look for a schema info node in the connected BioCypher graph and load the
        schema info if present.
        """
        db_uri = "bolt://" + connection_args.get("host") + \
            ":" + connection_args.get("port")
        neodriver = nu.Driver(
            db_name=connection_args.get("db_name") or "neo4j",
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
            model_name="gpt-4o-mini",
            connection_args=connection_args,
            schema_config_or_info_dict=schema_dict,
        )
        logger.debug("Agent connected: {}", lambda: rag_agent.agent.is_connected())
        return rag_agent.agent.is_connected()
    except Exception as e:
        logger.error(e)
        return False
