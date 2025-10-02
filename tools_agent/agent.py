import os
import logging
import jwt
from langchain_core.runnables import RunnableConfig
from typing import Optional, List
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from tools_agent.utils.tools import create_rag_tool
from langchain.chat_models import init_chat_model
from tools_agent.utils.token import fetch_tokens
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_core.tools import StructuredTool
from tools_agent.utils.tools import (
    wrap_mcp_authenticate_tool,
    create_langchain_mcp_tool,
)
from tools_agent.utils.structured_output import load_schema_model
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store

logger = logging.getLogger(__name__)

# Initialize LangGraph memory store
store = InMemoryStore()

UNEDITABLE_SYSTEM_PROMPT = "\nIf the tool throws an error requiring authentication, provide the user with a Markdown link to the authentication page and prompt them to authenticate."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that has access to a variety of tools."
)


class RagConfig(BaseModel):
    rag_url: Optional[str] = None
    """The URL of the rag server"""
    collections: Optional[List[str]] = None
    """The collections to use for rag"""


class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""


class GraphConfigPydantic(BaseModel):
    model_name: Optional[str] = Field(
        default="openai:gpt-4o",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o",
                "description": "The model to use in all generations",
                "options": [
                    {
                        "label": "Claude Sonnet 4",
                        "value": "anthropic:claude-sonnet-4-0",
                    },
                    {
                        "label": "Claude 3.7 Sonnet",
                        "value": "anthropic:claude-3-7-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Sonnet",
                        "value": "anthropic:claude-3-5-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Haiku",
                        "value": "anthropic:claude-3-5-haiku-latest",
                    },
                    {"label": "o4 mini", "value": "openai:o4-mini"},
                    {"label": "o3", "value": "openai:o3"},
                    {"label": "o3 mini", "value": "openai:o3-mini"},
                    {"label": "GPT 4o", "value": "openai:gpt-4o"},
                    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
                    {"label": "GPT 4.1 mini", "value": "openai:gpt-4.1-mini"},
                ],
            }
        },
    )
    temperature: Optional[float] = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        },
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        },
    )
    system_prompt: Optional[str] = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": f"The system prompt to use in all generations. The following prompt will always be included at the end of the system prompt:\n---{UNEDITABLE_SYSTEM_PROMPT}\n---",
                "default": DEFAULT_SYSTEM_PROMPT,
            }
        },
    )
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                # Here is where you would set the default tools.
                # "default": {
                #     "tools": ["Math_Divide", "Math_Mod"]
                # }
            }
        },
    )
    rag: Optional[RagConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "rag",
                # Here is where you would set the default collection. Use collection IDs
                # "default": {
                #     "collections": [
                #         "fd4fac19-886c-4ac8-8a59-fff37d2b847f",
                #         "659abb76-fdeb-428a-ac8f-03b111183e25",
                #     ]
                # },
            }
        },
    )


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    model_name = model_name.lower()
    model_to_key = {
        "openai:": "OPENAI_API_KEY",
        "anthropic:": "ANTHROPIC_API_KEY", 
        "google": "GOOGLE_API_KEY"
    }
    key_name = next((key for prefix, key in model_to_key.items() 
                    if model_name.startswith(prefix)), None)
    if not key_name:
        return None
    api_keys = config.get("configurable", {}).get("apiKeys", {})
    if api_keys and api_keys.get(key_name) and len(api_keys[key_name]) > 0:
        return api_keys[key_name]
    # Fallback to environment variable
    return os.getenv(key_name)


async def graph(config: RunnableConfig):
    cfg = GraphConfigPydantic(**config.get("configurable", {}))
    tools = []

    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if cfg.rag and cfg.rag.rag_url and cfg.rag.collections and supabase_token:
        for collection in cfg.rag.collections:
            rag_tool = await create_rag_tool(
                cfg.rag.rag_url, collection, supabase_token
            )
            tools.append(rag_tool)

    logger.info(f"MCP Config check: mcp_config={cfg.mcp_config is not None}")
    if cfg.mcp_config:
        logger.info(f"MCP Config details: url={cfg.mcp_config.url}, tools={cfg.mcp_config.tools}, auth_required={cfg.mcp_config.auth_required}")

    if cfg.mcp_config and cfg.mcp_config.auth_required:
        logger.info("MCP auth required, fetching tokens...")
        mcp_tokens = await fetch_tokens(config)
        logger.info(f"Fetched MCP tokens: {mcp_tokens is not None}")
    else:
        logger.info("MCP auth not required, skipping token fetch")
        mcp_tokens = None

    logger.info(f"MCP connection check: has_config={cfg.mcp_config is not None}, has_url={cfg.mcp_config and cfg.mcp_config.url is not None}, has_tools={cfg.mcp_config and cfg.mcp_config.tools is not None}, auth_ok={mcp_tokens is not None or (cfg.mcp_config and not cfg.mcp_config.auth_required)}")

    if (
        cfg.mcp_config
        and cfg.mcp_config.url
        and cfg.mcp_config.tools
        and (mcp_tokens or not cfg.mcp_config.auth_required)
    ):
        server_url = cfg.mcp_config.url.rstrip("/") + "/mcp"
        logger.info(f"Connecting to MCP server: {server_url}")
        logger.info(f"Looking for tools: {cfg.mcp_config.tools}")

        tool_names_to_find = set(cfg.mcp_config.tools)
        fetched_mcp_tools_list: list[StructuredTool] = []
        names_of_tools_added = set()

        # If the tokens are not None, then we need to add the authorization header. otherwise make headers None
        headers = (
            mcp_tokens is not None
            and {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
            or None
        )
        logger.info(f"Using auth headers: {headers is not None}")
        try:
            logger.info("Opening MCP client connection...")
            async with streamablehttp_client(server_url, headers=headers) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    logger.info("Initializing MCP session...")
                    await session.initialize()
                    logger.info("MCP session initialized successfully")

                    page_cursor = None

                    while True:
                        tool_list_page = await session.list_tools(cursor=page_cursor)

                        if not tool_list_page or not tool_list_page.tools:
                            break

                        for mcp_tool in tool_list_page.tools:
                            if not tool_names_to_find or (
                                mcp_tool.name in tool_names_to_find
                                and mcp_tool.name not in names_of_tools_added
                            ):
                                logger.info(f"Adding MCP tool: {mcp_tool.name}")
                                langchain_tool = create_langchain_mcp_tool(
                                    mcp_tool, mcp_server_url=server_url, headers=headers
                                )
                                fetched_mcp_tools_list.append(
                                    wrap_mcp_authenticate_tool(langchain_tool)
                                )
                                if tool_names_to_find:
                                    names_of_tools_added.add(mcp_tool.name)

                        page_cursor = tool_list_page.nextCursor

                        if not page_cursor:
                            break
                        if tool_names_to_find and len(names_of_tools_added) == len(
                            tool_names_to_find
                        ):
                            break

                    logger.info(f"Successfully loaded {len(fetched_mcp_tools_list)} MCP tool(s)")
                    tools.extend(fetched_mcp_tools_list)
        except Exception as e:
            logger.error(f"Failed to fetch MCP tools: {e}", exc_info=True)
            print(f"Failed to fetch MCP tools: {e}")
            pass

    model = init_chat_model(
        cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=get_api_key_for_model(cfg.model_name, config) or "No token found"
    )

    # Check for structured output schema
    schema_name = config.get("configurable", {}).get("OutputSchemaName", None)
    response_format = None

    if schema_name:
        logger.debug(f"Processing structured output request for schema: {schema_name}")
        try:
            # Extract user ID from JWT token
            user_id = None
            if supabase_token:
                try:
                    decoded = jwt.decode(supabase_token, options={"verify_signature": False})
                    user_id = decoded.get("sub")
                    logger.debug(f"Extracted user_id from JWT token")
                except Exception as jwt_error:
                    logger.warning(f"JWT decode error: {jwt_error}")

            response_format = await load_schema_model(schema_name, user_id)
            logger.info(f"Successfully loaded schema {schema_name} for structured output")
        except Exception as e:
            logger.error(f"Error loading schema {schema_name}: {e}")
            # Continue without structured output
            response_format = None

    logger.info(f"Creating agent with {len(tools)} tools")
    if tools:
        logger.info(f"Tool names: {[tool.name for tool in tools]}")

    return create_react_agent(
        prompt=cfg.system_prompt + UNEDITABLE_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        config_schema=GraphConfigPydantic,
        response_format=response_format,
        store=store
    )
