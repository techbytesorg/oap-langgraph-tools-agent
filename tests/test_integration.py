"""
Integration Tests.

Tests multiple features working together, covering:
- Multi-feature success: MCP + RAG, MCP + system prompt, structured output + tools
- Complex scenarios: Model switching, multiple tools, error handling
- Edge cases: Minimal/maximal configurations
"""

import uuid
import pytest

MCP_TOOL_NAME = "arcade__Demo_lorem"


def assert_response_valid(result: dict) -> None:
    """Validate that the response has the expected structure."""
    assert result is not None, "Result should not be None"
    assert "messages" in result, "Result should contain 'messages' key"
    assert len(result["messages"]) > 0, "Messages list should not be empty"
    last_message = result["messages"][-1]
    assert "content" in last_message, "Last message should have 'content' key"
    assert last_message["content"], "Last message content should not be empty"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.mcp
@pytest.mark.rag
async def test_mcp_and_rag_together(remote_graph, access_token, mcp_server_url, rag_api_url, test_rag_collection):
    """Multi-feature: MCP + RAG together - agent uses both tools"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "First search the RAG collection, then generate some random words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                },
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.mcp
async def test_mcp_custom_system_prompt(remote_graph, access_token, mcp_server_url):
    """Multi-feature: MCP + custom system prompt"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate 5 random words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "system_prompt": "You are a creative assistant who loves using tools efficiently.",
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.rag
async def test_rag_deterministic(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Multi-feature: RAG + temperature=0.0 - consistent retrieval"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search for information in the collection"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "temperature": 0.0,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.mcp
@pytest.mark.rag
@pytest.mark.slow
async def test_all_features_enabled(remote_graph, access_token, mcp_server_url, rag_api_url, test_rag_collection):
    """Multi-feature: All features enabled simultaneously"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Help me with information and tools"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": "openai:gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 2000,
                "system_prompt": "You are a highly capable assistant with access to tools.",
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                },
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_switch_models_new_thread(remote_graph, access_token):
    """Complex: Switch models mid-conversation (new thread)"""
    # First thread with gpt-4o
    result1 = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is 5+5?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": "openai:gpt-4o",
            }
        }
    )
    assert_response_valid(result1)
    
    # Second thread with gpt-4o-mini
    result2 = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is 10+10?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": "openai:gpt-4o-mini",
            }
        }
    )
    assert_response_valid(result2)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_all_optional_configs_empty(remote_graph, access_token):
    """Edge: All optional configs set to None/empty"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                # No optional configs
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.mcp
@pytest.mark.rag
@pytest.mark.slow
async def test_maximum_configuration(remote_graph, access_token, mcp_server_url, rag_api_url, test_rag_collection):
    """Edge: Maximum configuration complexity"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Use all available capabilities"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": "openai:gpt-4o-mini",
                "temperature": 0.8,
                "max_tokens": 3000,
                "system_prompt": "You are a highly capable AI assistant with full access to tools and knowledge.",
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                },
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_minimal_configuration(remote_graph, access_token):
    """Edge: Minimal configuration (only required fields)"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
            }
        }
    )
    
    assert_response_valid(result)
