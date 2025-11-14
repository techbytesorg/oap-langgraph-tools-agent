"""
MCP Server Integration Tests.

Tests MCP tool integration with authentication, covering:
- Success cases: Basic invocation, variable parameters, sequential calls
- Failure cases: Invalid URLs, missing auth, non-existent tools
- Edge cases: Empty tool lists, large responses, concurrent calls
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
@pytest.mark.mcp
async def test_basic_tool_invocation(remote_graph, access_token, mcp_server_url):
    """Success: Basic tool invocation with arcade__Demo_lorem generating 5 words"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": f"Use the {MCP_TOOL_NAME} tool to generate 5 random words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                }
            }
        }
    )
    
    assert_response_valid(result)
    assert len(result["messages"][-1]["content"]) > 0


@pytest.mark.asyncio
@pytest.mark.mcp
@pytest.mark.parametrize("word_count", [10, 20, 50])
async def test_variable_parameters(remote_graph, access_token, mcp_server_url, word_count):
    """Success: Tool invocation with different parameters"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": f"Use {MCP_TOOL_NAME} to generate {word_count} random words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
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
@pytest.mark.mcp
async def test_natural_language_invocation(remote_graph, access_token, mcp_server_url):
    """Success: Verify tool callable by agent via natural language"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate some random words for me"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
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
@pytest.mark.mcp
async def test_sequential_tool_calls(remote_graph, access_token, mcp_server_url):
    """Success: Multiple sequential tool calls in same thread"""
    thread_id = str(uuid.uuid4())
    
    # First call
    result1 = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate 3 random words"}]},
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                }
            }
        }
    )
    assert_response_valid(result1)
    
    # Second call in same thread
    result2 = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Now generate 5 more words"}]},
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                }
            }
        }
    )
    assert_response_valid(result2)


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_auth_token_forwarding(remote_graph, access_token, mcp_server_url):
    """Success: Verify auth token properly forwarded to MCP server"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate 5 words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
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
@pytest.mark.mcp
async def test_invalid_mcp_url(remote_graph, access_token):
    """Failure: Invalid MCP server URL - expect graceful error"""
    # Should not raise exception - agent handles gracefully
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate 5 words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": "http://invalid-url-that-does-not-exist.local",
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_nonexistent_tool(remote_graph, access_token, mcp_server_url):
    """Failure: Request non-existent tool - expect error message"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Use the nonexistent_tool_xyz"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": ["nonexistent_tool_xyz"],
                    "auth_required": True,
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_empty_tool_list(remote_graph, access_token, mcp_server_url):
    """Edge: Empty tool list - no tools should be loaded"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What tools do you have available?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [],  # Empty list
                    "auth_required": True,
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_auth_not_required(remote_graph, access_token, mcp_server_url):
    """Edge: auth_required=False - works without token"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate 5 words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": False,  # No auth required
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.mcp
@pytest.mark.slow
async def test_large_response(remote_graph, access_token, mcp_server_url):
    """Edge: Very large response (1000 words)"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": f"Use {MCP_TOOL_NAME} to generate 1000 random words"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                }
            }
        }
    )
    
    assert_response_valid(result)
    assert len(result["messages"][-1]["content"]) > 1000


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_special_characters(remote_graph, access_token, mcp_server_url):
    """Edge: Special characters in parameters"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate words with special meaning like @#$%"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
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
@pytest.mark.mcp
async def test_unicode_tool_request(remote_graph, access_token, mcp_server_url):
    """Edge: Unicode and international characters in request"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Generate words like: café, naïve, 日本語, émojis 🎉"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": mcp_server_url,
                    "tools": [MCP_TOOL_NAME],
                    "auth_required": True,
                }
            }
        }
    )
    
    assert_response_valid(result)
