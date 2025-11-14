"""
Error Recovery & Resilience Tests.

Tests system resilience and error recovery, covering:
- Recovery cases: Tool failures, network interruptions, partial responses
- Stress tests: Rapid requests, long conversations, large payloads, concurrent users
"""

import asyncio
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
@pytest.mark.slow
async def test_tool_fails_agent_continues(remote_graph, access_token):
    """Recovery: Tool fails, agent retries or provides fallback"""
    # Request a non-existent tool, agent should handle gracefully
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Use the nonexistent_tool_xyz to help me"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": "http://localhost:8000",
                    "tools": ["nonexistent_tool_xyz"],
                    "auth_required": False,
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_partial_response_handling(remote_graph, access_token):
    """Recovery: Partial response handling"""
    # Request with very low max_tokens, may get partial response
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Explain quantum physics, general relativity, and string theory in detail"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "max_tokens": 20,  # Very low, will truncate
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_agent_continues_after_error(remote_graph, access_token):
    """Recovery: Agent continues after tool error"""
    thread_id = str(uuid.uuid4())
    
    # First request with bad tool config
    await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Use the broken tool"}]},
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
                "mcp_config": {
                    "url": "http://invalid-url.local",
                    "tools": ["broken_tool"],
                    "auth_required": False,
                }
            }
        }
    )
    
    # Second request in same thread without tool
    result2 = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
            }
        }
    )
    
    assert_response_valid(result2)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_rapid_successive_requests(remote_graph, access_token):
    """Stress: Rapid successive requests (rate limiting)"""
    num_requests = 5
    
    for i in range(num_requests):
        result = await remote_graph.ainvoke(
            {"messages": [{"role": "user", "content": f"Request {i+1}: What is {i}+{i}?"}]},
            config={
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "x-supabase-access-token": access_token,
                }
            }
        )
        
        assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_very_long_conversation(remote_graph, access_token):
    """Stress: Very long conversation thread"""
    thread_id = str(uuid.uuid4())
    num_turns = 10
    
    for i in range(num_turns):
        result = await remote_graph.ainvoke(
            {"messages": [{"role": "user", "content": f"Turn {i+1}: Count to {i+1}"}]},
            config={
                "configurable": {
                    "thread_id": thread_id,  # Same thread
                    "x-supabase-access-token": access_token,
                }
            }
        )
        
        assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_large_message_payload(remote_graph, access_token):
    """Stress: Large message payloads"""
    # Create a large prompt
    large_prompt = "Analyze this text: " + " ".join(["data"] * 1000)  # ~5KB prompt
    
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": large_prompt}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_requests_same_user(remote_graph, access_token):
    """Stress: Concurrent requests from same user"""
    num_concurrent = 3
    
    # Create concurrent requests
    tasks = []
    for i in range(num_concurrent):
        task = remote_graph.ainvoke(
            {"messages": [{"role": "user", "content": f"Concurrent {i+1}: What is {i*2}+{i*2}?"}]},
            config={
                "configurable": {
                    "thread_id": str(uuid.uuid4()),  # Different threads
                    "x-supabase-access-token": access_token,
                }
            }
        )
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check results
    for result in results:
        if not isinstance(result, Exception):
            assert_response_valid(result)
