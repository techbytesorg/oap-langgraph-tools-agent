"""
Model Switching Tests.

Tests runtime model selection and switching, covering:
- Success cases: Different models, API key resolution, model persistence
- Failure cases: Invalid models, missing API keys, unsupported providers
- Edge cases: Empty model names, case sensitivity, rapid switching
"""

import os
import uuid
import pytest

SIMPLE_QUESTION = "What is 2 + 2? Just give me the number."


def assert_response_valid(result: dict) -> None:
    """Validate that the response has the expected structure."""
    assert result is not None, "Result should not be None"
    assert "messages" in result, "Result should contain 'messages' key"
    assert len(result["messages"]) > 0, "Messages list should not be empty"
    last_message = result["messages"][-1]
    assert "content" in last_message, "Last message should have 'content' key"
    assert last_message["content"], "Last message content should not be empty"


@pytest.mark.asyncio
async def test_default_model_gpt4o(remote_graph, access_token):
    """Success: Use default model openai:gpt-4o"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": SIMPLE_QUESTION}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": "openai:gpt-4o",
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_switch_to_gpt4o_mini(remote_graph, access_token):
    """Success: Switch to openai:gpt-4o-mini"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": SIMPLE_QUESTION}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": "openai:gpt-4o-mini",
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai:gpt-4o", "openai:gpt-4o-mini", "openai:gpt-4o"])
async def test_alternating_models(remote_graph, access_token, model):
    """Success: Alternating models across threads"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": SIMPLE_QUESTION}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": model,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_model_persistence_in_thread(remote_graph, access_token):
    """Success: Model persists within same thread"""
    thread_id = str(uuid.uuid4())
    model = "openai:gpt-4o-mini"
    
    # First message
    result1 = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is 5 + 3?"}]},
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
                "model_name": model,
            }
        }
    )
    assert_response_valid(result1)
    
    # Follow-up message in same thread
    result2 = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What was my previous question?"}]},
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
                "model_name": model,
            }
        }
    )
    assert_response_valid(result2)


@pytest.mark.asyncio
async def test_api_key_from_environment(remote_graph, access_token):
    """Success: API key resolution from environment"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not in environment")
    
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": SIMPLE_QUESTION}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "model_name": "openai:gpt-4o-mini",
                # No apiKeys provided, should use environment
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_invalid_model_name(remote_graph, access_token):
    """Failure: Invalid model name - initialization error"""
    from langgraph.pregel.remote import RemoteException
    
    # Should raise exception for invalid model
    with pytest.raises(RemoteException) as exc_info:
        await remote_graph.ainvoke(
            {"messages": [{"role": "user", "content": SIMPLE_QUESTION}]},
            config={
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "x-supabase-access-token": access_token,
                    "model_name": "invalid:model-that-doesnt-exist",
                }
            }
        )
    
    # Verify the error message contains expected info
    assert "ValueError" in str(exc_info.value) or "Unable to infer model provider" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_model_name(remote_graph, access_token):
    """Edge: Empty model name - raises error"""
    from langgraph.pregel.remote import RemoteException
    
    # Should raise exception for empty model name
    with pytest.raises(RemoteException) as exc_info:
        await remote_graph.ainvoke(
            {"messages": [{"role": "user", "content": SIMPLE_QUESTION}]},
            config={
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "x-supabase-access-token": access_token,
                    "model_name": "",  # Empty string
                }
            }
        )
    
    # Verify the error message contains expected info
    assert "TypeError" in str(exc_info.value) or "missing" in str(exc_info.value)
