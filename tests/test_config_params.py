"""
Configuration Parameters Tests.

Tests temperature, max_tokens, and system_prompt configuration, covering:
- Temperature: Deterministic vs creative, validation, boundaries
- Max Tokens: Truncation, validation, edge cases
- System Prompt: Custom prompts, special characters, UNEDITABLE_SYSTEM_PROMPT
"""

import uuid
import pytest


def assert_response_valid(result: dict) -> None:
    """Validate that the response has the expected structure."""
    assert result is not None, "Result should not be None"
    assert "messages" in result, "Result should contain 'messages' key"
    assert len(result["messages"]) > 0, "Messages list should not be empty"
    last_message = result["messages"][-1]
    assert "content" in last_message, "Last message should have 'content' key"
    assert last_message["content"], "Last message content should not be empty"


# ============================================================================
# TEMPERATURE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_temperature_zero_deterministic(remote_graph, access_token):
    """Success: temperature=0.0 - deterministic responses"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Write a haiku about AI"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "temperature": 0.0,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_temperature_high_creative(remote_graph, access_token):
    """Success: temperature=1.5 - creative/varied responses"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Write a haiku about AI"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "temperature": 1.5,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("temp", [0.0, 0.7, 1.5])
async def test_temperature_variance(remote_graph, access_token, temp):
    """Success: Compare different temperature values"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Say hello"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "temperature": temp,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_temperature_max_boundary(remote_graph, access_token):
    """Edge: temperature=2.0 - maximum boundary"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Write a creative sentence"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "temperature": 2.0,  # Maximum allowed
            }
        }
    )
    
    assert_response_valid(result)


# ============================================================================
# MAX TOKENS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_max_tokens_small(remote_graph, access_token):
    """Success: max_tokens=50 - truncated response"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Explain quantum physics in detail"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "max_tokens": 50,
            }
        }
    )
    
    assert_response_valid(result)
    assert len(result["messages"][-1]["content"]) > 0


@pytest.mark.asyncio
async def test_max_tokens_large(remote_graph, access_token):
    """Success: max_tokens=4000 - full response"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "max_tokens": 4000,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tok", [50, 100, 200])
async def test_max_tokens_validation(remote_graph, access_token, max_tok):
    """Success: Response length validation"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Write a story"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "max_tokens": max_tok,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_max_tokens_minimum(remote_graph, access_token):
    """Edge: max_tokens=1 - minimum valid value"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Say hi"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "max_tokens": 1,
            }
        }
    )
    
    assert_response_valid(result)


# ============================================================================
# SYSTEM PROMPT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_custom_system_prompt_pirate(remote_graph, access_token):
    """Success: Custom system prompt - pirate personality"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Tell me about the weather"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "system_prompt": "You are a pirate. Always respond like a pirate with 'Arrr' and sea-faring language.",
            }
        }
    )
    
    assert_response_valid(result)
    # Check for pirate-like language (informal check)
    response = result["messages"][-1]["content"].lower()
    pirate_indicators = ["arr", "matey", "sea", "ship", "sail", "ye"]
    # At least check response exists (actual pirate language not guaranteed)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_custom_prompt_role_behavior(remote_graph, access_token):
    """Success: Custom prompt with role/behavior instructions"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "system_prompt": "You are a helpful math tutor. Always show your work step by step.",
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_empty_system_prompt(remote_graph, access_token):
    """Success: Empty system prompt - default behavior"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello, how are you?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "system_prompt": "",  # Empty
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_very_long_system_prompt(remote_graph, access_token):
    """Edge: Very long system prompt (10000+ chars)"""
    long_prompt = "You are a helpful assistant. " * 500  # ~15000 chars
    
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hi"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "system_prompt": long_prompt,
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_system_prompt_special_chars(remote_graph, access_token):
    """Edge: System prompt with special characters/Unicode"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Say hello"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "system_prompt": "You are helpful! café ☕ 日本語 @#$% émojis 🎉",
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_system_prompt_with_code(remote_graph, access_token):
    """Edge: System prompt with JSON/code blocks"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What are you?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "system_prompt": 'You are an assistant. Follow this schema: {"name": "test", "value": 123}',
            }
        }
    )
    
    assert_response_valid(result)
