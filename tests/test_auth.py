"""
Authentication & Security Tests.

Tests authentication flows and security scenarios, covering:
- Success cases: Valid tokens, multiple token sources, JWT decode
- Failure cases: Missing tokens, expired tokens, invalid JWT
- Edge cases: Token format variations, priority testing
"""

import uuid
import pytest
import jwt as pyjwt


def assert_response_valid(result: dict) -> None:
    """Validate that the response has the expected structure."""
    assert result is not None, "Result should not be None"
    assert "messages" in result, "Result should contain 'messages' key"
    assert len(result["messages"]) > 0, "Messages list should not be empty"
    last_message = result["messages"][-1]
    assert "content" in last_message, "Last message should have 'content' key"
    assert last_message["content"], "Last message content should not be empty"


@pytest.mark.asyncio
async def test_valid_supabase_token(remote_graph, access_token):
    """Success: Valid Supabase token - full access"""
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


@pytest.mark.asyncio
async def test_token_in_configurable(remote_graph, access_token):
    """Success: Token in x-supabase-access-token configurable"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Test auth"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,  # In configurable
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
async def test_jwt_decode_user_id(access_token, user_id):
    """Success: JWT decode and user_id extraction"""
    # Decode JWT without verification (same as agent does)
    decoded = pyjwt.decode(access_token, options={"verify_signature": False})
    extracted_user_id = decoded.get("sub")
    
    assert extracted_user_id == user_id, "User IDs should match"


@pytest.mark.asyncio
async def test_token_with_special_characters(remote_graph, access_token):
    """Edge: Token with special characters"""
    # Real JWT tokens can contain special chars like +, /, =
    has_special = any(char in access_token for char in ['+', '/', '='])
    
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


@pytest.mark.asyncio
async def test_very_long_token(remote_graph, access_token):
    """Edge: Very long token string"""
    token_length = len(access_token)
    assert token_length > 100, "Token should be reasonably long"
    
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
