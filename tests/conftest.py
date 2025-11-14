"""
Pytest configuration and shared fixtures.
"""

import os
import pytest
from typing import Tuple
from supabase import create_client, Client
from langgraph.pregel.remote import RemoteGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
USER_EMAIL = os.environ.get("USER_EMAIL")
USER_PASSWORD = os.environ.get("USER_PASSWORD")
AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:2024")
ASSISTANT_ID = "agent"


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "mcp: marks tests that require MCP server")
    config.addinivalue_line("markers", "rag: marks tests that require RAG server")


@pytest.fixture(scope="session")
def auth_credentials() -> Tuple[str, str]:
    """
    Authenticate with Supabase and return access token and user info.
    Session-scoped to authenticate once per test session.
    
    Returns:
        Tuple[str, str]: (access_token, user_id)
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        pytest.skip("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    
    if not USER_EMAIL or not USER_PASSWORD:
        pytest.skip("USER_EMAIL and USER_PASSWORD must be set in .env file")
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    response = supabase.auth.sign_in_with_password({
        "email": USER_EMAIL,
        "password": USER_PASSWORD,
    })
    
    if response.session is None:
        pytest.fail(f"Supabase authentication failed: {response}")
    
    # Get user info to extract user_id
    access_token = response.session.access_token
    user_response = supabase.auth.get_user(access_token)
    user_id = user_response.user.id
    
    return access_token, user_id


@pytest.fixture
def access_token(auth_credentials) -> str:
    """Get access token from auth credentials"""
    return auth_credentials[0]


@pytest.fixture
def user_id(auth_credentials) -> str:
    """Get user ID from auth credentials"""
    return auth_credentials[1]


@pytest.fixture
def remote_graph(access_token) -> RemoteGraph:
    """
    Create and return a RemoteGraph instance with authentication headers.
    """
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    
    return RemoteGraph(
        ASSISTANT_ID,
        url=AGENT_URL,
        headers=headers
    )


@pytest.fixture
def mcp_server_url():
    """Get MCP server URL from environment"""
    url = os.environ.get("MCP_SERVER_URL")
    if not url:
        pytest.skip("MCP_SERVER_URL not set in environment")
    return url


@pytest.fixture
def rag_api_url():
    """Get RAG API URL from environment"""
    url = os.environ.get("RAG_API_URL")
    if not url:
        pytest.skip("RAG_API_URL not set in environment")
    return url


@pytest.fixture
def test_rag_collection():
    """Get test RAG collection ID from environment"""
    collection_id = os.environ.get("TEST_RAG_COLLECTION")
    if not collection_id:
        pytest.skip("TEST_RAG_COLLECTION not set in environment")
    return collection_id


def assert_response_valid(result: dict) -> None:
    """
    Validate that the response has the expected structure.
    
    Args:
        result: The result dictionary from agent invocation
    
    Raises:
        AssertionError: If the response structure is invalid
    """
    assert result is not None, "Result should not be None"
    assert "messages" in result, "Result should contain 'messages' key"
    assert len(result["messages"]) > 0, "Messages list should not be empty"
    
    # Check that the last message has content
    last_message = result["messages"][-1]
    assert "content" in last_message, "Last message should have 'content' key"
    assert last_message["content"], "Last message content should not be empty"

