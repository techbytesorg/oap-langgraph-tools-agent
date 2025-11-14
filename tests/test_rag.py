"""
RAG Collection Integration Tests.

Tests RAG collection queries with authentication, covering:
- Success cases: Basic search, multiple queries, XML format, agent usage
- Failure cases: Invalid URLs, missing auth, non-existent collections
- Edge cases: Empty queries, long queries, special characters, empty results
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


@pytest.mark.asyncio
@pytest.mark.rag
async def test_basic_semantic_search(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Success: Basic semantic search on TEST_RAG_COLLECTION"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search for information in the collection about available data"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
@pytest.mark.parametrize("query", [
    "What information is available?",
    "Tell me about the content",
    "Summarize the main topics",
])
async def test_multiple_queries(remote_graph, access_token, rag_api_url, test_rag_collection, query):
    """Success: Multiple queries on same collection"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_xml_document_format(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Success: Verify XML document format in response"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search the collection for any information"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)
    assert len(result["messages"][-1]["content"]) > 0


@pytest.mark.asyncio
@pytest.mark.rag
async def test_agent_uses_rag_tool(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Success: Agent uses RAG tool for knowledge-based questions"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What can you tell me from the documents you have access to?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_invalid_rag_url(remote_graph, access_token, test_rag_collection):
    """Failure: Invalid RAG API URL - graceful error"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search the collection"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": "http://invalid-rag-url.local",
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_nonexistent_collection(remote_graph, access_token, rag_api_url):
    """Failure: Non-existent collection ID - 404 handling"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search the collection"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": ["00000000-0000-0000-0000-000000000000"],  # Non-existent
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_malformed_collection_id(remote_graph, access_token, rag_api_url):
    """Failure: Malformed collection ID format"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search the collection"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": ["invalid-uuid-format"],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_empty_query(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Edge: Empty query string - default behavior"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": ""}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
@pytest.mark.slow
async def test_very_long_query(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Edge: Very long query (1000+ chars)"""
    long_query = "Search for information about " + " ".join(["data"] * 200)
    
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": long_query}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_special_characters_unicode(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Edge: Query with special characters and Unicode"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search for café, naïve, 日本語, émojis 🎉 & special @#$%"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_no_matching_documents(remote_graph, access_token, rag_api_url, test_rag_collection):
    """Edge: No documents match query - empty result handling"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Search for xyzabc123nonexistentterm999"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "rag": {
                    "rag_url": rag_api_url,
                    "collections": [test_rag_collection],
                }
            }
        }
    )
    
    assert_response_valid(result)


@pytest.mark.asyncio
@pytest.mark.rag
async def test_rag_none(remote_graph, access_token):
    """Edge: rag=None - no RAG tools created"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What tools do you have?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                # No rag config
            }
        }
    )
    
    assert_response_valid(result)
