# E2E Test Suite for LangChain v1 Agent

Comprehensive end-to-end test suite using **pytest** for the LangChain v1 agent migration, covering all configurable runtime context fields including MCP tools, RAG collections, model switching, and configuration parameters.

## Overview

This test suite includes **~80 test cases** across **7 test modules**:

- **test_mcp.py** (13 tests) - MCP server integration with authentication
- **test_rag.py** (13 tests) - RAG collection queries and document retrieval  
- **test_models.py** (8 tests) - Model switching and API key resolution
- **test_config_params.py** (16 tests) - Temperature, max_tokens, system_prompt
- **test_integration.py** (9 tests) - Multi-feature scenarios
- **test_auth.py** (5 tests) - Authentication flows and security
- **test_resilience.py** (8 tests) - Error recovery and stress testing

## Prerequisites

### 1. Environment Setup

Ensure you have a `.env` file in the project root with the following variables:

```bash
# Authentication (Required)
SUPABASE_URL=<your-supabase-url>
SUPABASE_KEY=<your-service-role-key>
USER_EMAIL=<test-user-email>
USER_PASSWORD=<test-user-password>

# API Keys (Required for model tests)
OPENAI_API_KEY=<your-openai-key>

# Services (Required for integration tests)
MCP_SERVER_URL=<your-mcp-server-url>
RAG_API_URL=<your-rag-api-url>
TEST_RAG_COLLECTION=<your-test-collection-id>

# Optional
AGENT_URL=http://localhost:2024  # Defaults to localhost:2024
```

### 2. LangGraph Development Server

The tests require a running LangGraph development server:

```bash
# Start the server (in a separate terminal)
uv run langgraph dev --no-browser
```

The server should be running at `http://localhost:2024` (or your configured `AGENT_URL`).

### 3. Dependencies

Install test dependencies:

```bash
# Using uv (recommended)
uv sync

# The test dependencies are in the 'test' dependency group
```

## Running Tests

### Run All Tests

```bash
# Run all tests
uv run pytest

# Or from project root
uv run pytest tests/
```

### Run Specific Test Module

```bash
# Run only MCP tests
uv run pytest tests/test_mcp.py

# Run only RAG tests
uv run pytest tests/test_rag.py

# Run only model switching tests
uv run pytest tests/test_models.py
```

### Run Tests by Marker

```bash
# Run only MCP-related tests
uv run pytest -m mcp

# Run only RAG-related tests
uv run pytest -m rag

# Run integration tests
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"

# Run only slow tests
uv run pytest -m slow
```

### Run Specific Test Function

```bash
# Run a single test
uv run pytest tests/test_mcp.py::test_basic_tool_invocation

# Run tests matching a pattern
uv run pytest -k "temperature"
uv run pytest -k "mcp and auth"
```

### Pytest Options

```bash
# Verbose output
uv run pytest -v

# Show print statements
uv run pytest -s

# Stop on first failure
uv run pytest -x

# Show test durations
uv run pytest --durations=10

# Run tests in parallel (if pytest-xdist installed)
uv run pytest -n auto

# Generate HTML report (if pytest-html installed)
uv run pytest --html=report.html
```

## Test Categories

### Success Cases (Happy Path)
Tests that verify features work correctly with valid inputs:
- Basic tool invocations
- Valid authentication
- Correct model responses
- Proper configuration handling

### Failure Cases (Error Handling)
Tests that verify graceful error handling:
- Invalid URLs
- Missing authentication
- Non-existent resources
- Malformed inputs

### Edge Cases (Boundary Conditions)
Tests that verify behavior at extremes:
- Empty/null values
- Very large inputs
- Concurrent operations
- Special characters

## Test Structure

### Fixtures (`conftest.py`)

Pytest fixtures provide shared setup:

- `auth_credentials` - Session-scoped Supabase authentication
- `access_token` - Supabase access token
- `user_id` - Authenticated user ID
- `remote_graph` - Configured RemoteGraph client
- `mcp_server_url` - MCP server URL (skips if not set)
- `rag_api_url` - RAG API URL (skips if not set)
- `test_rag_collection` - Test collection ID (skips if not set)
- `assert_response_valid()` - Response validation helper

### Test Pattern

Each test uses pytest conventions:

```python
@pytest.mark.asyncio
@pytest.mark.mcp  # Optional marker
async def test_feature_name(remote_graph, access_token, mcp_server_url):
    """Test description"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "prompt"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                # Feature-specific config...
            }
        }
    )
    
    assert_response_valid(result)
    assert len(result["messages"][-1]["content"]) > 0
```

## Expected Outputs

### Successful Test Run

```
=================== test session starts ====================
collected 80 items

tests/test_mcp.py::test_basic_tool_invocation PASSED [ 1%]
tests/test_mcp.py::test_variable_parameters[10] PASSED [ 2%]
...

=================== 80 passed in 120.45s ===================
```

### Failed Test Run

```
=================== test session starts ====================
collected 80 items

tests/test_mcp.py::test_basic_tool_invocation FAILED [ 1%]

======================= FAILURES ===========================
___________ test_basic_tool_invocation ____________

    async def test_basic_tool_invocation(...):
>       assert_response_valid(result)
E       AssertionError: Result should not be None

tests/test_mcp.py:25: AssertionError
================= 1 failed, 79 passed in 115.32s =============
```

### Skipped Tests

```
=================== test session starts ====================
collected 80 items

tests/test_mcp.py::test_basic_tool_invocation SKIPPED [1%]
...

============= 75 passed, 5 skipped in 98.21s ===============
```

Tests are automatically skipped if required environment variables are not set.

## Troubleshooting

### LangGraph Server Not Running

**Error:** Connection refused to `localhost:2024`

**Solution:**
```bash
# Start the server in a separate terminal
uv run langgraph dev --no-browser
```

### Authentication Failures

**Error:** `SUPABASE_URL and SUPABASE_KEY must be set`

**Solution:**
- Verify `.env` file exists in project root
- Check that all required environment variables are set
- Ensure `USER_EMAIL` and `USER_PASSWORD` are valid
- Tests will be skipped if credentials are missing

### MCP Tests Skipped

**Error:** Tests marked as `SKIPPED [MCP_SERVER_URL not set]`

**Solution:**
- Set `MCP_SERVER_URL` in `.env` file
- Verify MCP server is running
- Ensure `arcade__Demo_lorem` tool exists on the server

### RAG Tests Skipped

**Error:** Tests marked as `SKIPPED [RAG_API_URL not set]`

**Solution:**
- Set `RAG_API_URL` and `TEST_RAG_COLLECTION` in `.env`
- Verify the collection exists and has documents
- Ensure Supabase token has access to the collection

### Import Errors

**Error:** `ModuleNotFoundError`

**Solution:**
```bash
# Ensure dependencies are installed
uv sync

# Run pytest with uv
uv run pytest
```

### Pytest Not Found

**Error:** `pytest: command not found`

**Solution:**
```bash
# Install test dependencies
uv sync

# Always use uv run
uv run pytest
```

## Advanced Usage

### Pytest Configuration

Configuration is in `pytest.ini`:
- Test discovery patterns
- Default options
- Marker definitions
- Warning filters

### Custom Markers

```python
@pytest.mark.slow  # Long-running test
@pytest.mark.integration  # Integration test
@pytest.mark.mcp  # Requires MCP server
@pytest.mark.rag  # Requires RAG server
```

### Parametrized Tests

Some tests use `@pytest.mark.parametrize` to run multiple variations:

```python
@pytest.mark.parametrize("word_count", [10, 20, 50])
async def test_variable_parameters(remote_graph, access_token, word_count):
    # Test runs 3 times with different word_count values
    ...
```

### Fixtures and Dependency Injection

Pytest automatically injects fixtures as function parameters. Tests only request the fixtures they need.

## Adding New Tests

### 1. Create Test Function

Add to appropriate test module:

```python
@pytest.mark.asyncio
@pytest.mark.mcp  # If applicable
async def test_new_feature(remote_graph, access_token, mcp_server_url):
    """Test description"""
    result = await remote_graph.ainvoke(...)
    assert_response_valid(result)
    # Additional assertions
```

### 2. Run Your Test

```bash
# Run just your new test
uv run pytest tests/test_mcp.py::test_new_feature -v
```

### 3. Add Markers

Use appropriate markers:
- `@pytest.mark.slow` for tests >10 seconds
- `@pytest.mark.integration` for multi-feature tests
- `@pytest.mark.mcp` if requires MCP server
- `@pytest.mark.rag` if requires RAG server

## Test Coverage

The test suite covers:

✅ **Configuration Fields:**
- model_name (model switching)
- temperature (0.0 to 2.0)
- max_tokens (1 to 100000+)
- system_prompt (custom prompts)
- mcp_config (URL, tools, auth)
- rag (URL, collections)

✅ **Authentication:**
- Valid/invalid tokens
- Token sources (configurable/metadata)
- JWT decode
- Token priority

✅ **Tools:**
- MCP tool invocation
- RAG document search
- Tool failures and recovery
- Concurrent tool usage

✅ **Error Handling:**
- Network failures
- Invalid inputs
- Missing configurations
- Partial responses

✅ **Stress Testing:**
- Rapid requests
- Long conversations
- Large payloads
- Concurrent users

## Performance Benchmarks

Expected execution times (approximate):

| Module | Tests | Duration |
|--------|-------|----------|
| test_mcp.py | 13 | ~20-40s |
| test_rag.py | 13 | ~25-45s |
| test_models.py | 8 | ~15-30s |
| test_config_params.py | 16 | ~25-50s |
| test_integration.py | 9 | ~20-40s |
| test_auth.py | 5 | ~10-20s |
| test_resilience.py | 8 | ~20-40s |
| **Total** | **~80** | **~2-5 minutes** |

Times vary based on:
- Network latency
- LLM response times
- MCP/RAG server performance

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Start LangGraph Server
      run: |
        uv run langgraph dev --no-browser &
        sleep 10
      env:
        SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
        SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
        # ... other secrets
    
    - name: Run Tests
      run: uv run pytest -v -m "not slow"
      env:
        SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
        # ... all required env vars
```

## Support

For issues or questions:
1. Check this README for common problems
2. Review pytest output for specific error messages
3. Check the LangGraph server logs
4. Verify environment configuration

## License

Same as parent project.
