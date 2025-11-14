# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- `uv venv` - Create virtual environment
- `source .venv/bin/activate` - Activate virtual environment  
- `uv sync` - Install dependencies
- `cp .env.example .env` - Set up environment variables

### Code Quality
- `make format` or `ruff format .` - Format code
- `make lint` or `ruff check .` - Lint code
- `ruff check --fix .` - Auto-fix linting issues
- `ruff format --diff` - Show formatting differences

### Development Server
- `uv run langgraph dev --no-browser` - Start development server on http://localhost:2024
- The `--no-browser` flag prevents auto-opening LangGraph studio

## Architecture Overview

This is a LangGraph tools agent built for Open Agent Platform (OAP) that provides:

### Core Components
- **Agent Graph**: `tools_agent/agent.py:graph` - Main LangGraph react agent
- **Authentication**: `tools_agent/security/auth.py:auth` - Supabase JWT authentication
- **Tools Integration**: `tools_agent/utils/tools.py` - MCP server tools and RAG tool creation
- **Token Management**: `tools_agent/utils/token.py` - API key handling for OAP integration

### Configuration System
The agent uses `GraphConfigPydantic` class in `agent.py` for OAP integration. This class defines:
- Model selection (Claude, OpenAI, etc.)
- MCP server configurations (`MCPConfig`)
- RAG tool configurations (`RagConfig`) 
- System prompts and authentication requirements

### Authentication Flow
- Uses LangGraph custom auth with Supabase JWT tokens
- Requires `SUPABASE_URL` and `SUPABASE_KEY` environment variables
- Authenticates on thread/assistant CRUD operations and store access
- Auto-injects user ID into metadata for multi-tenant isolation

### Tools Architecture
- **MCP Integration**: Connects to external MCP servers for tool execution
- **RAG Tool**: LangConnect integration for retrieval-augmented generation
- **Tool Wrapping**: Automatic authentication wrapper for MCP tools via `wrap_mcp_authenticate_tool`

### API Key Management
- Supports user-provided API keys from OAP settings page
- Falls back to environment variables when user keys unavailable
- Keys passed through config `apiKeys` field from OAP

## Environment Variables Required
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_KEY` - Supabase service role key (for authentication)
- Model API keys (OpenAI, Anthropic, etc.) as fallbacks

## Project Structure
- `tools_agent/agent.py` - Main agent configuration and graph definition
- `tools_agent/security/` - Authentication handlers
- `tools_agent/utils/` - Tool creation utilities and token management
- `langgraph.json` - LangGraph server configuration
- `static/` - Static assets for the agent