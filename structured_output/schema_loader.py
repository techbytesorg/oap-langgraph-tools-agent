"""
Schema management script for CRUD operations on LangGraph memory store schemas.
"""

import os
import asyncio
from pydantic import BaseModel, Field
from typing import List
from langgraph_sdk import get_client
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
USER_EMAIL = os.environ.get("USER_EMAIL")
USER_PASSWORD = os.environ.get("USER_PASSWORD")

# LangGraph configuration
LANGGRAPH_API_URL = "http://localhost:2024"  # Change if needed

def authenticate_supabase() -> str:
    """Authenticate with Supabase and return access token"""
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.auth.sign_in_with_password({
        "email": USER_EMAIL,
        "password": USER_PASSWORD,
    })
    if response.session is None:
        raise RuntimeError(f"Supabase authentication failed: {response}")
    return response.session.access_token

def get_authenticated_client():
    """Get authenticated LangGraph client"""
    access_token = authenticate_supabase()
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    return get_client(url=LANGGRAPH_API_URL, headers=headers)

# Define test schemas using Pydantic

class OutputSchemaA(BaseModel):
    name: str = Field(..., description="Name of the item")
    allergen: str = Field(..., description="Allergen information")

class OutputSchemaB(BaseModel):
    product_id: int = Field(..., description="Product identifier")
    price: float = Field(..., description="Price of the product")

class AllergenEntry(BaseModel):
    allergen: str = Field(..., description="Name of the allergen, e.g., 'peanut'")
    reason: str = Field(..., description="Reason this allergen is high risk")

class AllergyAnalysisResponse(BaseModel):
    high_risk_allergen: List[AllergenEntry] = Field(description="List of allergen-risk mappings")
    recommendation: str = Field(description="Overall recommendation for handling this recipe")

async def store_schemas():
    """Store all test schemas in the LangGraph memory store"""
    client = get_authenticated_client()

    # Get user info to build namespace
    access_token = authenticate_supabase()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    user_response = supabase.auth.get_user(access_token)
    user_id = user_response.user.id

    schemas = {
        "OutputSchemaA": OutputSchemaA.model_json_schema(),
        "OutputSchemaB": OutputSchemaB.model_json_schema(),
        "AllergyAnalysisResponse": AllergyAnalysisResponse.model_json_schema(),
    }

    for name, schema_json in schemas.items():
        # Use [user_id, "schemas"] as namespace to satisfy auth requirements
        await client.store.put_item([user_id, "schemas"], name, schema_json)
        print(f"Stored schema: {name}")

async def list_schemas() -> List[str]:
    """List all available schemas in the memory store"""
    client = get_authenticated_client()

    # Get user info to build namespace
    access_token = authenticate_supabase()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    user_response = supabase.auth.get_user(access_token)
    user_id = user_response.user.id

    results = await client.store.search_items([user_id, "schemas"])
    schemas = [item["key"] for item in results["items"]]
    print("Available schemas:")
    for schema in schemas:
        print(f"  - {schema}")
    return schemas

async def delete_schema(schema_name: str):
    """Delete a schema from the memory store"""
    client = get_authenticated_client()

    # Get user info to build namespace
    access_token = authenticate_supabase()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    user_response = supabase.auth.get_user(access_token)
    user_id = user_response.user.id

    await client.store.delete_item([user_id, "schemas"], schema_name)
    print(f"Deleted schema: {schema_name}")

async def main():
    print("Loading test schemas for structured output...")
    await store_schemas()
    await list_schemas()

    # Example: delete a schema (uncomment to test)
    # await delete_schema("OutputSchemaB")
    # await list_schemas()

if __name__ == "__main__":
    asyncio.run(main())
