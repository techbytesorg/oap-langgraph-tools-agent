"""
Pytest configuration and shared fixtures.
"""

import os
import pytest
from typing import Tuple, List, Dict
from supabase import create_client, Client
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_client
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
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

    response = supabase.auth.sign_in_with_password(
        {
            "email": USER_EMAIL,
            "password": USER_PASSWORD,
        }
    )

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

    return RemoteGraph(ASSISTANT_ID, url=AGENT_URL, headers=headers)


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


# ============================================================================
# Structured Output Schema Definitions
# ============================================================================


class RecipeNutritionAnalysis(BaseModel):
    """Nutritional analysis of a recipe with validation constraints"""

    calories: int = Field(..., ge=0, le=10000, description="Total calories per serving")
    protein_grams: float = Field(..., ge=0, description="Protein in grams")
    fat_grams: float = Field(..., ge=0, description="Fat in grams")
    carbs_grams: float = Field(..., ge=0, description="Carbohydrates in grams")
    servings: int = Field(..., ge=1, le=20, description="Number of servings")
    prep_time_minutes: int = Field(
        ...,
        ge=1,
        multiple_of=5,
        description="Preparation time in minutes (multiples of 5)",
    )


class IngredientClassification(BaseModel):
    """Classification of a food ingredient"""

    ingredient_name: str = Field(..., description="Name of the ingredient")
    category: str = Field(
        ...,
        description="Primary food category",
        json_schema_extra={
            "enum": [
                "protein",
                "grain",
                "vegetable",
                "fruit",
                "dairy",
                "condiment",
                "spice",
            ]
        },
    )
    dietary_restrictions: List[str] = Field(
        ...,
        description="Dietary restrictions this ingredient fits",
        json_schema_extra={
            "items": {
                "enum": [
                    "vegan",
                    "vegetarian",
                    "gluten-free",
                    "dairy-free",
                    "nut-free",
                    "kosher",
                    "halal",
                ]
            }
        },
    )
    allergen_tags: List[str] = Field(
        ..., min_length=0, max_length=5, description="Allergen warnings (max 5)"
    )


class Violation(BaseModel):
    """A food safety violation"""

    violation_type: str = Field(..., description="Type of violation")
    severity: str = Field(
        ...,
        description="Severity level",
        json_schema_extra={"enum": ["minor", "major", "critical"]},
    )
    description: str = Field(..., description="Detailed description of the violation")
    corrective_action: str = Field(..., description="Required corrective action")


class FoodSafetyReport(BaseModel):
    """Food safety inspection report with nested violations"""

    inspection_id: str = Field(..., description="Unique inspection identifier")
    inspection_date: str = Field(..., description="Date of inspection (YYYY-MM-DD)")
    violations: List[Violation] = Field(..., description="List of violations found")
    overall_score: int = Field(
        ..., ge=0, le=100, description="Overall safety score (0-100)"
    )


class MealOption(BaseModel):
    """A single meal option"""

    dish_name: str = Field(..., description="Name of the dish")
    main_ingredients: List[str] = Field(
        ..., min_length=1, description="List of main ingredients"
    )
    estimated_cost: float = Field(..., gt=0, description="Estimated cost per serving")


class DailyMenu(BaseModel):
    """Menu for a single day"""

    day_name: str = Field(..., description="Day of the week")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    breakfast: MealOption = Field(..., description="Breakfast option")
    lunch: MealOption = Field(..., description="Lunch option")
    dinner: MealOption = Field(..., description="Dinner option")


class MenuPlanning(BaseModel):
    """Weekly menu planning for a restaurant"""

    week_start_date: str = Field(..., description="Start date of the week (YYYY-MM-DD)")
    daily_menus: List[DailyMenu] = Field(
        ..., min_length=1, max_length=7, description="Daily menu plans"
    )
    theme: str = Field(..., description="Overall theme for the week")


class RecipeStep(BaseModel):
    """A recipe step that can have sub-steps"""

    step_number: int = Field(..., ge=1, description="Step number in sequence")
    instruction: str = Field(..., description="Instruction text")
    duration_minutes: int = Field(..., ge=0, description="Time required for this step")
    sub_steps: List["RecipeStep"] | None = Field(
        default=None, description="Optional sub-steps"
    )


class RecipeInstructions(BaseModel):
    """Recipe instructions with potentially nested steps"""

    recipe_name: str = Field(..., description="Name of the recipe")
    total_time_minutes: int = Field(
        ..., ge=0, description="Total preparation and cooking time"
    )
    steps: List[RecipeStep] = Field(
        ..., min_length=1, description="List of preparation steps"
    )


# Enable forward references for RecipeStep
RecipeStep.model_rebuild()


class SupplierQuote(BaseModel):
    """Supplier quotation with optional fields"""

    supplier_name: str = Field(..., description="Name of the supplier company")
    contact_email: str = Field(
        ...,
        description="Contact email address",
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    )
    contact_phone: str = Field(
        ..., description="Contact phone number", pattern=r"^\(\d{3}\) \d{3}-\d{4}$"
    )
    quoted_price: float = Field(..., gt=0, description="Quoted price in dollars")
    delivery_date: str = Field(
        ...,
        description="Expected delivery date (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    notes: str | None = Field(None, description="Optional notes about the quote")


class QualityInspection(BaseModel):
    """Quality control inspection record"""

    inspector_email: str = Field(
        ...,
        description="Inspector's email",
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    )
    inspection_datetime: str = Field(
        ...,
        description="Date and time of inspection (ISO format)",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
    )
    batch_id: str = Field(
        ..., description="Batch identifier", pattern=r"^BT-\d{4}-\d{4}$"
    )
    quality_grade: str = Field(
        ...,
        description="Quality grade",
        json_schema_extra={"enum": ["A", "B", "C", "D", "F"]},
    )
    status: str = Field(
        ...,
        description="Inspection status",
        json_schema_extra={
            "enum": ["pending", "approved", "rejected", "review_required"]
        },
    )
    findings: List[str] = Field(..., description="List of inspection findings")


class AllergenEntry(BaseModel):
    """An allergen with associated risk information"""

    allergen: str = Field(..., description="Name of the allergen, e.g., 'peanut'")
    reason: str = Field(..., description="Reason this allergen is high risk")


class AllergyAnalysisResponse(BaseModel):
    """Analysis of allergens in a recipe"""

    high_risk_allergen: List[AllergenEntry] = Field(
        description="List of allergen-risk mappings"
    )
    recommendation: str = Field(
        description="Overall recommendation for handling this recipe"
    )


# ============================================================================
# Structured Output Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def schema_definitions() -> Dict[str, BaseModel]:
    """
    Return all 8 Pydantic schema models for testing.
    Session-scoped so schemas are defined once.
    """
    return {
        "RecipeNutritionAnalysis": RecipeNutritionAnalysis,
        "IngredientClassification": IngredientClassification,
        "FoodSafetyReport": FoodSafetyReport,
        "MenuPlanning": MenuPlanning,
        "RecipeInstructions": RecipeInstructions,
        "SupplierQuote": SupplierQuote,
        "QualityInspection": QualityInspection,
        "AllergyAnalysisResponse": AllergyAnalysisResponse,
    }


@pytest.fixture(scope="session")
async def schema_store(auth_credentials, schema_definitions):
    """
    Store all schemas in LangGraph memory store at session start.
    Yields the schema definitions for tests to use.
    Cleans up all schemas at session end.

    This is session-scoped to avoid repeated setup/teardown.
    """
    access_token, user_id = auth_credentials

    # Get authenticated client
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    client = get_client(url=AGENT_URL, headers=headers)

    # Setup: Store all schemas
    schemas_json = {}
    for name, schema_class in schema_definitions.items():
        schemas_json[name] = schema_class.model_json_schema()

    print(f"\n[Schema Store] Storing {len(schemas_json)} schemas for user {user_id}...")
    for name, schema_json in schemas_json.items():
        await client.store.put_item([user_id, "schemas"], name, schema_json)
        print(f"  ✅ Stored schema: {name}")

    # Yield for tests
    yield schemas_json

    # Teardown: Delete all schemas
    print(f"\n[Schema Store] Cleaning up {len(schemas_json)} schemas...")
    for name in schemas_json.keys():
        try:
            await client.store.delete_item([user_id, "schemas"], name)
            print(f"  🗑️  Deleted schema: {name}")
        except Exception as e:
            print(f"  ⚠️  Failed to delete schema {name}: {e}")


@pytest.fixture
def stored_schema(schema_store):
    """
    Function-scoped fixture that provides access to stored schemas.
    Depends on schema_store to ensure schemas are available.
    """
    return schema_store
