"""
Test suite for structured output feature.

Tests all 8 schema types covering OpenAI JSON schema spec plus edge cases.
Uses session-scoped fixtures for schema storage with automatic cleanup.
"""

import pytest
import uuid
import json
from typing import Dict, Any
from pydantic import ValidationError


# Test data constants
FRESHPREP_TEST_RECIPE = """
Pulled Chicken Fajitas
with Chipotle-Pineapple Sauce & Cheddar
Poultry - Local, Free Run, Antibiotic Free
Milk
Gluten
Seafood
Serves
2
Difficulty
Moderate
Time
30 Min
Ingredients
Onion
Chipotle Pepper in Adobo Sauce
Chicken Breast
Worcestershire Sauce
Aged White Cheddar Cheese
Flour Tortillas 6"
Cilantro
Assorted Mini Bell Peppers
Coconut Sugar
Green Cabbage
Pineapple-Lime Juice
Spice Blend
Lime Crema
* Pineapple-Lime Juice: Pineapple Juice , Fresh Lime Juice
* Spice Blend: Smoked Paprika , Cumin
* Lime Crema: Sour Cream , Mayonnaise , Fresh Lime Juice
"""

SALMON_QUINOA_RECIPE = """
Grilled Salmon with Quinoa and Roasted Vegetables
- Grilled Salmon (6oz Atlantic salmon fillet)
- Quinoa (1 cup cooked)
- Roasted Broccoli (1 cup florets)
- Olive Oil (2 tbsp for cooking)
- Lemon (1 wedge)
- Garlic and Herbs seasoning
Serves: 2 people
Estimated prep time: 30 minutes
"""


# ============================================================================
# Helper Functions
# ============================================================================


def validate_structured_response(result: dict, schema_name: str) -> Dict[str, Any]:
    """
    Assert that structured_response exists and return it.

    Args:
        result: The result dictionary from agent invocation
        schema_name: Expected schema name for logging

    Returns:
        Dict containing the structured response

    Raises:
        AssertionError: If structured_response is missing or invalid
    """
    assert result is not None, "Result should not be None"
    assert (
        "structured_response" in result
    ), f"Result should contain 'structured_response' for schema {schema_name}"

    structured_response = result["structured_response"]
    assert (
        structured_response is not None
    ), f"Structured response should not be None for schema {schema_name}"

    # Parse if it's a string
    if isinstance(structured_response, str):
        try:
            structured_response = json.loads(structured_response)
        except json.JSONDecodeError as e:
            pytest.fail(f"Structured response is not valid JSON: {e}")

    return structured_response


def assert_schema_conforms(data: Dict[str, Any], schema_class) -> None:
    """
    Validate that data conforms to the Pydantic schema.

    Args:
        data: The data to validate
        schema_class: The Pydantic model class

    Raises:
        AssertionError: If data doesn't conform to schema
    """
    try:
        # Attempt to parse with Pydantic
        instance = schema_class(**data)
        assert instance is not None, "Pydantic validation should create an instance"
    except ValidationError as e:
        pytest.fail(f"Data does not conform to schema {schema_class.__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error validating schema {schema_class.__name__}: {e}")


# ============================================================================
# Success Cases: All 8 Schema Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_nutrition_analysis(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test RecipeNutritionAnalysis - number constraints (ge, le, multipleOf)"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze the nutritional content of this recipe:\n\n{SALMON_QUINOA_RECIPE}\n\nProvide calories per serving (0-10000), protein/fat/carbs in grams, number of servings (1-20), and prep time in minutes.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "RecipeNutritionAnalysis",
            }
        },
    )

    structured_response = validate_structured_response(
        result, "RecipeNutritionAnalysis"
    )
    assert_schema_conforms(
        structured_response, schema_definitions["RecipeNutritionAnalysis"]
    )

    # Validate specific constraints
    assert (
        0 <= structured_response["calories"] <= 10000
    ), "Calories should be within range"
    assert structured_response["protein_grams"] >= 0, "Protein should be non-negative"
    assert 1 <= structured_response["servings"] <= 20, "Servings should be within range"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_ingredient_classification(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test IngredientClassification - enums and array constraints"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Classify this ingredient: Organic Almond Butter. Provide the ingredient name, categorize it (protein/grain/vegetable/fruit/dairy/condiment/spice), identify which dietary restrictions it fits (vegan/vegetarian/gluten-free/dairy-free/nut-free/kosher/halal), and list any allergen tags (max 5).",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "IngredientClassification",
            }
        },
    )

    structured_response = validate_structured_response(
        result, "IngredientClassification"
    )
    assert_schema_conforms(
        structured_response, schema_definitions["IngredientClassification"]
    )

    # Validate enum constraints
    valid_categories = [
        "protein",
        "grain",
        "vegetable",
        "fruit",
        "dairy",
        "condiment",
        "spice",
    ]
    assert (
        structured_response["category"] in valid_categories
    ), f"Category must be one of {valid_categories}"
    assert (
        len(structured_response["allergen_tags"]) <= 5
    ), "Allergen tags should not exceed 5"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_food_safety_report(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test FoodSafetyReport - nested objects with $ref"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a food safety inspection report for a commercial kitchen inspected on 2025-03-15. Inspection ID: INS-2025-0315. Found violations: 1) Improper food storage temperature in walk-in cooler (critical severity), corrective action: adjust thermostat and monitor; 2) Missing handwashing signage near prep area (minor severity), corrective action: install signage; 3) Expired ingredients in dry storage (major severity), corrective action: dispose and update inventory system. Calculate an overall safety score (0-100).",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "FoodSafetyReport",
            }
        },
    )

    structured_response = validate_structured_response(result, "FoodSafetyReport")
    assert_schema_conforms(structured_response, schema_definitions["FoodSafetyReport"])

    # Validate nested structure
    assert "violations" in structured_response, "Should have violations array"
    assert (
        len(structured_response["violations"]) > 0
    ), "Should have at least one violation"
    assert 0 <= structured_response["overall_score"] <= 100, "Score should be 0-100"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_menu_planning(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test MenuPlanning - complex nested objects"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a 3-day meal plan for a small restaurant starting Monday, April 1, 2025. Each day needs breakfast, lunch, and dinner. For each meal, provide dish name, main ingredients (list), and estimated cost per serving. Use a 'Spring Fresh' theme focusing on seasonal vegetables and local proteins.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "MenuPlanning",
            }
        },
    )

    structured_response = validate_structured_response(result, "MenuPlanning")
    assert_schema_conforms(structured_response, schema_definitions["MenuPlanning"])

    # Validate nested structure
    assert (
        len(structured_response["daily_menus"]) >= 1
    ), "Should have at least one daily menu"
    for daily_menu in structured_response["daily_menus"]:
        assert "breakfast" in daily_menu, "Each day should have breakfast"
        assert "lunch" in daily_menu, "Each day should have lunch"
        assert "dinner" in daily_menu, "Each day should have dinner"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_recipe_instructions(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test RecipeInstructions - recursive schema structures"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Provide step-by-step instructions for making Homemade Sourdough Bread. Include the recipe name and total time. Break down into main steps, and for complex steps like 'Prepare the dough' or 'Shape and proof', include sub-steps. Each step should have a step number, instruction text, and duration in minutes.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "RecipeInstructions",
            }
        },
    )

    structured_response = validate_structured_response(result, "RecipeInstructions")
    assert_schema_conforms(
        structured_response, schema_definitions["RecipeInstructions"]
    )

    # Validate recursive structure
    assert len(structured_response["steps"]) >= 1, "Should have at least one step"
    assert (
        structured_response["total_time_minutes"] >= 1
    ), "Total time should be at least 1 minute"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_supplier_quote(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test SupplierQuote - union types and pattern validation"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a supplier quote: Supplier name is 'Fresh Farms Co.', contact email john.smith@freshfarms.com, phone number (555) 123-4567, quoting $450.00 for organic produce delivery. Expected delivery date is 2025-04-15. Add optional notes about requiring refrigerated transport.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "SupplierQuote",
            }
        },
    )

    structured_response = validate_structured_response(result, "SupplierQuote")
    assert_schema_conforms(structured_response, schema_definitions["SupplierQuote"])

    # Validate patterns
    assert "@" in structured_response["contact_email"], "Email should contain @"
    assert structured_response["quoted_price"] > 0, "Quoted price should be positive"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_quality_inspection(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test QualityInspection - multiple enums and format validation"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a quality control inspection record: Batch ID BT-2025-0315 for processed tomatoes. Inspector email: qa.inspector@foodco.com. Inspection datetime: 2025-03-15T14:30:00. Assign a quality grade (A/B/C/D/F), set status (pending/approved/rejected/review_required), and list inspection findings such as color consistency, texture, no defects found.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "QualityInspection",
            }
        },
    )

    structured_response = validate_structured_response(result, "QualityInspection")
    assert_schema_conforms(structured_response, schema_definitions["QualityInspection"])

    # Validate enums
    valid_grades = ["A", "B", "C", "D", "F"]
    valid_statuses = ["pending", "approved", "rejected", "review_required"]
    assert (
        structured_response["quality_grade"] in valid_grades
    ), f"Grade must be one of {valid_grades}"
    assert (
        structured_response["status"] in valid_statuses
    ), f"Status must be one of {valid_statuses}"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_allergy_analysis(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Test AllergyAnalysisResponse - array of nested objects"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this recipe for food allergens:\n\n{FRESHPREP_TEST_RECIPE}\n\nIdentify high-risk allergens with reasons, and provide recommendations for people with allergies.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "AllergyAnalysisResponse",
            }
        },
    )

    structured_response = validate_structured_response(
        result, "AllergyAnalysisResponse"
    )
    assert_schema_conforms(
        structured_response, schema_definitions["AllergyAnalysisResponse"]
    )

    # Validate array structure
    assert "high_risk_allergen" in structured_response, "Should have allergen list"
    assert "recommendation" in structured_response, "Should have recommendation"
    if len(structured_response["high_risk_allergen"]) > 0:
        first_allergen = structured_response["high_risk_allergen"][0]
        assert (
            "allergen" in first_allergen
        ), "Each allergen entry should have 'allergen' field"
        assert (
            "reason" in first_allergen
        ), "Each allergen entry should have 'reason' field"


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.structured_output
async def test_schema_not_found(remote_graph, access_token, user_id, stored_schema):
    """Edge: Request with non-existent schema name"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Tell me about food safety"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "NonExistentSchema",
            }
        },
    )

    # Should either return text response or handle gracefully
    assert result is not None, "Should return a result"
    # The agent should handle missing schema gracefully


@pytest.mark.asyncio
@pytest.mark.structured_output
async def test_no_schema_provided(remote_graph, access_token, user_id):
    """Edge: Normal text output when no schema specified"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is food safety?"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
            }
        },
    )

    # Should return normal text response
    assert result is not None, "Result should not be None"
    assert "messages" in result, "Result should contain messages"
    assert len(result["messages"]) > 0, "Should have messages"
    # No structured_response expected
    if "structured_response" in result:
        assert (
            result["structured_response"] is None
        ), "Structured response should be None without schema"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_schema_with_empty_optional_fields(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Edge: Schema with optional/nullable fields"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a supplier quote: Supplier name is 'Quick Foods', contact email info@quickfoods.com, phone number (555) 999-8888, quoting $100.00 for delivery on 2025-05-01. Do NOT include any notes.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "SupplierQuote",
            }
        },
    )

    structured_response = validate_structured_response(result, "SupplierQuote")
    assert_schema_conforms(structured_response, schema_definitions["SupplierQuote"])

    # The optional 'notes' field may be None or absent
    if "notes" in structured_response:
        # notes is optional, can be None or a string
        assert structured_response["notes"] is None or isinstance(
            structured_response["notes"], str
        )


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_multiple_schemas_same_thread(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Edge: Switch schemas within same conversation thread"""
    thread_id = str(uuid.uuid4())

    # First request with RecipeNutritionAnalysis
    result1 = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze nutrition of:\n{SALMON_QUINOA_RECIPE}",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "RecipeNutritionAnalysis",
            }
        },
    )

    structured_response1 = validate_structured_response(
        result1, "RecipeNutritionAnalysis"
    )
    assert_schema_conforms(
        structured_response1, schema_definitions["RecipeNutritionAnalysis"]
    )

    # Second request in same thread with different schema
    result2 = await remote_graph.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "Classify ingredient: Organic Spinach"}
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "IngredientClassification",
            }
        },
    )

    structured_response2 = validate_structured_response(
        result2, "IngredientClassification"
    )
    assert_schema_conforms(
        structured_response2, schema_definitions["IngredientClassification"]
    )


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_schema_with_special_characters(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Edge: Schema handling UTF-8/special characters in output"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Classify this ingredient: Jalapeño Peppers (with tilde). Include émojis 🌶️ if needed, and special characters like ñ, é, ü.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "IngredientClassification",
            }
        },
    )

    structured_response = validate_structured_response(
        result, "IngredientClassification"
    )
    assert_schema_conforms(
        structured_response, schema_definitions["IngredientClassification"]
    )

    # Check that ingredient name is captured
    assert (
        len(structured_response["ingredient_name"]) > 0
    ), "Ingredient name should not be empty"


# ============================================================================
# Failure Recovery
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.structured_output
async def test_invalid_schema_format(
    remote_graph, access_token, user_id, stored_schema
):
    """Failure: Malformed schema name"""
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Tell me about food"}]},
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "!!!Invalid@Schema#Name!!!",
            }
        },
    )

    # Should handle gracefully
    assert result is not None, "Should return a result even with invalid schema name"


@pytest.mark.asyncio
@pytest.mark.structured_output
@pytest.mark.slow_integration
async def test_schema_with_minimal_input(
    remote_graph, access_token, user_id, stored_schema, schema_definitions
):
    """Edge: Provide minimal input and expect agent to fill in reasonable defaults"""
    result = await remote_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze nutrition for: Grilled Chicken Breast",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "x-supabase-access-token": access_token,
                "user_id": user_id,
                "OutputSchemaName": "RecipeNutritionAnalysis",
            }
        },
    )

    structured_response = validate_structured_response(
        result, "RecipeNutritionAnalysis"
    )
    assert_schema_conforms(
        structured_response, schema_definitions["RecipeNutritionAnalysis"]
    )

    # Agent should provide reasonable estimates even with minimal input
    assert structured_response["calories"] > 0, "Should estimate non-zero calories"
    assert structured_response["protein_grams"] > 0, "Chicken should have protein"
