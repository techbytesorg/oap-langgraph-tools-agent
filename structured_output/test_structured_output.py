"""
Test script for structured output feature using remote agent authentication.
This script follows the authentication pattern from the external access POC.
"""

import os
import asyncio
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv
from langgraph.pregel.remote import RemoteGraph

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
USER_EMAIL = os.environ.get("USER_EMAIL")
USER_PASSWORD = os.environ.get("USER_PASSWORD")

# Agent configuration
AGENT_URL = "http://localhost:2024"  # Local development server
ASSISTANT_ID = "agent"  # Default assistant ID for local development

# Test recipe for allergy analysis
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

def authenticate_supabase():
    """Authenticate with Supabase and return access token and user info"""
    print(f"SUPABASE_URL: {SUPABASE_URL}")
    print(f"SUPABASE_KEY: {'***' if SUPABASE_KEY else 'None'}")
    print(f"USER_EMAIL: {USER_EMAIL}")
    print(f"USER_PASSWORD: {'***' if USER_PASSWORD else 'None'}")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

    if not USER_EMAIL or not USER_PASSWORD:
        raise ValueError("user_email and user_password must be set in .env file")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    response = supabase.auth.sign_in_with_password({
        "email": USER_EMAIL,
        "password": USER_PASSWORD,
    })

    # Get user info to extract user_id (same as schema_loader.py)
    access_token = response.session.access_token
    user_response = supabase.auth.get_user(access_token)
    user_id = user_response.user.id

    print(f"Authenticated user_id: {user_id}")

    return access_token, user_id

async def test_structured_output():
    """Test the structured output feature with different schemas"""

    # Authenticate and get access token and user_id
    access_token, user_id = authenticate_supabase()

    # Set up headers for authentication
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    # Create remote graph connection
    remote_graph = RemoteGraph(
        ASSISTANT_ID,
        url=AGENT_URL,
        headers=headers
    )

    # Test cases with different schemas
    test_cases = [
        {
            "name": "No Schema",
            "config": {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "x-supabase-access-token": access_token,
                }
            },
            "prompt": "what is the value of 2 + 3?",
        },
        {
            "name": "OutputSchemaA",
            "config": {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "x-supabase-access-token": access_token,
                    "user_id": user_id,
                    "OutputSchemaName": "OutputSchemaA"
                }
            },
            "prompt": "what is the value of 2 + 3?",
        },
        {
            "name": "OutputSchemaB",
            "config": {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "x-supabase-access-token": access_token,
                    "user_id": user_id,
                    "OutputSchemaName": "OutputSchemaB"
                }
            },
            "prompt": "what is the value of 2 + 3?",
        },
        {
            "name": "AllergyAnalysisResponse",
            "config": {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "x-supabase-access-token": access_token,
                    "user_id": user_id,
                    "OutputSchemaName": "AllergyAnalysisResponse"
                }
            },
            "prompt": f"Analyze this recipe for food allergens:\n\n{FRESHPREP_TEST_RECIPE}",
        }
    ]

    # Run test cases
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_case['name']}")
        print(f"{'='*50}")

        try:
            # Invoke the graph
            result = await remote_graph.ainvoke({
                "messages": [{"role": "user", "content": test_case["prompt"]}]
            }, config=test_case["config"])

            # Check if structured output is expected
            schema_name = test_case["config"]["configurable"].get("OutputSchemaName")
            if schema_name:
                structured_response = result.get("structured_response")
                if structured_response:
                    print(f"✅ Structured output ({schema_name}):")
                    print(structured_response)
                else:
                    print("❌ No structured response found, showing text output:")
                    print(result["messages"][-1]["content"])
            else:
                print("Text output:")
                print(result["messages"][-1]["content"])

        except Exception as e:
            print(f"Error in test case '{test_case['name']}': {e}")

async def main():
    """Main function to run structured output tests"""
    print("Starting structured output tests...")
    print(f"Agent URL: {AGENT_URL}")
    print(f"Assistant ID: {ASSISTANT_ID}")

    try:
        await test_structured_output()
        print(f"\n{'='*50}")
        print("All tests completed!")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(main())