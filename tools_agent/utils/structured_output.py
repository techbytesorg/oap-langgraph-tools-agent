from pydantic import create_model, Field as PydanticField
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

# Map JSON Schema types to Python types (simplified)
TYPE_MAP = {
    "string": (str, ...),
    "integer": (int, ...),
    "number": (float, ...),
    "boolean": (bool, ...),
}

def create_pydantic_model_from_json_schema(schema_json: Dict[str, Any], model_name: str):
    """
    Create a Pydantic model from a JSON schema dictionary.

    Args:
        schema_json: The JSON schema dictionary
        model_name: Name for the generated Pydantic model

    Returns:
        Dynamically created Pydantic model class
    """
    fields = {}
    properties = schema_json.get("properties", {})
    required_fields = schema_json.get("required", [])

    for field_name, field_info in properties.items():
        py_type, default = TYPE_MAP.get(field_info.get("type", "string"), (str, ...))
        default_value = ... if field_name in required_fields else None
        description = field_info.get("description", "")
        fields[field_name] = (py_type, PydanticField(default_value, description=description))

    model = create_model(model_name, **fields)
    return model

async def load_schema_model(schema_name: str, user_id: str = None):
    """
    Load a schema model from the LangGraph memory store using get_store().

    Args:
        schema_name: Name of the schema to load
        user_id: User ID for namespace (optional, will try both user-specific and global)

    Returns:
        Pydantic model class for the requested schema

    Raises:
        ValueError: If schema is not found in memory store
    """
    from langgraph.config import get_store

    logger.debug(f"Loading schema model: {schema_name} for user: {user_id}")

    # Use LangGraph's get_store() to access the managed store
    store = get_store()
    logger.debug(f"Got store from get_store(): {type(store)}")

    # Try user-specific namespace first if user_id provided
    if user_id:
        try:
            namespace = (user_id, "schemas")
            logger.debug(f"Trying user-specific namespace: {namespace}")
            result = await store.aget(namespace, schema_name)
            if result:
                logger.debug(f"Found schema in user-specific namespace")
                schema_json = result.value
                return create_pydantic_model_from_json_schema(schema_json, schema_name)
            else:
                logger.debug(f"Schema not found in user-specific namespace")
        except Exception as e:
            logger.debug(f"Error accessing user-specific namespace: {e}")

    # Fallback to global namespace for backwards compatibility
    try:
        global_namespace = ("schemas",)
        logger.debug(f"Trying global namespace: {global_namespace}")
        result = await store.aget(global_namespace, schema_name)
        if result:
            schema_json = result.value
            logger.debug(f"Found schema in global namespace")
            return create_pydantic_model_from_json_schema(schema_json, schema_name)
    except Exception as e:
        logger.debug(f"Error accessing global namespace: {e}")

    logger.warning(f"Schema {schema_name} not found in any namespace")
    raise ValueError(f"Schema {schema_name} not found in memory store")