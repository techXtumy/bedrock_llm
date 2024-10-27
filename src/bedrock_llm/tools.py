# tools.py
import inspect
from typing import Callable, Dict, Any
from src.bedrock_llm.types.enums import DescriptionFormat


# Dictionary to store tool metadata and functions
tool_registry: Dict[str, Dict] = {}


def parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    Parse a docstring into a basic structured description format.
    """
    description_lines = docstring.strip().splitlines()
    description = description_lines[0] if description_lines else "No description provided."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for line in description_lines[1:]:
        line = line.strip()
        if ":" in line and "-" in line:
            param_name, rest = line.split(":", 1)
            param_name = param_name.strip()
            type_info, desc = rest.split("-", 1)
            type_info = type_info.strip()
            desc = desc.strip()
            input_schema["properties"][param_name] = {"type": type_info, "description": desc}
            input_schema["required"].append(param_name)

    return {
        "description": description,
        "input_schema": input_schema
    }


def format_tool_metadata(name: str, func: Callable, docstring: str, format_type: DescriptionFormat) -> Dict[str, Any]:
    """
    Format the tool metadata based on the specified format type.
    """
    parsed_doc = parse_docstring(docstring)
    
    if format_type == DescriptionFormat.CLAUDE:
        return {
            "name": name,
            "description": parsed_doc["description"],
            "input_schema": parsed_doc["input_schema"]
        }
    elif format_type == DescriptionFormat.NORMAL:
        return {
            "name": name,
            "description": parsed_doc["description"],
            "parameters": parsed_doc["input_schema"]
        }
    elif format_type == DescriptionFormat.REACT:
        return {
            "tool_id": name,
            "summary": parsed_doc["description"],
            "schema": parsed_doc["input_schema"]
        }
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def tool(format: str):
    """
    Decorator to register a function as an LLM tool with specified format metadata.
    """
    def decorator(func: Callable) -> Callable:
        metadata = format_tool_metadata(func.__name__, func, func.__doc__ or "No description provided.", format)
        tool_registry[func.__name__] = {
            "func": func,
            "async": inspect.iscoroutinefunction(func),
            "metadata": metadata
        }
        return func
    return decorator
