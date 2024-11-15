"""Tool schema definitions.

Contains tool-related schema classes:
- ToolMetadata: Tool description and parameter information
- PropertyAttr: Property attributes for tool parameters
- InputSchema: JSON schema for tool input
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from ..schema.cache import CacheControl


class PropertyAttr(BaseModel):
    """Property attributes for tool parameters."""

    type: Literal["string", "integer", "float", "boolean"] = Field(
        description="Data type of the parameter"
    )
    enum: Optional[List[str]] = Field(description="Possible enum values")
    description: str = Field(description="Description of the parameter")
    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)

    def __hash__(self) -> int:
        return hash(
            (self.type, tuple(self.enum) if \
                self.enum else None, self.description)
        )


class InputSchema(BaseModel):
    
    """JSON schema for tool input."""

    type: Literal["object", "dict"] = Field(description="Type of input schema")
    properties: Optional[Dict[str, PropertyAttr]] = Field(
        description="Tool input properties schema"
    )
    required: Optional[List[str]] = Field(description="Required fields")
    def __hash__(self) -> int:
        return hash(
            (
                self.type,
                tuple(sorted(self.properties.items())) \
                    if self.properties else None,
                tuple(sorted(self.required)) \
                    if self.required else None,
            )
        )


class ToolMetadata(BaseModel):
    """Tool metadata including description and parameters."""

    type: Optional[
        Literal["custom", \
            "computer_20241022", \
            "text_editor_20241022", \
            "bash_20241022"
        ]
    ] = Field(description="Type of tool")
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    input_schema: Optional[InputSchema] = Field(description="Tool input schema")
    cache_control: Optional[CacheControl] = Field(
        description="Cache control for the tool"
    )
    def __hash__(self) -> int:
        return hash(
                self.type,
                self.name,
                self.description,
                self.input_schema,
                self.cache_control,
            )
