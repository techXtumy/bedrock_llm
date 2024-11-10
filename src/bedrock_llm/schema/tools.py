from pydantic import BaseModel
from typing import Literal, Dict, List, Optional
from ..schema.cache import CacheControl


class PropertyAttr(BaseModel):
    """
    Attribute of the property that you defined for the params

    Attributes:
        type (Literal["string", "integer", "float", "boolean"]): The type of the property.
        description (str): The description of the property.

    Example:
        >>> name_attr = PropertyAttr(
        ...     type="string",
        ...     description="The name of the person"
        ... )
    """
    type: Literal["string", "integer", "float", "boolean"]
    enum: Optional[List[str]] = None
    description: str
    
    def model_dump(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_unset', True)
        return super().model_dump(**kwargs)


class InputSchema(BaseModel):
    """
    JSON schema for this tool's input.

    This defines the shape of the input that your tool accepts and that the model will produce.
    
    Attributes:
        type (Literal["object"]): The type of input schema.
        properties (Dict[str, PropertyAttr] | None): A dictionary of property names and their corresponding schemas. If nothing, put empty Dict
        required (List[str] | None): A list of property names that are required. If nothing, put empty List

    Example:
        >>> input_schema = InputSchema(
        ...     type="object",
        ...     properties={
        ...         "name": PropertyAttr(type="string", description="The name of the person"),
        ...         "age": PropertyAttr(type="integer", description="The age of the person")
        ...     },
        ...     required=["name"]
        ... )
    """
    type: Literal["object", "dict"]
    properties: Optional[Dict[str, PropertyAttr]]
    required: Optional[List[str]]
   
    
class ToolMetadata(BaseModel):
    """
    Metadata for a Claude tool.

    Attributes:
        type (Literal["custom", "computer_20241022", "text_editor_20241022", "bash_20241022"] | None): The type of tool, only valid for 3.5 new Sonnet
        name (str): The name of the tool.
        description (str): The description of the tool.
        parameters (InputSchema | None): The parameters for the tool.
        cache_control (CacheControl | None): The cache control for the tool.

    Example:
        >>> tool_metadata = ToolMetadata(
        ...     name="PersonInfo",
        ...     description="Get information about a person",
        ...     input_schema=InputSchema(
        ...         type="object",
        ...         properties={
        ...             "name": PropertyAttr(type="string", description="The name of the person"),
        ...             "age": PropertyAttr(type="integer", description="The age of the person")
        ...         },
        ...         required=["name"]
        ...     )
        ... )
    
    ## Best practices for tool definitions
    To get the best performance out of Claude when using tools, follow these guidelines:

    1.  Provide extremely detailed descriptions. 
        This is by far the most important factor in tool performance. 
        Your descriptions should explain every detail about the tool, including:
        -   What the tool does
        -   When it should be used (and when it shouldn’t)
        -   What each parameter means and how it affects the tool’s behavior
        -   Any important caveats or limitations, such as what information the tool does not return if the tool name is unclear. 
            The more context you can give Claude about your tools, the better it will be at deciding when and how to use them. 
            Aim for at least 3-4 sentences per tool description, more if the tool is complex.
    2.  Prioritize descriptions over examples. 
        While you can include examples of how to use a tool in its description or in the accompanying prompt, 
        this is less important than having a clear and comprehensive explanation of the tool’s purpose and parameters. 
        Only add examples after you’ve fully fleshed out the description.
    """
    type: Optional[Literal["custom", "computer_20241022", "text_editor_20241022", "bash_20241022"]] = None
    name: str
    description: str
    input_schema: Optional[InputSchema]
    cache_control: Optional[CacheControl] = None
    
    def model_dump(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_unset', True)
        return super().model_dump(**kwargs)
