from pydantic import BaseModel
from typing import Literal, Dict, List, Optional


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
    description: str


class InputSchema(BaseModel):
    """
    JSON schema for this tool's input.

    This defines the shape of the input that your tool accepts and that the model will produce.
    
    Attributes:
        type (Literal["object"]): The type of input schema.
        properties (Dict | None): A dictionary of property names and their corresponding schemas.
        required (List[str] | None): A list of property names that are required.

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
    type: Literal["object"]
    properties: Dict | None = None
    required: List[str] | None = None
    

class ToolMetadata(BaseModel):
    """
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
    # type: Literal["custom"] | None = None
    name: str
    description: Optional[str]
    input_schema: InputSchema
    # cache_control: CacheControl | None = None