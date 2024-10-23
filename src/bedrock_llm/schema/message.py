from pydantic import BaseModel
from typing import List, Literal, Dict

from src.bedrock_llm.schema.cache import CacheControl


class UserMetadat(BaseModel):
    """
    An object describing metadata about the request.
    
    Attributes:
        user_id (str | None): An external identifier for the user who is associated with the request. 
                    This should be a uuid, hash value, or other opaque identifier. 
                    Anthropic may use this id to help detect abuse. 
                    Do not include any identifying information such as name, email address, or phone number..

    Example:
        >>> user_metadata = UserMetadata(
        ...     user_id="XXXXXXXX"
        ... )
    """
    user_id: str | None = None


class Image(BaseModel):
    """ An image represented as base64 data.

    Attributes:
        type (Literal["base_64"]): The type of image, must be "base_64".
        media_type (Literal["image/png", "image/jpeg", "image/gif", "image/webp"]): The media type of the image.
        data (str): The base64-encoded image data.

    Example:
        >>> image = Image(
        ...     type="base_64",
        ...     media_type="image/png",
        ...     data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
        ... )

    See more: https://docs.anthropic.com/en/docs/build-with-claude/vision#prompt-examples
    """
    type: Literal["base_64"]
    media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    data: str


class SystemBlock(BaseModel):
    """
    System prompt.

    A system prompt is a way of providing context and instructions to Claude, such as specifying a particular goal or role.
    
    Attributes:
        cache_control (CacheControl | None): Controls caching behavior for the system prompt.
            Defaults to None.
        type (Literal["text"]): The type of block, must be "text".
        text (str): The actual system prompt text content.

    Example:
        >>> system_block = SystemBlock(
        ...     type="text",
        ...     text="You are an expert in Python programming.",
        ...     cache_control=CacheControl("ephemeral")
        ... )
    
    See more: https://docs.anthropic.com/en/docs/system-prompts
    """
    cache_control: CacheControl | None = None
    type: Literal["text"]
    text: str
   
    
class TextBlock(BaseModel):
    """ Text block.

    Attributes:
        cache_control (CacheControl | None): Controls caching behavior for the text block.
            Defaults to None.
        type (Literal["text"]): The type of block, must be "text".
        text (str): The actual text content.

    Example:
        >>> text_block = TextBlock(
        ...     type="text",
        ...     text="Hello, world!",
        ...     cache_control=CacheControl("ephemeral")
        ... )
    """
    # cache_control: CacheControl | None = None
    type: Literal["text"]
    text: str


class ImageBlock(BaseModel):
    """ Image block.

    Attributes:
        cache_control (CacheControl | None): Controls caching behavior for the image block.
            Defaults to None.
        type (Literal["image"]): The type of block, must be "image".
        source (Image): The source of the image.

    Example:
        >>> image_block = ImageBlock(
        ...     cache_control=CacheControl("ephemeral")
        ...     type="image",
        ...     source=Image(
        ...         type="base_64",
        ...         media_type="image/png",
        ...         data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
        ...     )
        ... )
        
    See more: https://docs.anthropic.com/en/docs/build-with-claude/vision#prompt-examples
    """
    # cache_control: CacheControl | None = None
    type: Literal["image"]
    source: Image


class ToolUseBlock(BaseModel):
    """Tool use block.

    Attributes:
        cache_control (CacheControl | None): Controls caching behavior for the tool use block.
            Defaults to None.
        type (Literal["tool_use"]): The type of block, must be "tool_use".
        id (str): The ID of the tool use.
        name (str): The name of the tool use.
        input (Dict): The input to the tool use.

    Example:
        >>> tool_use_block = ToolUseBlock(
        ...     cache_control: CacheControl("ephemeral")
        ...     type="tool_use",
        ...     id="tool_use_id",
        ...     name="tool_name",
        ...     input={"key": "value"}
        ... )
    """
    # cache_control: CacheControl | None = None
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict


class ToolResultBlock(BaseModel):
    """Tool result block.

    Attributes:
        cache_control (CacheControl | None): Controls caching behavior for the tool result block.
            Defaults to None.
        type (Literal["tool_result"]): The type of block, must be "tool_result".
        tool_use_id (str): The ID of the tool use associated with this result.
        is_error (bool): Whether the tool call resulted in an error.
        content (str): The content of the tool result.

    Example:
        >>> tool_result_block = ToolResultBlock(
        ...     cache_control: CacheControl("ephemeral")
        ...     type="tool_result",
        ...     tool_use_id="tool_use_id",
        ...     is_error=False,
        ...     content="The capital of France is Paris."
        ... )
    """
    # cache_control: CacheControl | None = None
    type: Literal["tool_result"]
    tool_use_id: str
    is_error: bool
    content: str


class MessageBlock(BaseModel):
    """Input messages.

    Our models are trained to operate on alternating user and assistant conversational turns. 
    When creating a new Message, you specify the prior conversational turns with the messages parameter, 
    and the model then generates the next Message in the conversation. 
    Consecutive `user` or `assistant` turns in your request will be combined into a single turn.

    Attributes:
        role (Literal["user", "assistant"]): Whether this is a user prompt or an assistant response.
        content (List[TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock] | str): The content of the message.

    Example:
        >>> message_block = MessageBlock(
        ...     role="user",
        ...     content="What is the capital of France?"
        ... )
        >>> message_block = MessageBlock(
        ...     role="assistant",
        ...     content=[
        ...         TextBlock(
        ...             type="text",
        ...             text="The capital of France is Paris."
        ...         )
        ...     ]
        ... )
        >>> message_block = MessageBlock(
        ...     role="user",
        ...     content=[
        ...         TextBlock(
        ...             text="What is this picture represent?"
        ...         ),
        ...         ImageBlock(
        ...             source=Image(
        ...                 type="base_64",
        ...                 media_type="image/png",
        ...                 data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        ...             )
        ...         )
        ...     ]
        ... )
    """
    role: Literal["user", "assistant"]
    content: List[TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock] | str    
