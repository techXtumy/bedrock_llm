from pydantic import BaseModel
from typing import List, Literal, Dict, Any, Optional, Union

from ..schema.cache import CacheControl


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
    user_id: Optional[str] = None


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
    cache_control: Optional[CacheControl] = None
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
    cache_control: Optional[CacheControl] = None
    type: Literal["text"]
    text: str
    
    def model_dump(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_unset', True)
        return super().model_dump(**kwargs)


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
    cache_control: Optional[CacheControl] = None
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
    cache_control: Optional[CacheControl] = None
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict
    
    def model_dump(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_unset', True)
        return super().model_dump(**kwargs)


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
    cache_control: Optional[CacheControl] = None
    type: Literal["tool_result"]
    tool_use_id: str
    is_error: bool
    content: Union[TextBlock, ImageBlock, str]
    
    def model_dump(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_unset', True)
        return super().model_dump(**kwargs)


class ToolCallBlock(BaseModel):
    """
    Tool call block for the Jamba Model assistant role.

    Attributes:
        id (str): The ID of the tool call.
        type (str): This param is only valid for **Jamba Model (AI21)**
        function (Dict[str | Any]): The function call to make.

    Example:
        >>> tool_call_block = ToolCallBlock(
        ...     id="tool_call_id",
        ...     type="function",
        ...     function={"name": "tool_name", "arguments": {"key": "value"}}
        ... )
    """
    id: str
    type: Optional[str]
    function: Dict[str, Any]


class MessageBlock(BaseModel):
    """Input messages. **Only for Chat Model**

    Our models are trained to operate on alternating user and assistant conversational turns. 
    When creating a new Message, you specify the prior conversational turns with the messages parameter, 
    and the model then generates the next Message in the conversation. 
    Consecutive `user` or `assistant` turns in your request will be combined into a single turn. 
    The `tool` and `system` role turn is only for **Jamba Model (AI21)** and **Mistral Large Model**.

    Attributes:
        role (Literal["user", "assistant", "tool", "system"]): Whether this is a user prompt or an assistant response. The `tool` and `system` role is explicitly only for **Jamba Model (AI21)** and **Mistral Large Model**
        content (List[TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock] | str): The content of the message.
        tool_calls (List[ToolCallBlock] | None):If the assistant called a tool as requested and successfully returned a result, include the tool call results here to enable context for future responses by the model. Explicitly only for **Jamba Model (AI21)** and **Mistral Large Model**. For Anthropic model, use `ToolUseBlock` inside the `content` instead.
        tool_calls_id (str | None): The ID of the tool call after running the tool. This will act as a tool result ID for context. Explicitly only for **Jamba Model (AI21)**. For Anthropic model, use `ToolResultBlock` inside the `content` instead.

    Note:
        - For Anthropic model, use `ToolUseBlock` and `ToolResultBlock` inside the `content` instead of `tool_calls` and `tool_calls_id`.
        
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
        
        See more for Jamaba Model: https://docs.ai21.com/reference/jamba-15-api-ref
    """
    role: Literal["user", "assistant", "tool", "system"]
    content: Union[List[Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock, List]], str]
    tool_calls: Optional[List[ToolCallBlock]] = None
    tool_calls_id: Optional[str] = None
    
    # Override model_dump to automatically exclude None and unset fields
    def model_dump(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_unset', True)
        return super().model_dump(**kwargs)

    # Override model_dump_json similarly
    def model_dump_json(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_unset', True)
        return super().model_dump_json(**kwargs)


class DocumentBlock(BaseModel):
    """
    Document block.

    Attributes:
        content (str): The content of the document.
        metadata (List[Dict[Literal["key", "value"], str]]): The metadata of the document.

    Example:
        >>> document_block = DocumentBlock(
        ...     content="This is a document.",
        ...     metadata=[{"key": "source", "value": "example.com"}]
        ... )
    """
    content: str
    metadata: List[Dict[Literal["key", "value"], str]]