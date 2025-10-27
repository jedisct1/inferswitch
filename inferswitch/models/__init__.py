"""
Pydantic models for request/response validation.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    """Represents a content block in messages."""

    type: str
    text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    id: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = None
    cache_control: Optional[Dict[str, Any]] = None  # Explicit field for cache_control

    class Config:
        extra = "allow"  # Allow extra fields


class Message(BaseModel):
    """Represents a message in a conversation."""

    role: str
    content: Union[str, List[ContentBlock], List[Dict[str, Any]]]

    class Config:
        extra = "allow"  # Allow extra fields

    def model_dump(self, **kwargs):
        # Override to handle content serialization properly
        data = super().model_dump(**kwargs)
        # If content is a list of ContentBlock objects, convert to dicts
        if (
            isinstance(self.content, list)
            and self.content
            and isinstance(self.content[0], ContentBlock)
        ):
            data["content"] = [
                block.model_dump(exclude_none=True, by_alias=True, exclude_unset=True)
                for block in self.content
            ]
        return data


class MessagesRequest(BaseModel):
    """Request model for the messages endpoint."""

    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    service_tier: Optional[str] = None
    thinking: Optional[Dict[str, Any]] = None
    container: Optional[str] = None
    mcp_servers: Optional[List[Dict[str, Any]]] = None


class CountTokensRequest(BaseModel):
    """Request model for the count tokens endpoint."""

    model: str
    messages: List[Message]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    tools: Optional[List[Dict[str, Any]]] = None


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Response model for the messages endpoint."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Dict[str, Any]]
    model: str
    stop_reason: str
    stop_sequence: Optional[str] = None
    usage: Usage


class CountTokensResponse(BaseModel):
    """Response model for the count tokens endpoint."""

    input_tokens: int


__all__ = [
    "ContentBlock",
    "Message",
    "MessagesRequest",
    "MessagesResponse",
    "CountTokensRequest",
    "CountTokensResponse",
    "Usage",
]
