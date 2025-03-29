"""Core type definitions."""

from typing import Any, Dict, List, Optional, TypedDict, Union
from enum import Enum, auto


class AgentLoop(Enum):
    """Enumeration of available loop types."""

    ANTHROPIC = auto()  # Anthropic implementation
    OMNI = auto()  # OmniLoop implementation
    OPENAI = auto()  # OpenAI implementation
    # Add more loop types as needed


class AgentResponse(TypedDict, total=False):
    """Agent response format."""

    id: str
    object: str
    created_at: int
    status: str
    error: Optional[str]
    incomplete_details: Optional[Any]
    instructions: Optional[Any]
    max_output_tokens: Optional[int]
    model: str
    output: List[Dict[str, Any]]
    parallel_tool_calls: bool
    previous_response_id: Optional[str]
    reasoning: Dict[str, str]
    store: bool
    temperature: float
    text: Dict[str, Dict[str, str]]
    tool_choice: str
    tools: List[Dict[str, Union[str, int]]]
    top_p: float
    truncation: str
    usage: Dict[str, Any]
    user: Optional[str]
    metadata: Dict[str, Any]
    response: Dict[str, List[Dict[str, Any]]]
    # Additional fields for error responses
    role: str
    content: Union[str, List[Dict[str, Any]]]
