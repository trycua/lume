"""Base agent loop implementation."""

import logging
import importlib.util
from typing import Dict, Optional, Type, TYPE_CHECKING, Any, cast, Callable, Awaitable

from computer import Computer
from .types import AgentLoop
from .base import BaseLoop

# For type checking only
if TYPE_CHECKING:
    from ..providers.omni.types import LLMProvider

logger = logging.getLogger(__name__)


class LoopFactory:
    """Factory class for creating agent loops."""

    # Registry to store loop implementations
    _loop_registry: Dict[AgentLoop, Type[BaseLoop]] = {}

    @classmethod
    def create_loop(
        cls,
        loop_type: AgentLoop,
        api_key: str,
        model_name: str,
        computer: Computer,
        provider: Any = None,
        save_trajectory: bool = True,
        trajectory_dir: str = "trajectories",
        only_n_most_recent_images: Optional[int] = None,
        acknowledge_safety_check_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
    ) -> BaseLoop:
        """Create and return an appropriate loop instance based on type."""
        if loop_type == AgentLoop.ANTHROPIC:
            # Lazy import AnthropicLoop only when needed
            try:
                from ..providers.anthropic.loop import AnthropicLoop
            except ImportError:
                raise ImportError(
                    "The 'anthropic' provider is not installed. "
                    "Install it with 'pip install cua-agent[anthropic]'"
                )

            return AnthropicLoop(
                api_key=api_key,
                model=model_name,
                computer=computer,
                save_trajectory=save_trajectory,
                base_dir=trajectory_dir,
                only_n_most_recent_images=only_n_most_recent_images,
            )
        elif loop_type == AgentLoop.OPENAI:
            # Lazy import OpenAILoop only when needed
            try:
                from ..providers.openai.loop import OpenAILoop
            except ImportError:
                raise ImportError(
                    "The 'openai' provider is not installed. "
                    "Install it with 'pip install cua-agent[openai]'"
                )

            return OpenAILoop(
                api_key=api_key,
                model=model_name,
                computer=computer,
                save_trajectory=save_trajectory,
                base_dir=trajectory_dir,
                only_n_most_recent_images=only_n_most_recent_images,
                acknowledge_safety_check_callback=acknowledge_safety_check_callback,
            )
        elif loop_type == AgentLoop.OMNI:
            # Lazy import OmniLoop and related classes only when needed
            try:
                from ..providers.omni.loop import OmniLoop
                from ..providers.omni.parser import OmniParser
                from ..providers.omni.types import LLMProvider
            except ImportError:
                raise ImportError(
                    "The 'omni' provider is not installed. "
                    "Install it with 'pip install cua-agent[all]'"
                )

            if provider is None:
                raise ValueError("Provider is required for OMNI loop type")

            # We know provider is the correct type at this point, so cast it
            provider_instance = cast(LLMProvider, provider)

            return OmniLoop(
                provider=provider_instance,
                api_key=api_key,
                model=model_name,
                computer=computer,
                save_trajectory=save_trajectory,
                base_dir=trajectory_dir,
                only_n_most_recent_images=only_n_most_recent_images,
                parser=OmniParser(),
            )
        else:
            raise ValueError(f"Unsupported loop type: {loop_type}")
