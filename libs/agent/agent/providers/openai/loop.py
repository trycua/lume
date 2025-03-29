"""OpenAI Agent Response API provider implementation."""

import logging
import asyncio
import base64
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable, Awaitable, TYPE_CHECKING

from computer import Computer
from ...core.base import BaseLoop
from ...core.types import AgentResponse
from ...core.messages import StandardMessageManager, ImageRetentionConfig

from .api.client import OpenAIClient
from .api_handler import OpenAIAPIHandler
from .response_handler import OpenAIResponseHandler
from .tools.manager import ToolManager
from .types import LLMProvider, ResponseItemType
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OpenAILoop(BaseLoop):
    """OpenAI-specific implementation of the agent loop.

    This class extends BaseLoop to provide specialized support for OpenAI's Agent Response API
    with computer control capabilities.
    """

    ###########################################
    # INITIALIZATION AND CONFIGURATION
    ###########################################

    def __init__(
        self,
        api_key: str,
        computer: Computer,
        model: str = "computer-use-preview",
        only_n_most_recent_images: Optional[int] = 2,
        base_dir: Optional[str] = "trajectories",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        save_trajectory: bool = True,
        acknowledge_safety_check_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        **kwargs,
    ):
        """Initialize the OpenAI loop.

        Args:
            api_key: OpenAI API key
            model: Model name (ignored, always uses computer-use-preview)
            computer: Computer instance
            only_n_most_recent_images: Maximum number of recent screenshots to include in API requests
            base_dir: Base directory for saving experiment data
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            save_trajectory: Whether to save trajectory data
            acknowledge_safety_check_callback: Optional callback for safety check acknowledgment
            **kwargs: Additional provider-specific arguments
        """
        # Always use computer-use-preview model
        if model != "computer-use-preview":
            logger.info(
                f"Overriding provided model '{model}' with required model 'computer-use-preview'"
            )

        # Initialize base class with core config
        super().__init__(
            computer=computer,
            model="computer-use-preview",  # Always use computer-use-preview
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_dir=base_dir,
            save_trajectory=save_trajectory,
            only_n_most_recent_images=only_n_most_recent_images,
            **kwargs,
        )

        # Initialize message manager
        self.message_manager = StandardMessageManager(
            config=ImageRetentionConfig(num_images_to_keep=only_n_most_recent_images)
        )

        # OpenAI-specific attributes
        self.provider = LLMProvider.OPENAI
        self.client = None
        self.retry_count = 0
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback
        self.queue = asyncio.Queue()  # Initialize queue
        self.last_response_id = None  # Store the last response ID across runs

        # Initialize handlers
        self.api_handler = OpenAIAPIHandler(self)
        self.response_handler = OpenAIResponseHandler(self)

        # Initialize tool manager with callback
        self.tool_manager = ToolManager(
            computer=computer, acknowledge_safety_check_callback=acknowledge_safety_check_callback
        )

    ###########################################
    # CLIENT INITIALIZATION - IMPLEMENTING ABSTRACT METHOD
    ###########################################

    async def initialize_client(self) -> None:
        """Initialize the OpenAI API client and tools.

        Implements abstract method from BaseLoop to set up the OpenAI-specific
        client, tool manager, and message manager.
        """
        try:
            logger.info(f"Initializing OpenAI client with model {self.model}...")

            # Initialize client
            self.client = OpenAIClient(api_key=self.api_key, model=self.model)

            # Initialize tool manager
            await self.tool_manager.initialize()

            logger.info(f"Initialized OpenAI client with model {self.model}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.client = None
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

    ###########################################
    # MAIN LOOP - IMPLEMENTING ABSTRACT METHOD
    ###########################################

    async def run(self, messages: List[Dict[str, Any]]) -> AsyncGenerator[AgentResponse, None]:
        """Run the agent loop with provided messages.

        Args:
            messages: List of message objects in standard format

        Yields:
            Agent response format
        """
        try:
            logger.info("Starting OpenAI loop run")

            # Create queue for response streaming
            queue = asyncio.Queue()

            # Ensure client is initialized
            if self.client is None:
                logger.info("Initializing client...")
                await self.initialize_client()
                if self.client is None:
                    raise RuntimeError("Failed to initialize client")
                logger.info("Client initialized successfully")

            # Start loop in background task
            loop_task = asyncio.create_task(self._run_loop(queue, messages))

            # Process and yield messages as they arrive
            while True:
                try:
                    item = await queue.get()
                    if item is None:  # Stop signal
                        break
                    yield item
                    queue.task_done()
                except Exception as e:
                    logger.error(f"Error processing queue item: {str(e)}")
                    continue

            # Wait for loop to complete
            await loop_task

            # Send completion message
            yield {
                "role": "assistant",
                "content": "Task completed successfully.",
                "metadata": {"title": "✅ Complete"},
            }

        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            yield {
                "role": "assistant",
                "content": f"Error: {str(e)}",
                "metadata": {"title": "❌ Error"},
            }

    ###########################################
    # AGENT LOOP IMPLEMENTATION
    ###########################################

    async def _run_loop(self, queue: asyncio.Queue, messages: List[Dict[str, Any]]) -> None:
        """Run the agent loop with provided messages.

        Args:
            queue: Queue for response streaming
            messages: List of messages in standard format
        """
        try:
            # Use the instance-level last_response_id instead of creating a local variable
            # This way it persists between runs

            # Capture initial screenshot
            try:
                # Take screenshot
                screenshot = await self.computer.interface.screenshot()
                logger.info("Screenshot captured successfully")

                # Convert to base64 if needed
                if isinstance(screenshot, bytes):
                    screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
                else:
                    screenshot_base64 = screenshot

                # Save screenshot if requested
                if self.save_trajectory:
                    # Ensure screenshot_base64 is a string
                    if not isinstance(screenshot_base64, str):
                        logger.warning(
                            "Converting non-string screenshot_base64 to string for _save_screenshot"
                        )
                        if isinstance(screenshot_base64, (bytearray, memoryview)):
                            screenshot_base64 = base64.b64encode(screenshot_base64).decode("utf-8")
                    self._save_screenshot(screenshot_base64, action_type="state")
                    logger.info("Screenshot saved to trajectory")

                # First add any existing user messages that were passed to run()
                user_query = None
                for msg in messages:
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                        if isinstance(user_content, str) and user_content:
                            user_query = user_content
                            # Add the user's original query to the message manager
                            self.message_manager.add_user_message(
                                [{"type": "text", "text": user_content}]
                            )
                            break

                # Add screenshot to message manager
                message_content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_base64,
                        },
                    }
                ]

                # Add appropriate text with the screenshot
                message_content.append(
                    {
                        "type": "text",
                        "text": user_query,
                    }
                )

                # Add the screenshot and text to the message manager
                self.message_manager.add_user_message(message_content)

                # Process user request and convert our standard message format to one OpenAI expects
                messages = self.message_manager.messages
                logger.info(f"Starting agent loop with {len(messages)} messages")

                # Create initial turn directory
                if self.save_trajectory:
                    self._create_turn_dir()

                # Call API
                screen_size = await self.computer.interface.get_screen_size()
                response = await self.api_handler.send_initial_request(
                    messages=messages,
                    display_width=str(screen_size["width"]),
                    display_height=str(screen_size["height"]),
                    previous_response_id=self.last_response_id,
                )

                # Store response ID for next request
                # OpenAI API response structure: the ID is in the response dictionary
                if isinstance(response, dict) and "id" in response:
                    self.last_response_id = response["id"]  # Update instance variable
                    logger.info(f"Received response with ID: {self.last_response_id}")
                else:
                    logger.warning(
                        f"Could not find response ID in OpenAI response: {type(response)}"
                    )
                    # Don't reset last_response_id to None - keep the previous value if available

                # Process API response
                await queue.put(response)

                # Loop to continue processing responses until task is complete
                task_complete = False
                while not task_complete:
                    # Check if there are any computer calls
                    output_items = response.get("output", []) or []
                    computer_calls = [
                        item for item in output_items if item.get("type") == "computer_call"
                    ]

                    if not computer_calls:
                        logger.info("No computer calls in response, task may be complete.")
                        task_complete = True
                        continue

                    # Process the first computer call
                    computer_call = computer_calls[0]
                    action = computer_call.get("action", {})
                    call_id = computer_call.get("call_id")

                    # Check for safety checks
                    pending_safety_checks = computer_call.get("pending_safety_checks", [])
                    acknowledged_safety_checks = []

                    if pending_safety_checks:
                        # Log safety checks
                        for check in pending_safety_checks:
                            logger.warning(
                                f"Safety check: {check.get('code')} - {check.get('message')}"
                            )

                        # If we have a callback, use it to acknowledge safety checks
                        if self.acknowledge_safety_check_callback:
                            acknowledged = await self.acknowledge_safety_check_callback(
                                pending_safety_checks
                            )
                            if not acknowledged:
                                logger.warning("Safety check acknowledgment failed")
                                await queue.put(
                                    {
                                        "role": "assistant",
                                        "content": "Safety checks were not acknowledged. Cannot proceed with action.",
                                        "metadata": {"title": "⚠️ Safety Warning"},
                                    }
                                )
                                continue
                            acknowledged_safety_checks = pending_safety_checks

                    # Execute the action
                    try:
                        # Create a new turn directory for this action if saving trajectories
                        if self.save_trajectory:
                            self._create_turn_dir()

                        # Execute the tool
                        result = await self.tool_manager.execute_tool("computer", action)

                        # Take screenshot after action
                        screenshot = await self.computer.interface.screenshot()
                        if isinstance(screenshot, bytes):
                            screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
                        else:
                            screenshot_base64 = screenshot

                        # Create computer_call_output
                        computer_call_output = {
                            "type": "computer_call_output",
                            "call_id": call_id,
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_base64}",
                            },
                        }

                        # Add acknowledged safety checks if any
                        if acknowledged_safety_checks:
                            computer_call_output["acknowledged_safety_checks"] = (
                                acknowledged_safety_checks
                            )

                        # Save to message manager for history
                        self.message_manager.add_system_message(
                            f"[Computer action executed: {action.get('type')}]"
                        )
                        self.message_manager.add_user_message([computer_call_output])

                        # For follow-up requests with previous_response_id, we only need to send
                        # the computer_call_output, not the full message history
                        # The API handler will extract this from the message history
                        if isinstance(self.last_response_id, str):
                            response = await self.api_handler.send_computer_call_request(
                                messages=self.message_manager.messages,
                                display_width=str(screen_size["width"]),
                                display_height=str(screen_size["height"]),
                                previous_response_id=self.last_response_id,  # Use instance variable
                            )

                        # Store response ID for next request
                        if isinstance(response, dict) and "id" in response:
                            self.last_response_id = response["id"]  # Update instance variable
                            logger.info(f"Received response with ID: {self.last_response_id}")
                        else:
                            logger.warning(
                                f"Could not find response ID in OpenAI response: {type(response)}"
                            )
                            # Keep using the previous response ID if we can't find a new one

                        # Process the response
                        # await self.response_handler.process_response(response, queue)
                        await queue.put(response)
                    except Exception as e:
                        logger.error(f"Error executing computer action: {str(e)}")
                        await queue.put(
                            {
                                "role": "assistant",
                                "content": f"Error executing action: {str(e)}",
                                "metadata": {"title": "❌ Error"},
                            }
                        )
                        task_complete = True

            except Exception as e:
                logger.error(f"Error capturing initial screenshot: {str(e)}")
                await queue.put(
                    {
                        "role": "assistant",
                        "content": f"Error capturing screenshot: {str(e)}",
                        "metadata": {"title": "❌ Error"},
                    }
                )
                await queue.put(None)  # Signal that we're done
                return

            # Signal that we're done
            await queue.put(None)

        except Exception as e:
            logger.error(f"Error in _run_loop: {str(e)}")
            await queue.put(
                {
                    "role": "assistant",
                    "content": f"Error: {str(e)}",
                    "metadata": {"title": "❌ Error"},
                }
            )
            await queue.put(None)  # Signal that we're done

    def get_last_response_id(self) -> Optional[str]:
        """Get the last response ID.

        Returns:
            The last response ID or None if no response has been received
        """
        return self.last_response_id

    def set_last_response_id(self, response_id: str) -> None:
        """Set the last response ID.

        Args:
            response_id: OpenAI response ID to set
        """
        self.last_response_id = response_id
        logger.info(f"Manually set response ID to: {self.last_response_id}")
