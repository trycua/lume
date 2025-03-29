"""OpenAI API client for Agent Response API."""

import logging
import json
import os
import httpx
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI's Agent Response API."""

    def __init__(
        self,
        api_key: str,
        model: str = "computer-use-preview",
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        timeout: int = 120,
        **kwargs,
    ):
        """Initialize OpenAI API client.

        Args:
            api_key: OpenAI API key
            model: Model to use for completions (should always be computer-use-preview)
            base_url: Base URL for API requests
            max_retries: Maximum number of retries for API calls
            timeout: Timeout for API calls in seconds
            **kwargs: Additional arguments to pass to the httpx client
        """
        self.api_key = api_key

        # Always use computer-use-preview model
        if model != "computer-use-preview":
            logger.warning(
                f"Overriding provided model '{model}' with required model 'computer-use-preview'"
            )
            model = "computer-use-preview"

        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

        # Create httpx client with auth and timeout
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "computer-use-2023-09-30",  # Required beta header for computer use
            },
            **kwargs,
        )

        # Additional initialization for organization if available
        openai_org = os.environ.get("OPENAI_ORG")
        if openai_org:
            self.client.headers["OpenAI-Organization"] = openai_org

        logger.info(f"Initialized OpenAI client with model {model}")

    async def create_response(
        self,
        input: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        truncation: str = "auto",
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a response using the OpenAI Agent Response API.

        Args:
            input: List of messages in the conversation (must be in Agent Response API format)
            tools: List of tools available to the agent
            truncation: How to handle truncation (auto, truncate)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters to include in the request

        Returns:
            Response from the API
        """
        url = f"{self.base_url}/responses"

        # Prepare request payload
        payload = {
            "model": self.model,
            "input": input,
            "temperature": temperature,
            "top_p": top_p,
            "truncation": truncation,
            **kwargs,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        try:
            logger.debug(f"Sending request to {url}")

            # Make API call
            response = await self.client.post(url, json=payload)

            # Check for errors
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text
                try:
                    # Try to parse the error as JSON for better debugging
                    error_json = json.loads(error_detail)
                    logger.error(f"HTTP error from OpenAI API: {json.dumps(error_json, indent=2)}")
                except:
                    logger.error(f"HTTP error from OpenAI API: {error_detail}")
                raise

            result = response.json()
            logger.debug("Received successful response")
            return result

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e, "response") else str(e)
            logger.error(f"HTTP error from OpenAI API: {error_detail}")
            raise RuntimeError(f"OpenAI API error: {error_detail}")
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")

    async def close(self):
        """Close the httpx client."""
        await self.client.aclose()
