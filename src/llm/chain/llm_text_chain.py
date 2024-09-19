import os
import asyncio
from openlimit import ChatRateLimiter
from openai import AsyncOpenAI
from typing import List, Dict
from src.llm.prompt.base_templates import (
    TEXT_PROMPT_TEMPLATE_ZH_V1
)

# Initialize the rate limiter
rate_limiter = ChatRateLimiter(request_limit=500, token_limit=200_000)

@rate_limiter.is_limited()
async def call_openai(messages: List[Dict], openai_llm_endpoint: str = 'gpt-4o-mini') -> str:
    """
    Call OpenAI's API with rate limiting, preparing the parameters and making the API request.

    Args:
        messages (List[Dict]): List of messages to send to the model.
        openai_llm_endpoint (str): The model name for the API call (default: 'gpt-4o-mini').

    Returns:
        str: The response content from the LLM.
    """
    # Prepare chat parameters for the API call
    chat_params = {
        "model": openai_llm_endpoint,
        "messages": messages,
    }

    # Initialize the OpenAI client
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Make the API call to OpenAI's chat completion endpoint
    response  = await client.chat.completions.create(**chat_params)

    return response

async def call_openai_parallel(messages_list: List[List[Dict]]) -> List[str]:
    """
    Send all message batches to the LLM in parallel using asyncio.gather.
    
    Args:
        messages_list (List[List[Dict]]): List of message batches to be sent to the OpenAI API.
    
    Returns:
        List[str]: List of responses from the LLM.
    """
    # Gather all the responses asynchronously
    tasks = [call_openai(messages=msg) for msg in messages_list]
    
    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    
    # Return the list of responses
    return responses