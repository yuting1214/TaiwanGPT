import os
from openai import AsyncOpenAI
from typing import List, Optional, Tuple
from src.llm.prompt.base_templates import (
    TEXT_PROMPT_TEMPLATE_ZH_V1
)

async def llm_OpenAI_chain(
        user_input: str,
        openai_llm_endpoint: str
    ) -> str:

    client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Build the conversation history for the prompt
    messages = [{"role": "system", "content": TEXT_PROMPT_TEMPLATE_ZH_V1}]
    # Add the current user input to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Make the API call to OpenAI's chat completion endpoint
    chat_completion = await client.chat.completions.create(
        model=openai_llm_endpoint,
        messages=messages
    )

    # Extract and return the llm's response
    llm_response = chat_completion.choices[0].message.content
    return llm_response