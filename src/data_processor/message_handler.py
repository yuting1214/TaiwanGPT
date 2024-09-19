import tiktoken
from typing import List, Dict
from datasets import DatasetDict
from src.llm.prompt.base_templates import TEXT_PROMPT_TEMPLATE_ZH_V1

def format_dataset_as_messages(dataset_dict: DatasetDict) -> List[List[Dict[str, str]]]:
    """
    Formats each row in the 'test' split of the dataset into a list of messages.
    
    Args:
        dataset_dict (DatasetDict): A DatasetDict containing 'test' and 'dev' splits.
    
    Returns:
        List[List[Dict[str, str]]]: A list of message lists for the 'test' split.
    """

    # Define the fixed system role message
    system_message = {
        "role": "system",
        "content": TEXT_PROMPT_TEMPLATE_ZH_V1
    }

    # Get the 'test' dataset
    test_dataset = dataset_dict['test']
    
    # Initialize a list to hold the formatted messages
    messages_list = []
    
    for example in test_dataset:
        # Create the user role message using 'user_content' from the dataset
        user_message = {
            "role": "user",
            "content": example["user_content"]
        }
        
        # Combine the system message and user message into a list
        messages = [system_message, user_message]
        
        # Add the list of messages to the final list
        messages_list.append(messages)
    
    return messages_list

def num_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens