import tiktoken
from collections import defaultdict
from typing import List, Dict
from datasets import DatasetDict, Dataset
from src.llm.prompt.base_templates import TEXT_PROMPT_TEMPLATE_ZH_V1

def format_fine_tune_dataset_as_messages(dataset: Dataset) -> List[List[Dict[str, str]]]:
    """
    Formats each row in the dataset into a list of messages.
    
    Args:
        dataset (Dataset): A Dataset class from Hugging Face.
    
    Returns:
        List[List[Dict[str, str]]]: A list of message lists.
    """

    # Define the fixed system role message
    system_message = {
        "role": "system",
        "content": TEXT_PROMPT_TEMPLATE_ZH_V1
    }
    
    # Initialize a list to hold the formatted messages
    messages_list = []
    
    for example in dataset:
        # Combine the system message and user message into a list
        messages = [system_message] + example['messages']
        
        # Add the list of messages to the final list
        messages_list.append(messages)
    
    return messages_list

def format_fine_tune_dataset_as_openai_input(dataset: Dataset) -> List[List[Dict[str, str]]]:
    """
    Formats each row in the dataset into a list of messages.
    
    Args:
        dataset (Dataset): A Dataset class from Hugging Face.
    
    Returns:
        List[List[Dict[str, str]]]: A list of message lists.
    """

    # Define the fixed system role message
    system_message = {
        "role": "system",
        "content": TEXT_PROMPT_TEMPLATE_ZH_V1
    }
    
    # Initialize a list to hold the formatted messages
    messages_list = []
    
    for example in dataset:
        # Combine the system message and user message into a list
        messages = [system_message] + example['messages']
        
        # Add the list of messages to the final list
        messages_list.append({"messages": messages})
    
    return messages_list


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

def format_fine_tune_dataset_as_openai_input_with_threshold(
        dataset: Dataset, 
        M_token_threshold: int,
        model: str = "gpt-4o-mini",
        prompt_template: str = TEXT_PROMPT_TEMPLATE_ZH_V1
    ) -> List[List[Dict[str, str]]]:
    """
    Formats each row in the dataset into a list of messages and returns only 
    the subset of the dataset such that the total tokens do not exceed the given threshold.
    
    Args:
        dataset (Dataset): A Dataset class from Hugging Face.
        M_token_threshold (int): Maximum number of tokens (M) for the dataset.
        model (str): Model name to use for token calculation.
        prompt_template (str): The system prompt template to be added in each message list.

    Returns:
        List[List[Dict[str, str]]]: A list of message lists within the token threshold.
    """
    
    # Define the fixed system role message
    system_message = {
        "role": "system",
        "content": prompt_template
    }
    
    # Initialize variables
    messages_list = []
    threshold = M_token_threshold * 1_000_000
    total_tokens = 0

    for example in dataset:
        # Combine the system message and user messages into a list
        messages = [system_message] + example['messages']
        
        # Calculate the number of tokens for this example
        example_tokens = num_tokens_from_messages(messages, model=model)

        # Check if adding this example would exceed the token threshold
        if total_tokens + example_tokens > threshold:
            break
        
        # Add the current example's tokens to the total
        total_tokens += example_tokens
        
        # Append to the messages list
        messages_list.append({
            "messages": messages
        })
    
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

def check_openai_format_errors(dataset: List[Dict]) -> None:
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")