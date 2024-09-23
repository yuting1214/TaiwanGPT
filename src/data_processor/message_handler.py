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
        prompt_template: str = "Your default prompt template here"
    ) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    Formats each row in the dataset into a list of messages and returns only 
    the subset of the dataset such that the total tokens do not exceed the given threshold.
    
    Args:
        dataset (Dataset): A Dataset class from Hugging Face.
        M_token_threshold (int): Maximum number of tokens (M) for the dataset.
        model (str): Model name to use for token calculation.
        prompt_template (str): The system prompt template to be added in each message list.

    Returns:
        List[Dict[str, List[Dict[str, str]]]]: A list of message dictionaries 
        within the token threshold, where each dictionary has a 'messages' key 
        containing the list of message dicts.
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
        # Ensure that 'messages' is a list of dictionaries in the example
        if 'messages' not in example or not isinstance(example['messages'], list):
            continue

        # Combine the system message and user messages into a list
        messages = [system_message] + example['messages']
        
        # Check if the message list is valid according to OpenAI format requirements
        if not is_valid_openai_example({"messages": messages}):
            continue

        # Calculate the number of tokens for this example
        try:
            example_tokens = num_tokens_from_messages(messages, model=model)
        except Exception as e:
            # Log the error message if token calculation fails and continue with the next example
            print(f"Error calculating tokens for example: {e}")
            continue

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

def is_valid_openai_example(example: Dict) -> bool:
    """
    Validates a single example for OpenAI message format requirements.
    
    Args:
        example (Dict): A dictionary representing a data example with a list of messages.
    
    Returns:
        bool: True if the example is valid, False otherwise.
    """
    # Check that the input is a dictionary
    if not isinstance(example, dict):
        return False

    # Check that 'messages' is a list in the example
    messages = example.get("messages")
    if not isinstance(messages, list):
        return False
    
    # Check that the number of messages is odd (system + alternating user and assistant)
    if len(messages) % 2 != 1:
        return False

    # Ensure the first message has the role 'system'
    if not messages or messages[0].get("role") != "system":
        return False
    
    previous_role = None
    for message in messages:
        # Check that each message has required keys: 'role' and 'content'
        if "role" not in message or "content" not in message:
            return False

        # Check for unrecognized keys
        unrecognized_keys = [k for k in message if k not in ("role", "content", "name", "function_call", "weight")]
        if unrecognized_keys:
            return False

        # Validate the role
        role = message["role"]
        if role not in ("system", "user", "assistant", "function"):
            return False

        # Validate that the message has either 'content' or 'function_call' and content must be a string
        content = message.get("content")
        function_call = message.get("function_call")
        if (not content and not function_call) or (content and not isinstance(content, str)):
            return False

        # Check that 'user' role is always followed by 'assistant'
        if previous_role == "user" and role != "assistant":
            return False
        
        # Update the previous role for the next iteration
        previous_role = role
    
    # Check that there's at least one 'assistant' message
    if not any(message.get("role") == "assistant" for message in messages):
        return False

    return True


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

        if not len(messages) % 2 == 1:
            format_errors["messages_not_format_correct"] += 1

        # Ensure 'system' role is the first message
        if not messages or messages[0].get("role") != "system":
            format_errors["first_message_not_system"] += 1
        
        previous_role = None
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            # Check for unrecognized keys
            unrecognized_keys = [k for k in message if k not in ("role", "content", "name", "function_call", "weight")]
            if unrecognized_keys:
                format_errors["message_unrecognized_key"] += len(unrecognized_keys)
            
            # Validate the role
            role = message["role"]
            if role not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

            # Check for role alternation (user followed by assistant)
            if previous_role == "user" and role != "assistant":
                format_errors["user_not_followed_by_assistant"] += 1
            
            # Update the previous role
            previous_role = role
        
        # Ensure there's at least one assistant message
        if not any(message.get("role") == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")