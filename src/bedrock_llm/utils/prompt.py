from typing import Optional


def llama_format(user_messages, system: Optional[str]=None):
    
    if not system:
        system = "You are a helpful assistant."
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_messages}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    


def mistral_format(user_messages, system: Optional[str]=None):

    if not system:
        system = "You are a helpful assistant."

    return f"""<s>[INST] {system}

{user_messages} [/INST] """
