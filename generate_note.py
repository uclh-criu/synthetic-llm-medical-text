import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

def generate_note(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4",
    temperature: float = 0.8
) -> str:
    """
    Generate a single note using a prompt.
    
    Args:
        prompt: The main prompt text
        system_prompt: Optional system prompt for setting context/behavior
        model: The OpenAI model to use
        temperature: Controls randomness (0.0-1.0)
    
    Returns:
        Generated text as string
    """
    load_dotenv()
    client = OpenAI()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content

def generate_batch(
    prompt: str,
    n_samples: int = 1,
    **kwargs
) -> List[str]:
    """
    Generate multiple samples using the same prompt.
    
    Args:
        prompt: The main prompt text
        n_samples: Number of samples to generate
        **kwargs: Additional arguments passed to generate_basic_note
    
    Returns:
        List of generated texts
    """
    return [generate_note(prompt, **kwargs) for _ in range(n_samples)]
