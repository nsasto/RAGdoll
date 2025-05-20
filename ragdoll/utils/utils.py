from typing import Type, Optional
from pydantic import BaseModel, ValidationError
import json
import re

def fix_json(s: str) -> str:
    """
    Attempts to fix common JSON errors like missing quotes,
    trailing commas, and unescaped characters. This is a simplified
    fixer and may not handle all cases.
    """
    s = re.sub(r'([{,]\s*)([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', s)  # Fix unquoted keys
    s = re.sub(r',\s*}', '}', s)  # Remove trailing commas
    s = re.sub(r',\s*]', ']', s)  # Remove trailing commas
    s = s.replace('\\', '\\\\')  # Escape backslashes
    s = re.sub(r'([^\\])"', r'\1\\"', s)  # Basic string escaping
    return s

def json_parse(response: str, pydantic_object: Optional[Type[BaseModel]] = dict, max_retries: int = 3) -> Optional[BaseModel]:
    """
    Robustly parses a string response into a Pydantic object or a dictionary if no Pydantic model is provided.
    Handles common LLM output issues.

    Args:
        response: The string response to parse (typically from an LLM).
        pydantic_object: The Pydantic model to parse the response into, or dict if not specified.
        max_retries: Maximum number of attempts to parse.

    Returns:
        The parsed Pydantic object or dictionary, or None if parsing fails after multiple retries.
    """
    for attempt in range(max_retries):
        try:
            # 1. Attempt direct JSON parsing first. This is the ideal case.
            try:
                json_response = json.loads(response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except json.JSONDecodeError:
                pass  # If it's not valid JSON, move to the next attempt

            # 2. Clean up common LLM formatting issues:
            cleaned_response = response.strip()
            cleaned_response = re.sub(r'```(json)?\n?', '', cleaned_response)  # Remove ```json and ```
            cleaned_response = re.sub(r'```', '', cleaned_response)
            cleaned_response = re.sub(r'\n+', '\n', cleaned_response)  # Reduce multiple newlines
            cleaned_response = cleaned_response.strip()  # Remove leading/trailing whitespace

            # 3. Attempt to extract a JSON-like substring.
            if "{" in cleaned_response and "}" in cleaned_response:
                start_index = cleaned_response.find("{")
                end_index = cleaned_response.rfind("}") + 1
                json_like_substring = cleaned_response[start_index:end_index]
                try:
                    json_response = json.loads(json_like_substring)
                    if pydantic_object == dict:
                        return json_response
                    return pydantic_object.model_validate(json_response)
                except json.JSONDecodeError:
                    pass

            # 4. Handle potential issues with extra newlines or incomplete JSON
            try:
                json_response = json.loads(cleaned_response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except json.JSONDecodeError:
                pass

            # 5. Use the fix_json function to fix common JSON errors.
            fixed_json_response = fix_json(cleaned_response)
            try:
                json_response = json.loads(fixed_json_response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except json.JSONDecodeError:
                pass

            # 6. Try parsing the original response after fixes.
            try:
                json_response = json.loads(cleaned_response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except ValidationError as e:
                if attempt < max_retries - 1:
                    print(f"Parsing failed on attempt {attempt + 1}: {e}. Retrying...")
                else:
                    print(f"Parsing failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Parsing failed on attempt {attempt + 1}: {e}. Retrying...")
                else:
                    print(f"Parsing failed after {max_retries} attempts: {e}")
                    return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"An unexpected error occurred on attempt {attempt + 1}: {e}. Retrying...")
            else:
                print(f"An unexpected error occurred after {max_retries} attempts: {e}")
                return None

    return None  # Return None if all attempts fail