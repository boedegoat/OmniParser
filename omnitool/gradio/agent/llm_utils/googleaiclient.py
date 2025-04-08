import os
from typing import List, Dict, Any, Tuple, Union
from PIL import Image, UnidentifiedImageError
import copy
import google.generativeai as genai
from google.generativeai import types as genai_types

from .utils import is_image_path

def run_gemini_interleaved(
    messages: List[Dict[str, Any]],
    system: str,
    model_name: str,
    api_key: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, int]]:
    """
    Calls the Google Gemini API with interleaved text and image inputs.
    System instruction is passed during model initialization if supported.

    Args:
        messages: History of messages. Role should be "user" or "assistant".
                  Content is a list of text strings or image paths.
        system: The system instruction string.
        model_name: The specific Gemini model ID (e.g., "gemini-1.5-pro-latest").
        api_key: The Google AI API key.
        max_tokens: Maximum number of tokens to generate.
        temperature: The sampling temperature.

    Returns:
        A tuple containing:
          - The response text (str). Contains an error message if call failed.
          - A dictionary with token counts:
            {"input_tokens": int, "output_tokens": int, "total_tokens": int}.
            Contains zeros if call failed or metadata unavailable.
    """
    response_text = ""
    token_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        genai.configure(api_key=api_key)
        system_content = None
        if system:
            system_content = system

        
        model = genai.GenerativeModel(
                model_name,
                system_instruction=system_content)
        print(f"Initialized Gemini Model '{model_name}' with System Instruction (if provided).")
       

        #Making the message history
        gemini_history: List[Dict[str, Any]] = []
        processed_messages = copy.deepcopy(messages)
        for msg in processed_messages:
            role = msg.get("role")
            content = msg.get("content")

            # convert role (to make same rule as the library documentation)
            if role == "assistant": role = "model"
            elif role == "tool": continue 
            if role not in ["user", "model"]: continue 

            gemini_parts: List[Union[str, Image.Image]] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        gemini_parts.append(item["text"])
                    elif isinstance(item, str) and is_image_path(item):
                        try:
                            if os.path.exists(item):
                                img = Image.open(item)
                                gemini_parts.append(img)
                            else:
                                print(f"Warning [Gemini Helper]: Image path not found, skipping: {item}")
                                gemini_parts.append(f"[Image not found: {os.path.basename(item)}]")
                        except FileNotFoundError:
                            print(f"Warning [Gemini Helper]: Image file not found, skipping: {item}")
                            gemini_parts.append(f"[Image not found: {os.path.basename(item)}]")
                        except UnidentifiedImageError:
                             print(f"Warning [Gemini Helper]: Could not identify image file, skipping: {item}")
                             gemini_parts.append(f"[Unidentifiable image: {os.path.basename(item)}]")
                        except Exception as e:
                            print(f"Warning [Gemini Helper]: Error loading image {item}, skipping: {e}")
                            gemini_parts.append(f"[Error loading image: {os.path.basename(item)}]")
                    elif isinstance(item, str):
                        gemini_parts.append(item)
            elif isinstance(content, str): 
                 gemini_parts.append(content)

            if gemini_parts:
                 gemini_history.append({"role": role, "parts": gemini_parts})

        generation_config = genai_types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        print(f"Calling Gemini API [Helper]: Model={model_name}, History Size={len(gemini_history)}")
        response = model.generate_content(
             contents=gemini_history,
             generation_config=generation_config
        )
        response_text = response.text
        
        try:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            token_info = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
            print(f"Gemini Response received [Helper]. Tokens: In={input_tokens}, Out={output_tokens}, Total={token_info['total_tokens']}")
        except AttributeError:
             print("Warning [Gemini Helper]: Could not retrieve token usage metadata.")
             # Biarkan token_info sebagai nol

    except ImportError as e:
         print(f"Error [Gemini Helper]: {e}")
         response_text = f"ImportError: Please install google-generativeai. {e}"
    except AttributeError as e:
         print(f"Error calling Gemini API [Helper] (AttributeError): {e}")
         error_message = f"Error setting up Gemini call (AttributeError): {e}."
         response_text = error_message
    except TypeError as e:
         print(f"Error calling Gemini API [Helper] (TypeError): {e}")
         error_message = f"Type error during Gemini call (check arguments/version): {e}."
         response_text = error_message
    except Exception as e:
        print(f"Error calling Gemini API [Helper]: {e}")
        error_type = type(e).__name__
        error_details = str(e)
        response_text = f"Error interacting with Gemini API: {error_type} - {error_details}"
    return response_text, token_info