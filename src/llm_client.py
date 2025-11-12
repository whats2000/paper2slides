"""LLM client functions for interacting with language models."""

import logging
import os
import re
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletion

from prompts import PromptManager
from .latex_utils import sanitize_frametitles

# Initialize prompt manager
prompt_manager = PromptManager()


def extract_content_from_response(
    response: ChatCompletion, language: str = "latex"
) -> str | None:
    """
    :param response: Response from the language model
    :param language: Language to extract (default is 'latex')
    :return: Extracted content
    """
    pattern = re.compile(rf"```{language}\s*(.*?)```", re.DOTALL)
    match = pattern.search(response.choices[0].message.content)
    content = match.group(1).strip() if match else None
    return content


def resolve_api_credentials(api_key: str | None = None, base_url: str | None = None) -> tuple[str, str | None]:
    """
    Resolve API key and base URL from provided values or environment variables.
    
    Args:
        api_key: Optional API key (will check environment if None)
        base_url: Optional base URL (will check environment if None)
        
    Returns:
        Tuple of (resolved_api_key, resolved_base_url)
        
    Raises:
        RuntimeError: If no API key can be found
    """
    resolved_api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
    )
    if not resolved_api_key:
        raise RuntimeError(
            "No API key provided. Set OPENAI_API_KEY or DASHSCOPE_API_KEY."
        )
    
    # Determine base_url
    resolved_base_url = base_url
    if not resolved_base_url:
        if resolved_api_key == os.environ.get("DASHSCOPE_API_KEY"):
            # DashScope provider
            resolved_base_url = (
                os.environ.get("DASHSCOPE_BASE_URL") 
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif os.environ.get("OPENAI_BASE_URL"):
            # Custom OpenAI-compatible provider
            resolved_base_url = os.environ.get("OPENAI_BASE_URL")
    
    return resolved_api_key, resolved_base_url


def get_model_name(model_name: str, base_url: str | None) -> str:
    """
    Auto-adjust model name for DashScope if needed.
    
    Args:
        model_name: Requested model name
        base_url: Base URL being used
        
    Returns:
        Adjusted model name
    """
    if (
        isinstance(base_url, str)
        and "dashscope.aliyuncs.com" in base_url
        and isinstance(model_name, str)
        and (
            model_name.startswith("gpt-")
            or model_name.startswith("o1")
            or model_name.startswith("o3")
        )
    ):
        return os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
    return model_name


def create_llm_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """
    Create an OpenAI client with resolved credentials.
    
    Args:
        api_key: Optional API key
        base_url: Optional base URL
        
    Returns:
        Configured OpenAI client
    """
    resolved_api_key, resolved_base_url = resolve_api_credentials(api_key, base_url)
    
    client_kwargs = {"api_key": resolved_api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    
    return OpenAI(**client_kwargs)


def call_llm(
    system_message: str,
    user_prompt: str,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
) -> str | None:
    """
    Call the LLM with system and user messages.
    
    Args:
        system_message: System message for the LLM
        user_prompt: User prompt for the LLM
        api_key: API key
        model_name: Model name
        base_url: Optional base URL
        
    Returns:
        Extracted content from response, or None on error
    """
    try:
        client = create_llm_client(api_key, base_url)
        resolved_base_url = client.base_url.host if hasattr(client.base_url, 'host') else str(client.base_url)
        model_to_use = get_model_name(model_name, resolved_base_url)
        
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                ChatCompletionSystemMessageParam(content=system_message, role='system'),
                ChatCompletionUserMessageParam(content=user_prompt, role='user'),
            ],
        )
        
        content = extract_content_from_response(response)
        if content:
            return sanitize_frametitles(content)
        return None
        
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        # Provide guidance for DashScope access issues
        if "dashscope.aliyuncs.com" in str(base_url or "") and (
            "403" in str(e) or "access_denied" in str(e)
        ):
            logging.error(
                "DashScope access denied. Ensure your key has access to the model. "
                "Set DASHSCOPE_MODEL to a model you can use (e.g., qwen-plus)."
            )
        return None


def call_llm_for_revision(
    latex_source: str,
    beamer_code: str,
    linter_log: str,
    figure_paths: list[str],
    api_key: str,
    model_name: str,
    base_url: str | None = None,
) -> str | None:
    """
    Call LLM for revision/fixing of Beamer code (stage 3).
    
    Args:
        latex_source: Original LaTeX source
        beamer_code: Current Beamer code
        linter_log: Linter output
        figure_paths: Available figure paths
        api_key: API key
        model_name: Model name
        base_url: Optional base URL
        
    Returns:
        Fixed Beamer code, or None on error
    """
    system_message, user_prompt = prompt_manager.build_prompt(
        stage=3,  # revise stage
        latex_source=latex_source,
        beamer_code=beamer_code,
        linter_log=linter_log,
        figure_paths=figure_paths,
    )
    
    return call_llm(system_message, user_prompt, api_key, model_name, base_url)


def process_stage(
    stage: int,
    latex_source: str,
    beamer_code: str,
    linter_log: str,
    figure_paths: list[str],
    slides_tex_path: str,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
) -> bool:
    """
    Process a specific stage of slide generation.
    
    Args:
        stage: Stage number (1, 2, or 3)
        latex_source: Original LaTeX source
        beamer_code: Current Beamer code (empty for stage 1)
        linter_log: Linter log (used in stage 3)
        figure_paths: Available figure paths
        slides_tex_path: Path to save the slides
        api_key: API key
        model_name: Model name
        base_url: Optional base URL
        
    Returns:
        True on success, False on failure
    """
    system_message, user_prompt = prompt_manager.build_prompt(
        stage=stage,
        latex_source=latex_source,
        beamer_code=beamer_code,
        linter_log=linter_log,
        figure_paths=figure_paths,
    )

    try:
        client = create_llm_client(api_key, base_url)
        resolved_base_url = client.base_url.host if hasattr(client.base_url, 'host') else str(client.base_url)
        model_to_use = get_model_name(model_name, resolved_base_url)
        
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                ChatCompletionSystemMessageParam(content=system_message, role='system'),
                ChatCompletionUserMessageParam(content=user_prompt, role='user'),
            ],
        )

        logging.info("Received response from LLM.")

    except Exception as e:
        logging.error(f"Error generating prompt for stage {stage}: {e}")
        # Provide guidance for DashScope access issues
        if "dashscope.aliyuncs.com" in str(base_url or "") and (
            "403" in str(e) or "access_denied" in str(e)
        ):
            logging.error(
                "DashScope access denied. Ensure your key has access to the model. "
                "Set DASHSCOPE_MODEL to a model you can use (e.g., qwen-plus)."
            )
        return False

    new_beamer_code = extract_content_from_response(response)

    new_beamer_code = sanitize_frametitles(new_beamer_code)

    if not new_beamer_code:
        logging.error("No beamer code found in the response.")
        return False

    with open(slides_tex_path, "w") as file:
        file.write(new_beamer_code)
    logging.info(f"Beamer code saved to {slides_tex_path}")
    return True
