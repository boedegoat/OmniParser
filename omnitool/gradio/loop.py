"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
from collections.abc import Callable

from anthropic import APIResponse
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaMessageParam
)
from tools import ToolResult

from agent.llm_utils.omniparserclient import OmniParserClient

from agent.vlm_agent import VLMAgent
from executor.anthropic_executor import AnthropicExecutor

from agent.models import LLM_Provider

BETA_FLAG = "computer-use-2024-10-22"

def sampling_loop_sync(
    *,
    args,
    model: str,
    provider: LLM_Provider | None,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    save_folder: str = "./uploads"
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    print('in sampling_loop_sync, model:', model)
    omniparser_client = OmniParserClient(host_device=args.host_device, url=f"http://{args.omniparser_server_url}/parse/")

    actor = VLMAgent(
        model=model,
        provider=provider,
        api_key=api_key,
        api_response_callback=api_response_callback,
        output_callback=output_callback,
        max_tokens=max_tokens,
        only_n_most_recent_images=only_n_most_recent_images
    )

    executor = AnthropicExecutor(
        args=args,
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
    )
    print(f"Model Inited: {model}, Provider: {provider}")
    
    tool_result_content = None
    
    print(f"Start the message loop. User messages: {messages}")

    while True:
        parsed_screen = omniparser_client()
        tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

        for message, tool_result_content in executor(tools_use_needed, messages):
            yield message
    
        if not tool_result_content:
            return messages