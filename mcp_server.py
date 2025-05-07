import asyncio
import argparse
from typing import cast
from mcp.server.fastmcp import FastMCP

# User's ComputerTool import
try:
    from omnitool.gradio.tools.computer import ComputerTool, ToolResult, ToolError
except ImportError as e:
    print(f"Error importing ComputerTool: {e}")
    print("Please ensure 'omnitool' is in your PYTHONPATH or adjust the import path.")
    print("You might need to run this server from the root of your 'OmniParser' project.")
    raise

# --- Initialize ComputerTool ---
# For an MCP server controlling the local machine, host_device should be 'local'.
# ComputerTool's __init__ is: def __init__(self, args, is_scaling: bool = False):
computer_tool_instance = ComputerTool(args=argparse.Namespace(
    host_device="local",
    omniparser_server_url="https://f119-180-243-8-192.ngrok-free.app/"
), is_scaling=False)

mcp = FastMCP("OmniParser")

async def safe_computer_action(action_str: str, text: str | None = None, coordinate: tuple[int, int] | None = None, **kwargs):
    """Wraps the ComputerTool call to handle its specific ToolResult and ToolError."""
    try:
        # ComputerTool.__call__ is async and expects a string for 'action'
        tool_result: ToolResult = await computer_tool_instance(
            action=action_str, # Use the string literal for action
            text=text,
            coordinate=coordinate,
            **kwargs
        )
        return {"status": "success", "output": tool_result.output, "base64_image": getattr(tool_result, 'base64_image', None)}
    except ToolError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error in ComputerTool: {str(e)}"}

# --- MCP Tool Handler Functions ---

@mcp.tool()
async def handle_mouse_move(params):
    x = cast(int, params.get("x"))
    y = cast(int, params.get("y"))
    if x is None or y is None:
        return {"status": "error", "message": "Missing x or y coordinate."}
    return await safe_computer_action(action_str="mouse_move", coordinate=(x,y))

@mcp.tool()
async def handle_mouse_click(params):
    button = cast(str, params.get("button", "left"))
    click_type = cast(str, params.get("click_type", "click")) # click, double

    action_map = {
        ("left", "click"): "left_click",
        ("right", "click"): "right_click",
        ("middle", "click"): "middle_click",
        ("left", "double"): "double_click",
    }
    computer_action_str = action_map.get((button.lower(), click_type.lower()))

    if not computer_action_str:
        return {"status": "error", "message": f"Unsupported button/click_type: {button}/{click_type}"}
    return await safe_computer_action(action_str=computer_action_str)

@mcp.tool()
async def handle_mouse_drag(params):
    x = cast(int, params.get("x"))
    y = cast(int, params.get("y"))
    if x is None or y is None:
        return {"status": "error", "message": "Missing x or y coordinate for drag."}
    return await safe_computer_action(action_str="left_click_drag", coordinate=(x,y))

@mcp.tool()
async def handle_keyboard_type(params):
    text_to_type = cast(str, params.get("text_to_type"))
    if text_to_type is None:
        return {"status": "error", "message": "Missing text_to_type."}
    return await safe_computer_action(action_str="type", text=text_to_type)

@mcp.tool()
async def handle_keyboard_press_key(params):
    key_to_press = cast(str, params.get("key_to_press"))
    if key_to_press is None:
        return {"status": "error", "message": "Missing key_to_press."}
    return await safe_computer_action(action_str="key", text=key_to_press)

@mcp.tool()
async def handle_get_cursor_position(params):
    return await safe_computer_action(action_str="cursor_position")
    
@mcp.tool()
async def handle_wait_seconds(params):
    duration = cast(float, params.get("duration", 0.0)) # Default to 0 if not provided
    if duration > 0:
        # Custom wait, as ComputerTool's "wait" is fixed (usually 1s)
        await asyncio.sleep(duration)
        return {"status": "success", "output": f"Waited for {duration} seconds."}
    else:
        # Use ComputerTool's default "wait" action
        return await safe_computer_action(action_str="wait")

if __name__ == "__main__":
    mcp.run()