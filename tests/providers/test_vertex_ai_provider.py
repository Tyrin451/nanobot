import pytest
from nanobot.providers.vertex_ai_provider import VertexAIProvider
from nanobot.providers.base import LLMResponse, ToolCallRequest

def test_convert_tools():
    provider = VertexAIProvider(project_id="test", location="us-central1")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]
    converted = provider._convert_tools(tools)
    assert len(converted) == 1
    tool = converted[0]
    # Tool object
    assert len(tool.function_declarations) == 1
    fd = tool.function_declarations[0]
    assert fd.name == "get_weather"
    assert fd.description == "Get current weather."
    # The parameters shouldn't crash

def test_convert_messages():
    provider = VertexAIProvider(project_id="test", location="us-central1")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "What's the weather?"}
    ]
    sys, contents = provider._convert_messages(messages)
    assert sys == "You are a helpful assistant"
    assert len(contents) == 3

    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello"

    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Hi"

    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "What's the weather?"

def test_convert_messages_merges_same_role():
    provider = VertexAIProvider(project_id="test", location="us-central1")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Hi"},
        {"role": "assistant", "content": "I am good."}
    ]
    sys, contents = provider._convert_messages(messages)
    assert len(contents) == 2

    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello\nHow are you?"

    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Hi\nI am good."

def test_convert_messages_with_tools():
    provider = VertexAIProvider(project_id="test", location="us-central1")
    messages = [
        {"role": "user", "content": "Hello", "tool_calls": [{"function": {"name": "func", "arguments": "{}"}}]},
        {"role": "tool", "name": "func", "content": "result"}
    ]
    sys, contents = provider._convert_messages(messages)
    assert len(contents) == 2

    assert contents[0].role == "user"
    assert len(contents[0].parts) == 2
    assert contents[0].parts[0].text == "Hello"
    assert contents[0].parts[1].function_call is not None
    assert contents[0].parts[1].function_call.name == "func"

    assert contents[1].role == "user"
    assert contents[1].parts[0].function_response is not None
    assert contents[1].parts[0].function_response.name == "func"
