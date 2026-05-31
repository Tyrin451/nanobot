from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# google-genai is optional
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

class VertexAIProvider(LLMProvider):
    """LLM provider using Google GenAI SDK for Vertex AI APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gemini-1.5-pro",
        *,
        project_id: str | None = None,
        location: str | None = None,
        credentials_json: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model

        if genai is None:
            raise ImportError(
                "google-genai is required for Vertex AI. "
                "Install it with `pip install google-genai`."
            )

        # Allow environment variable overrides if explicitly not provided
        if not project_id:
            project_id = os.environ.get("VERTEX_AI_PROJECT_ID")
        if not location:
            location = os.environ.get("VERTEX_AI_LOCATION")

        self.project_id = project_id
        self.location = location

        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "project": self.project_id,
            "location": self.location,
        }

        if credentials_json:
            try:
                from google.oauth2 import service_account
                creds_dict = json.loads(credentials_json)
                creds = service_account.Credentials.from_service_account_info(creds_dict)
                client_kwargs["credentials"] = creds
            except Exception as e:
                logger.error(f"Failed to load vertex_ai credentials_json: {e}")
        elif api_key:
            client_kwargs["api_key"] = api_key

        self._client = genai.Client(**client_kwargs)

    def get_default_model(self) -> str:
        return self.default_model

    def _strip_prefix(self, model: str) -> str:
        if model.startswith("vertex_ai/"):
            return model[len("vertex_ai/"):]
        return model

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[Any]]:
        contents = []
        system_instruction = None

        # Merge consecutive messages with the same role as required by Gemini
        merged_messages = []
        for msg in messages:
            if not merged_messages:
                merged_messages.append(dict(msg))
                continue

            last_msg = merged_messages[-1]
            if last_msg.get("role") == msg.get("role") and msg.get("role") in ("user", "assistant"):
                if isinstance(last_msg["content"], str) and isinstance(msg["content"], str):
                    last_msg["content"] += "\n" + msg["content"]
                elif isinstance(last_msg["content"], list) and isinstance(msg["content"], list):
                    last_msg["content"].extend(msg["content"])
                elif isinstance(last_msg["content"], str) and isinstance(msg["content"], list):
                    last_msg["content"] = [{"type": "text", "text": last_msg["content"]}] + msg["content"]
                elif isinstance(last_msg["content"], list) and isinstance(msg["content"], str):
                    last_msg["content"].append({"type": "text", "text": msg["content"]})

                if "tool_calls" in msg:
                    last_msg.setdefault("tool_calls", []).extend(msg["tool_calls"])
            else:
                merged_messages.append(dict(msg))

        for msg in merged_messages:
            role = msg.get("role")
            if role == "system":
                if not system_instruction:
                    system_instruction = msg.get("content", "")
                continue

            parts = []
            if isinstance(msg.get("content"), list):
                for p in msg.get("content"):
                    if p.get("type") == "text":
                        parts.append(types.Part.from_text(text=p["text"]))
                    elif p.get("type") == "image_url":
                        url = p.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            parts.append(
                                types.Part.from_bytes(
                                    data=base64.b64decode(url.split(",")[1]),
                                    mime_type=url.split(";")[0].split(":")[1],
                                )
                            )
                        else:
                            parts.append(
                                types.Part.from_uri(uri=url, mime_type="image/jpeg")
                            )
            elif isinstance(msg.get("content"), str) and msg.get("content") and role != "tool":
                parts.append(types.Part.from_text(text=msg.get("content")))

            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    parts.append(
                        types.Part.from_function_call(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                    )

            if role == "tool":
                parts.append(
                    types.Part.from_function_response(
                        name=msg.get("name", "unknown_tool"),
                        response={"content": msg.get("content", "")},
                    )
                )

            # Gemini uses "user" and "model" roles
            gemini_role = "user" if role in ("user", "tool") else "model"
            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        return system_instruction, contents

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[Any] | None:
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            schema = func.get("parameters")

            # types.Schema handles python dicts transparently
            fd = types.FunctionDeclaration(
                name=func.get("name"),
                description=func.get("description"),
                parameters=schema,
            )
            function_declarations.append(fd)

        if not function_declarations:
            return None

        return [types.Tool(function_declarations=function_declarations)]

    def _build_config(
        self,
        system: str | None,
        max_tokens: int,
        temperature: float,
        tools: list[Any] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            kwargs["system_instruction"] = system

        if tools:
            kwargs["tools"] = tools

        if tool_choice:
            if tool_choice == "required":
                kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )
            elif tool_choice == "none":
                kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                )
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                name = tool_choice.get("function", {}).get("name")
                if name:
                    kwargs["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=[name]
                        )
                    )

        return types.GenerateContentConfig(**kwargs)

    # ------------------------------------------------------------------
    # Core API Implementation
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        model_name = self._strip_prefix(model or self.default_model)
        system, contents = self._convert_messages(self._sanitize_empty_content(messages))
        gemini_tools = self._convert_tools(tools)
        config = self._build_config(system, max_tokens, temperature, gemini_tools, tool_choice)

        try:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return self._parse_response(response)
        except Exception as e:
            return self._handle_error(e)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        model_name = self._strip_prefix(model or self.default_model)
        system, contents = self._convert_messages(self._sanitize_empty_content(messages))
        gemini_tools = self._convert_tools(tools)
        config = self._build_config(system, max_tokens, temperature, gemini_tools, tool_choice)

        try:
            full_text = ""
            tool_calls = []

            async for chunk in await self._client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    full_text += chunk.text
                    if on_content_delta:
                        await on_content_delta(chunk.text)

                if chunk.function_calls:
                    for fc in chunk.function_calls:
                        tc = {
                            "id": fc.id or f"call_{len(tool_calls)}",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {})),
                            },
                            "type": "function",
                        }
                        tool_calls.append(tc)
                        if on_tool_call_delta:
                            # Stream simulated delta since genai doesn't give us raw json deltas
                            await on_tool_call_delta({
                                "index": len(tool_calls) - 1,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": tc["function"]["arguments"]
                                }
                            })

            finish_reason = "stop"
            return LLMResponse(
                content=full_text,
                tool_calls=[
                    ToolCallRequest(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in tool_calls
                ] if tool_calls else None,
                finish_reason=finish_reason,
            )

        except Exception as e:
            return self._handle_error(e)

    def _parse_response(self, response: Any) -> LLMResponse:
        content = ""
        tool_calls: list[ToolCallRequest] | None = None
        finish_reason = "stop"

        if response.text:
            content = response.text

        if response.function_calls:
            tool_calls = []
            for fc in response.function_calls:
                # The SDK might not provide IDs for tool calls in older models
                call_id = fc.id or f"call_{len(tool_calls)}"
                tool_calls.append(
                    ToolCallRequest(
                        id=call_id,
                        name=fc.name,
                        arguments=fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {}),
                    )
                )

        if response.candidates and response.candidates[0].finish_reason:
            fr = response.candidates[0].finish_reason
            if fr == "MAX_TOKENS":
                finish_reason = "length"
            elif fr == "SAFETY":
                finish_reason = "error"
                content = content or "(Content blocked by safety filters)"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    def _handle_error(self, e: Exception) -> LLMResponse:
        logger.error(f"Vertex AI API error: {e!r}")
        # Simplistic error handling
        return LLMResponse(
            content="",
            finish_reason="error",
            error_message=str(e),
            error_type=type(e).__name__,
        )
