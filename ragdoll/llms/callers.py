from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class BaseLLMCaller(Protocol):
    """Minimal interface for objects that can answer prompts."""

    async def call(self, prompt: str) -> str:
        ...


class LangChainLLMCaller(BaseLLMCaller):
    """Adapter around LangChain BaseLanguageModel instances."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def call(self, prompt: str) -> str:
        result = await self._invoke(prompt)
        if hasattr(result, "content"):
            return result.content  # type: ignore[return-value]
        if isinstance(result, str):
            return result
        return str(result)

    async def _invoke(self, prompt: str) -> Any:
        if hasattr(self.llm, "ainvoke"):
            return await self.llm.ainvoke(prompt)

        invoke = getattr(self.llm, "invoke", None)
        if asyncio.iscoroutinefunction(invoke):
            return await invoke(prompt)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.llm.invoke(prompt))
