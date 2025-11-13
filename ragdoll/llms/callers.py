from __future__ import annotations

import asyncio
import threading
from typing import Any, Protocol, runtime_checkable


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


def call_llm_sync(llm_caller: BaseLLMCaller, prompt: str) -> str:
    """
    Execute ``BaseLLMCaller.call`` from synchronous code.

    If no event loop is running, this simply uses ``asyncio.run``. When invoked
    from an active event loop, it spins up a dedicated thread with its own loop
    to avoid blocking the caller's loop.
    """

    async def _invoke() -> str:
        return await llm_caller.call(prompt)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if not loop or not loop.is_running():
        return asyncio.run(_invoke())

    result: dict[str, str] = {}
    error: list[BaseException] = []

    def _run_in_thread() -> None:
        thread_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(thread_loop)
            result["value"] = thread_loop.run_until_complete(_invoke())
        except BaseException as exc:  # pragma: no cover - defensive
            error.append(exc)
        finally:
            thread_loop.close()
            asyncio.set_event_loop(None)

    worker = threading.Thread(target=_run_in_thread, daemon=True)
    worker.start()
    worker.join()

    if error:
        raise error[0]

    return result.get("value", "")
