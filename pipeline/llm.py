"""LLM client abstraction supporting OpenAI, Anthropic, and Ollama providers."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai", "anthropic", or "ollama"
    model: str = "gpt-4.1-mini"
    base_url: Optional[str] = None  # override for OpenAI-compatible APIs
    api_key_env: Optional[str] = None
    max_concurrent: int = 20
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    time_ms: float  # wall-clock time for this call


class LLMClient:
    """Unified LLM client with support for OpenAI, Anthropic, and Ollama."""

    def __init__(self, config: LLMConfig):
        self.config = config

        # Adjust defaults for ollama
        if config.provider == "ollama" and config.max_concurrent == 20:
            config.max_concurrent = 1

        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._client = None

    def _get_openai_client(self):
        """Lazy-init OpenAI async client."""
        if self._client is None:
            import openai
            import os

            kwargs = {}
            if self.config.provider == "ollama":
                kwargs["base_url"] = self.config.base_url or "http://localhost:11434/v1"
                kwargs["api_key"] = "ollama"
            else:
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                if self.config.api_key_env:
                    kwargs["api_key"] = os.environ[self.config.api_key_env]

            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    def _get_anthropic_client(self):
        """Lazy-init Anthropic async client."""
        if self._client is None:
            import anthropic
            import os

            kwargs = {}
            if self.config.api_key_env:
                kwargs["api_key"] = os.environ[self.config.api_key_env]

            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Send a completion request with retry logic."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with self._semaphore:
                    if self.config.provider == "anthropic":
                        return await self._complete_anthropic(system, user)
                    else:
                        return await self._complete_openai(system, user)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "rate" in err_str
                    or "429" in err_str
                    or "overloaded" in err_str
                    or "capacity" in err_str
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                raise

    async def _complete_openai(self, system: str, user: str) -> LLMResponse:
        """Complete using OpenAI-compatible API (also used for Ollama)."""
        client = self._get_openai_client()

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = response.choices[0].message.content or ""
        # Strip Qwen3-style thinking tags (model thinks in <think>...</think>)
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_ms=elapsed_ms,
        )

    async def _complete_anthropic(self, system: str, user: str) -> LLMResponse:
        """Complete using Anthropic API."""
        client = self._get_anthropic_client()

        start = time.perf_counter()
        response = await client.messages.create(
            model=self.config.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = ""
        if response.content:
            content = response.content[0].text

        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_ms=elapsed_ms,
        )

    async def complete_batch(
        self, prompts: list[dict], desc: str = ""
    ) -> list[LLMResponse]:
        """Send multiple completion requests concurrently with progress bar.

        Each prompt dict should have 'system' and 'user' keys.
        """
        from tqdm.asyncio import tqdm_asyncio

        tasks = [
            self.complete(p["system"], p["user"])
            for p in prompts
        ]

        results = await tqdm_asyncio.gather(
            *tasks,
            desc=desc or "LLM calls",
            total=len(tasks),
        )
        return list(results)


# CLI test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM client")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    # Set default models per provider
    model = args.model
    if model is None:
        if args.provider == "openai":
            model = "gpt-4.1-mini"
        elif args.provider == "anthropic":
            model = "claude-haiku-4-5-20251001"
        elif args.provider == "ollama":
            model = "llama3.1:8b"

    config = LLMConfig(
        provider=args.provider,
        model=model,
        base_url=args.base_url,
    )

    client = LLMClient(config)

    async def main():
        resp = await client.complete(
            system="You are a helpful assistant.",
            user=args.prompt,
        )
        print(f"Response: {resp.content}")
        print(f"Input tokens: {resp.input_tokens}")
        print(f"Output tokens: {resp.output_tokens}")
        print(f"Time: {resp.time_ms:.0f}ms")

    asyncio.run(main())
