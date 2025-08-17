from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, AsyncIterator, Tuple
import time, json, logging, asyncio, aiohttp, requests
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

logger = logging.getLogger(__name__)

def _now_ms() -> int:
    return int(time.time() * 1000)

class MultiModelChatAPI(BaseChatModel):
    api_url: str
    api_key: str = ""
    model_name: str = Field(default="default", alias="model")

    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0

    timeout: Optional[int] = 30
    max_retries: int = 2
    stop: Optional[List[str]] = None

    @property
    def _llm_type(self) -> str:
        return "custom-multimodel-chatapi"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_url": self.api_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    # ------------ helpers ------------
    def _messages_to_api(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, (ToolMessage, FunctionMessage)):
                role = "tool"
            else:
                role = "user"
            out.append({"role": role, "content": str(m.content)})
        return out

    def _default_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _build_payload(
            self,
            messages: List[BaseMessage],
            stream: bool = False,
            extra_stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": self._messages_to_api(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": stream,
        }
        stops = extra_stop or self.stop
        if stops:
            payload["stop"] = stops
        payload.update(kwargs or {})
        return payload

    def _apply_client_side_stop(self, text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        for token in stop:
            if not token:
                continue
            idx = text.find(token)
            if idx != -1:
                return text[: idx + len(token)]
        return text

    def _parse_api_response(
            self, data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
        # Non-streaming schema you provided:
        # choices[0].message.content + usage{prompt_tokens, completion_tokens, total_tokens}
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except Exception:
            text = data.get("text") or data.get("content") or ""

        resp_meta = {
            "id": data.get("id"),
            "model_name": data.get("model", self.model_name),
            "created": data.get("created"),
            "system_fingerprint": data.get("system_fingerprint"),
            "object": data.get("object"),
        }
        u = data.get("usage") or {}
        usage = {
            "input_tokens": int(u.get("prompt_tokens", 0)),
            "output_tokens": int(u.get("completion_tokens", 0)),
            "total_tokens": int(u.get("total_tokens", u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))),
        }
        return text, resp_meta, usage

    # ------------ sync ------------
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stream=False, extra_stop=stop, **kwargs)
        headers = self._default_headers()
        start = _now_ms()
        err = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                text, meta, usage = self._parse_api_response(data)
                text = self._apply_client_side_stop(text, stop or self.stop)
                msg = AIMessage(
                    content=text,
                    response_metadata={**meta, "latency_ms": _now_ms() - start},
                    usage_metadata=UsageMetadata(usage),
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])
            except Exception as e:
                err = e
                if attempt >= self.max_retries:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"Exhausted retries: {err}")

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stream=True, extra_stop=stop, **kwargs)
        headers = self._default_headers()
        start = _now_ms()
        buffered = ""

        def _emit(token: str):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token, usage_metadata=UsageMetadata({"input_tokens": 0, "output_tokens": 1, "total_tokens": 1})))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            return chunk

        with requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # plain text fallback
                    token = line
                    finish_reason = None
                else:
                    ch = (obj.get("choices") or [{}])[0]
                    delta = (ch.get("delta") or {})
                    token = delta.get("content", "") or ""
                    finish_reason = ch.get("finish_reason")
                if token:
                    prospective = buffered + token
                    clipped = self._apply_client_side_stop(prospective, stop or self.stop)
                    allowed = clipped[len(buffered):]
                    if allowed:
                        buffered = clipped
                        yield _emit(allowed)
                    if (stop or self.stop) and (clipped != prospective):
                        break
                if finish_reason is not None:
                    # model signaled end
                    break

        meta = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"latency_ms": _now_ms() - start, "model_name": self.model_name},
            )
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=meta)
        yield meta

    # ------------ async ------------
    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stream=False, extra_stop=stop, **kwargs)
        headers = self._default_headers()
        start = _now_ms()
        err = None
        for attempt in range(self.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.api_url, headers=headers, json=payload) as r:
                        if r.status >= 400:
                            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:256]}")
                        data = await r.json()
                text, meta, usage = self._parse_api_response(data)
                text = self._apply_client_side_stop(text, stop or self.stop)
                msg = AIMessage(
                    content=text,
                    response_metadata={**meta, "latency_ms": _now_ms() - start},
                    usage_metadata=UsageMetadata(usage),
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])
            except Exception as e:
                err = e
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"Exhausted retries: {err}")

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stream=True, extra_stop=stop, **kwargs)
        headers = self._default_headers()
        start = _now_ms()
        buffered = ""

        def mk_chunk(token: str) -> ChatGenerationChunk:
            return ChatGenerationChunk(
                message=AIMessageChunk(
                    content=token,
                    usage_metadata=UsageMetadata({"input_tokens": 0, "output_tokens": 1, "total_tokens": 1}),
                )
            )

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.api_url, headers=headers, json=payload) as r:
                if r.status >= 400:
                    raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:256]}")
                async for raw_bytes in r.content:
                    if not raw_bytes:
                        continue
                    line = raw_bytes.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break

                    token, finish_reason = "", None
                    try:
                        obj = json.loads(line)
                        ch = (obj.get("choices") or [{}])[0]
                        delta = (ch.get("delta") or {})
                        token = delta.get("content", "") or ""
                        finish_reason = ch.get("finish_reason")
                    except json.JSONDecodeError:
                        token = line

                    if token:
                        prospective = buffered + token
                        clipped = self._apply_client_side_stop(prospective, stop or self.stop)
                        allowed = clipped[len(buffered):]
                        if allowed:
                            buffered = clipped
                            chunk = mk_chunk(allowed)
                            if run_manager:
                                await run_manager.on_llm_new_token(allowed, chunk=chunk)
                            yield chunk
                        if (stop or self.stop) and (clipped != prospective):
                            break
                    if finish_reason is not None:
                        break

        meta = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"latency_ms": _now_ms() - start, "model_name": self.model_name},
            )
        )
        if run_manager:
            await run_manager.on_llm_new_token("", chunk=meta)
        yield meta
