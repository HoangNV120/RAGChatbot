from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from typing import Dict, List, Any, Optional, Iterator, AsyncIterator
import requests
import json
import logging
import aiohttp

logger = logging.getLogger(__name__)

class MultiModelChatAPI(BaseChatModel):
    """
    Custom Chat Model that inherits from BaseChatModel to use a custom API endpoint
    with support for multiple models and API keys.
    """
    
    # Required parameters
    api_url: str
    model_name: str = "default"
    api_key: str = ""
    
    # Optional parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: Optional[int] = 60
    streaming: bool = False
    
    # API key mapping for different models
    model_api_keys: Dict[str, str] = {}
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return f"custom-multimodel-{self.model_name}"
    
    def _get_api_key(self) -> str:
        """Get the appropriate API key based on the model."""
        if self.model_name in self.model_api_keys:
            return self.model_api_keys[self.model_name]
        return self.api_key
    
    def _convert_messages_to_dict(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages to the format expected by the API."""
        message_dicts = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, ChatMessage):
                role = message.role
            else:
                raise ValueError(f"Got unknown message type: {type(message)}")
            
            message_dict = {"role": role, "content": message.content}
            message_dicts.append(message_dict)
        
        return message_dicts
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from the model."""
        message_dicts = self._convert_messages_to_dict(messages)
        
        # Prepare the payload
        payload = {
            "model": self.model_name,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            payload[key] = value
        
        headers = {}
        if self._get_api_key():
            headers["Authorization"] = f"Bearer {self._get_api_key()}"
        headers["Content-Type"] = "application/json"
        
        try:
            print(f"Sending request to API using model: {self.model_name}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Get the model used in the response
            model_used = result.get("model", self.model_name)
            print(f"Received response from model: {model_used}")
            
            # Extract the assistant's message
            if "choices" in result and len(result["choices"]) > 0:
                # OpenAI-like response format
                message = result["choices"][0]["message"]
                ai_message = AIMessage(content=message.get("content", ""))
                
                # Add model info to message metadata
                ai_message.additional_kwargs["model"] = model_used
            elif "response" in result:
                # Custom response format
                ai_message = AIMessage(content=result["response"])
                
                # Add model info to message metadata
                ai_message.additional_kwargs["model"] = model_used
            else:
                # Fallback
                ai_message = AIMessage(content=str(result))
                ai_message.additional_kwargs["model"] = model_used
            
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
            
        except Exception as e:
            print(f"Error calling API with model {self.model_name}: {e}")
            raise
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate a chat response from the model."""
        message_dicts = self._convert_messages_to_dict(messages)
        
        # Prepare the payload
        payload = {
            "model": self.model_name,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            payload[key] = value
        
        headers = {}
        if self._get_api_key():
            headers["Authorization"] = f"Bearer {self._get_api_key()}"
        headers["Content-Type"] = "application/json"
        
        try:
            print(f"Sending async request to API using model: {self.model_name}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    
                    # Check content type before parsing JSON
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('application/json'):
                        logger.warning(f"Received non-JSON response with content type: {content_type}")
                        # Try to get text content for better error message
                        text_content = await response.text()
                        error_msg = f"API returned non-JSON content type: {content_type}"
                        print(f"{error_msg}. First 100 chars of response: {text_content[:100]}")
                        ai_message = AIMessage(content=f"Error: The API endpoint returned HTML or non-JSON content instead of a proper API response. Please check the API URL configuration.")
                        return ChatResult(generations=[ChatGeneration(message=ai_message)])
                    
                    result = await response.json()
                    
                    # Get the model used in the response
                    model_used = result.get("model", self.model_name)
                    print(f"Received async response from model: {model_used}")
                    
                    # Extract the assistant's message
                    if "choices" in result and len(result["choices"]) > 0:
                        # OpenAI-like response format
                        message = result["choices"][0]["message"]
                        ai_message = AIMessage(content=message.get("content", ""))
                        
                        # Add model info to message metadata
                        ai_message.additional_kwargs["model"] = model_used
                    elif "response" in result:
                        # Custom response format
                        ai_message = AIMessage(content=result["response"])
                        
                        # Add model info to message metadata
                        ai_message.additional_kwargs["model"] = model_used
                    else:
                        # Fallback
                        ai_message = AIMessage(content=str(result))
                        ai_message.additional_kwargs["model"] = model_used
                    
                    return ChatResult(generations=[ChatGeneration(message=ai_message)])
                    
        except aiohttp.ContentTypeError as e:
            print(f"Error parsing JSON response for model {self.model_name}: {e}")
            ai_message = AIMessage(content=f"Error: The API returned invalid JSON. Error details: {str(e)}")
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        except Exception as e:
            print(f"Error calling API asynchronously with model {self.model_name}: {e}")
            ai_message = AIMessage(content=f"Error: Failed to communicate with the API. Error details: {str(e)}")
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatResult]:
        """Stream the chat response from the model."""
        if not self.streaming:
            raise ValueError("Streaming is not enabled for this model")
        
        message_dicts = self._convert_messages_to_dict(messages)
        
        # Prepare the payload
        payload = {
            "model": self.model_name,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": True,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            payload[key] = value
        
        headers = {}
        if self._get_api_key():
            headers["Authorization"] = f"Bearer {self._get_api_key()}"
        headers["Content-Type"] = "application/json"
        
        try:
            print(f"Starting stream request to API using model: {self.model_name}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Process the streaming response
            content = ""
            model_used = self.model_name
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        data_str = line_str[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            # Try to extract model information if available
                            if "model" in data and not model_used:
                                model_used = data["model"]
                                print(f"Streaming from model: {model_used}")
                                
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_chunk = delta["content"]
                                    content += content_chunk
                                    
                                    if run_manager:
                                        run_manager.on_llm_new_token(content_chunk)
                                    
                                    ai_message = AIMessage(content=content)
                                    ai_message.additional_kwargs["model"] = model_used
                                    
                                    yield ChatResult(
                                        generations=[ChatGeneration(message=ai_message)]
                                    )
                        except json.JSONDecodeError:
                            print(f"Failed to parse streaming response: {data_str}")
                            
            print(f"Completed streaming from model: {model_used}")
                            
        except Exception as e:
            print(f"Error streaming from API with model {self.model_name}: {e}")
            raise
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResult]:
        """Asynchronously stream the chat response from the model."""
        if not self.streaming:
            raise ValueError("Streaming is not enabled for this model")
        
        message_dicts = self._convert_messages_to_dict(messages)
        
        # Prepare the payload
        payload = {
            "model": self.model_name,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": True,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            payload[key] = value
        
        headers = {}
        if self._get_api_key():
            headers["Authorization"] = f"Bearer {self._get_api_key()}"
        headers["Content-Type"] = "application/json"
        
        try:
            print(f"Starting async stream request to API using model: {self.model_name}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    params={"stream": "true"}
                ) as response:
                    response.raise_for_status()
                    
                    # Process the streaming response
                    content = ""
                    model_used = self.model_name
                    async for line in response.content:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data:"):
                            data_str = line_str[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                # Try to extract model information if available
                                if "model" in data and not model_used:
                                    model_used = data["model"]
                                    print(f"Async streaming from model: {model_used}")
                                    
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content_chunk = delta["content"]
                                        content += content_chunk
                                        
                                        if run_manager:
                                            await run_manager.on_llm_new_token(content_chunk)
                                        
                                        ai_message = AIMessage(content=content)
                                        ai_message.additional_kwargs["model"] = model_used
                                        
                                        yield ChatResult(
                                            generations=[ChatGeneration(message=ai_message)]
                                        )
                            except json.JSONDecodeError:
                                print(f"Failed to parse streaming response: {data_str}")
                    
                    print(f"Completed async streaming from model: {model_used}")
                                
        except Exception as e:
            print(f"Error streaming from API asynchronously with model {self.model_name}: {e}")
            raise
