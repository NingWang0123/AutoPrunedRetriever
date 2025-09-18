import json
import aiohttp
import asyncio
from typing import Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            # Increase timeout to accommodate long LLM responses
            timeout = aiohttp.ClientTimeout(
                total=180,      # Total timeout: 3 minutes
                connect=15,     # Connection timeout: 15 seconds
                sock_read=120   # Read timeout: 2 minutes
            )
            # Configure connection pool limits
            connector = aiohttp.TCPConnector(
                limit=10,           # Total connection limit
                limit_per_host=5,   # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    async def ainvoke(self, prompt: str, model: str = "qwen2.5:72b", **kwargs) -> 'OllamaResponse':
        """Asynchronously invoke Ollama API with improved retry and backoff mechanism"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_ctx": kwargs.get("num_ctx", 32768)
            }
        }
        
        # Improved retry logic
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return OllamaResponse(result["message"]["content"])
                    
                    elif response.status in [500, 503]:  # Server errors
                        if attempt < max_retries - 1:
                            # Exponential backoff
                            delay = base_delay * (2 ** attempt)
                            print(f"🔄 Ollama service busy (status {response.status}), retrying after {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_text = await response.text()
                            raise ValueError(f"Ollama service unavailable, status: {response.status}, error: {error_text}")
                    
                    else:
                        error_text = await response.text()
                        raise ValueError(f"Ollama invocation failed, status: {response.status}, error: {error_text}")
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"🔄 Connection error, retrying after {delay} seconds: {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ValueError(f"Failed to connect to Ollama service: {e}")
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


class OllamaResponse:
    def __init__(self, content: str):
        self.content = content

class OllamaWrapper:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
        
    async def ainvoke(self, prompt, config=None):
        return await self.client.ainvoke(prompt, model=self.model_name)
