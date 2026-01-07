import requests
import json
from typing import List, Dict, Optional, Union
import time
from functools import wraps
import threading
import traceback
from loguru import logger
from tqdm import tqdm
import re, sys, os
from typing import Union, Tuple, Optional
import httpx
from openai import OpenAI, APIConnectionError, APITimeoutError


def retry(exception_to_check, tries=3, delay=5, backoff=1):
    """
    Decorator used to automatically retry a failed function. Parameters:

    exception_to_check: The type of exception to catch.
    tries: Maximum number of retry attempts.
    delay: Waiting time between each retry.
    backoff: Multiplicative factor to increase the waiting time after each retry.
    """

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exception_to_check as e:
                    print(f"{str(e)}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


def timeout_decorator(timeout):
    class TimeoutException(Exception):
        pass

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException("Function call timed out")]  # Nonlocal mutable variable

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e  # type: ignore

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                print(f"Function {func.__name__} timed out, retrying...")
                return wrapper(*args, **kwargs)
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


class GPT:
    def __init__(
        self,
        base_url: str = os.getenv("OPENAI_BASE_URL", ""),
        appkey=os.getenv("OPENAI_API_KEY", "EMPTY"),
        model="",
        stream=False,
        temperature=0.2,
        if_reasoner=False,
    ):
        self.base_url = base_url
        self.appkey = appkey
        self.model = model
        self.stream = stream
        self.temperature = temperature
        self.if_reasoner = if_reasoner

        reasoner_models = ["deepseek-reasoner", "qvq-max", "qwq-32b", "qwq-plus"]
        stream_models = ["qvq-max", "qwq-32b", "qwq-plus"]

        if self.model in reasoner_models:
            self.if_reasoner = True
        if self.model in stream_models:
            self.stream = True

        # proxies = "socks5://127.0.0.1:61107"

        # http_client = httpx.Client(
        #     proxies=proxies, # type: ignore
        #     transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        # )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.appkey if self.appkey else "EMPTY",
            # http_client=http_client,
            max_retries=0,
            timeout=180.0,
        )

    @timeout_decorator(timeout=3600)
    @retry(exception_to_check=Exception, tries=3, delay=5, backoff=1)
    def send_chat_request(self, messages) -> Tuple[str, str, str, str, dict]:
        start_time = time.time()
        formatted_messages = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages

        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": 8192,
            "stream": self.stream,
        }

        result = ""
        reasoning_content = ""
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            if not self.stream:
                response = self.client.chat.completions.create(**params)
                message = response.choices[0].message
                full_content = message.content or ""
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                    result = full_content
                else:
                    reasoning_content, result = extract_content_outside_think(full_content)
                if response.usage:
                    token_usage = response.usage.model_dump()

            else:
                response = self.client.chat.completions.create(**params)
                full_content_accumulator = ""
                reasoning_accumulator = ""
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_content_accumulator += delta.content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_accumulator += delta.reasoning_content
                    if hasattr(chunk, "usage") and chunk.usage:
                        token_usage = chunk.usage.model_dump()
                if reasoning_accumulator:
                    reasoning_content = reasoning_accumulator
                    result = full_content_accumulator
                else:
                    reasoning_content, result = extract_content_outside_think(full_content_accumulator)

        except APIConnectionError:
            logger.warning("Api connection failed，retrying...")
            raise
        except Exception as e:
            logger.warning("An exception occurred during the request.")
            traceback.print_exc()
            raise e

        # end_time = time.time()
        # logger.info(f"耗时：{(end_time - start_time):.2f}s")

        return result, None, reasoning_content, None, token_usage  # type: ignore


def extract_content_outside_think(text):
    """split think and response

    Parameters
    ----------
    text : _type_
        origin response

    Returns
    -------
    _type_
        reasoning content, result
    """
    inside_pattern = r"<think>(.*?)</think>"
    inside_content = re.findall(inside_pattern, text, flags=re.DOTALL)
    outside_pattern = r"<think>.*?</think>"
    outside_content_raw = re.split(outside_pattern, text, flags=re.DOTALL)
    cleaned_parts = (part.strip() for part in outside_content_raw if part and part.strip())
    outside_content_string = " ".join(cleaned_parts)
    return inside_content, outside_content_string
