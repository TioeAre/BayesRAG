import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config

from functools import partial
from lightrag.rerank import cohere_rerank
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

max_tokens = int(os.getenv("OPENAI_LLM_MAX_TOKENS", 4096))


def llm_model_func_qwen3(prompt, system_prompt=None, history_messages=[], **kwargs):
    if project_config.LLM_MODEL_NAME.lower().startswith("qwen3-vl"):
        return openai_complete_if_cache(
            project_config.LLM_MODEL_NAME,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=project_config.API_KEY,
            base_url=project_config.QWEN3_VL_BASE_URL,
            openai_client_configs={
                "timeout": 3600,
            },
            extra_body={
                "chat_template_kwargs": {"enable_thinking": project_config.ENABLE_THINK},
            },
            max_tokens=max_tokens,
            max_completion_tokens=max_tokens,
            **kwargs,
        )

    return openai_complete_if_cache(
        project_config.LLM_MODEL_NAME,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=project_config.API_KEY,
        base_url=project_config.UNI_BASE_URL,
        openai_client_configs={
            "timeout": 3600,
        },
        extra_body={
            "chat_template_kwargs": {"enable_thinking": project_config.ENABLE_THINK},
        },
        max_tokens=max_tokens,
        max_completion_tokens=max_tokens,
        **kwargs,
    )


def llm_model_func_gpt4o(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
        project_config.GPT4o_MODEL_NAME,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=project_config.API_KEY,
        base_url=project_config.UNI_BASE_URL,
        openai_client_configs={"timeout": 3600},
        **kwargs,
    )


def vision_model_func_qwen3vl(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
):
    if messages:
        return openai_complete_if_cache(
            "Qwen3-VL-32B-Instruct",
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=project_config.API_KEY,
            base_url=project_config.QWEN3_VL_BASE_URL,
            openai_client_configs={
                "timeout": 3600,
            },
            extra_body={
                "chat_template_kwargs": {"enable_thinking": project_config.ENABLE_THINK},
            },
            max_tokens=max_tokens,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
    elif image_data:
        return openai_complete_if_cache(
            "Qwen3-VL-32B-Instruct",
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                (
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt}
                ),
            ],
            api_key=project_config.API_KEY,
            base_url=project_config.QWEN3_VL_BASE_URL,
            openai_client_configs={
                "timeout": 3600,
            },
            extra_body={
                "chat_template_kwargs": {"enable_thinking": project_config.ENABLE_THINK},
            },
            max_tokens=max_tokens,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
    else:
        return llm_model_func_qwen3(prompt, system_prompt, history_messages, **kwargs)


def vision_model_func_gpt4o(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
):
    if messages:
        return openai_complete_if_cache(
            project_config.GPT4o_MODEL_NAME,
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=project_config.API_KEY,
            base_url=project_config.UNI_BASE_URL,
            **kwargs,
        )
    elif image_data:
        return openai_complete_if_cache(
            project_config.GPT4o_MODEL_NAME,
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                (
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt}
                ),
            ],
            api_key=project_config.API_KEY,
            base_url=project_config.UNI_BASE_URL,
            **kwargs,
        )
    else:
        return llm_model_func_gpt4o(prompt, system_prompt, history_messages, **kwargs)


embedding_func = EmbeddingFunc(
    embedding_dim=project_config.EMBEDDING_DIM,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model=(
            project_config.EMBEDDING_MODEL_NAME.split("vllm-")[1]
            if project_config.EMBEDDING_MODEL_NAME.startswith("vllm")
            else project_config.EMBEDDING_MODEL_NAME
        ),
        api_key=project_config.API_KEY,
        base_url=project_config.EMBEDDING_BASE_URL_RAGANYTHING,
    ),
)

rerank_model_func = partial(
    cohere_rerank,
    model=(
        project_config.RERANK_MODEL_NAME.split("vllm-")[1]
        if project_config.RERANK_MODEL_NAME.startswith("vllm")
        else project_config.RERANK_MODEL_NAME
    ),
    api_key=os.getenv("RERANK_BINDING_API_KEY", "None"),
    base_url=project_config.RERANKER_BASE_URL_VEC,
    # enable_chunking=os.getenv("RERANK_ENABLE_CHUNKING", "false").lower() == "true",
    # max_tokens_per_doc=int(os.getenv("RERANK_MAX_TOKENS_PER_DOC", "4096")),
)
