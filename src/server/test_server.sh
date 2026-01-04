#!/usr/bin/bash

# # test qwen3-vl-32b
curl -X POST \
  "https://example.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer " \
  -d'{
  "model": "Qwen3-VL-32B-Instruct",
  "stream": false,
  "messages": [
  {"role":"system",
  "content":[
    {"type": "text", "text": "You are a helpful assistant."}]},
  {
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://fuss10.elemecdn.com/e/5d/4a731a90594a4af544c0c25941171jpeg.jpeg"}},
      {"type": "text", "text": "图中描绘的是什么内容?"}
    ]
  }]
}'

# test openai_proxy
curl -X POST \
  "127.0.0.1:61217/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API KEY>" \
  -d'{
  "model": "azure-gpt-4o",
  "stream": false,
  "messages": [
  {"role":"system",
  "content":[
    {"type": "text", "text": "You are a helpful assistant."}]},
  {
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://fuss10.elemecdn.com/e/5d/4a731a90594a4af544c0c25941171jpeg.jpeg"}},
      {"type": "text", "text": "图中描绘的是什么内容?"}
    ]
  }]
}'

# test openai_proxy qwen qwen3_32b qwen2.5-vl-7b
# curl -X POST \
#   "127.0.0.1:61217/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer <API KEY>" \
#   -d'{
#   "model": "qwen2.5-vl-7b",
#   "messages": [
#   {
#     "role": "user",
#     "content": [
#       {"type": "image_url", "image_url": {"url": "https://fuss10.elemecdn.com/e/5d/4a731a90594a4af544c0c25941171jpeg.jpeg"}},
#       {"type": "text", "text": "图中描绘的是什么内容?"}
#     ]
#   }]
# }'

# id:chatcmpl-CmxEebtzsvod2Ls6eM84IEbOlvnN6
# data: {"choices": [{"content_filter_results": {}, "delta": {}, "finish_reason": "stop", "index": 0}], "created": 1765783480, "endFilter": false, "id": "chatcmpl-CmxEebtzsvod2Ls6eM84IEbOlvnN6", "model": "gpt-4o-2024-08-06", "object": "chat.completion.chunk"}
# retry:3000
# id:[DONE]
# data:[DONE]
# retry:3000