from enum import Enum


class ModelName(str, Enum):
    CLAUDE_3_5_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_5_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    LLAMA_3_2_1B = "us.meta.llama3-2-1b-instruct-v1:0"
    LLAMA_3_2_3B = "us.meta.llama3-2-3b-instruct-v1:0"
    LLAMA_3_2_11B = "us.meta.llama3-2-11b-instruct-v1:0"
    LLAMA_3_2_90B = "us.meta.llama3-2-90b-instruct-v1:0"
    TITAN_LITE = "amazon.titan-text-lite-v1:0"
    TITAN_EXPRESS = "amazon.titan-text-express-v1:0"
    TITAN_PREMIER = "amazon.titan-text-premier-v1:0"
    MISTRAL = "mistral.mistral-7b-instruct-v0:2"
    JURASSIC = "ai21.j2-ultra-v1:0"
