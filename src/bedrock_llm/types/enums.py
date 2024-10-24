from enum import Enum


class ModelName(str, Enum):
    CLAUDE_3_5_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_5_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    LLAMA_3_2_1B = "us.meta.llama3-2-1b-instruct-v1:0"
    LLAMA_3_2_3B = "us.meta.llama3-2-3b-instruct-v1:0"
    LLAMA_3_2_11B = "us.meta.llama3-2-11b-instruct-v1:0"
    LLAMA_3_2_90B = "us.meta.llama3-2-90b-instruct-v1:0"
    TITAN_LITE = "amazon.titan-text-lite-v1"
    TITAN_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_PREMIER = "amazon.titan-text-premier-v1:0"
    JAMBA_1_5_MINI = "ai21.jamba-1-5-mini-v1:0"
    JAMBA_1_5_LARGE = "ai21.jamba-1-5-large-v1:0"
    MISTRAL_LARGE_2 = "mistral.mistral-large-2407-v1:0"
    MISTRAL_7B = "mistral.mistral-7b-instruct-v0:2"
