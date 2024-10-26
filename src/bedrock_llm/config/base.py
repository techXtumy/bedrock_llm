from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0)
    exponential_backoff: bool = Field(default=True)