from typing import Literal

from pydantic import BaseModel


class CacheControl(BaseModel):
    type: Literal["ephemeral"]
