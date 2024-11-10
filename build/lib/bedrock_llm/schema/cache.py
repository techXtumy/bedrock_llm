from pydantic import BaseModel
from typing import Literal

class CacheControl(BaseModel):
    type: Literal["ephemeral"]