from fastapi.routing import APIRouter

from bedrock_be.web.api import monitoring

api_router = APIRouter()
api_router.include_router(monitoring.router)
