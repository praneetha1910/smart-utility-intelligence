"""Health-check route."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {"status": "ok"}
