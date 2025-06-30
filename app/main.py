from fastapi import FastAPI
from app.routes.match import router as match_router

app = FastAPI(
    title="Citadel Profile Discovery API",
    description="Backend service for profile recommendation and feedback ingestion",
    version="1.0.0"
)

# Mount the match router under /api
app.include_router(match_router, prefix="/api")

# Root health-check endpoint
@app.get("/", tags=["health"])
def health_check():
    return {"status": "alive"}