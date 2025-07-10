from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.match import router as match_router

app = FastAPI(
    title="Citadel Profile Discovery API",
    description="Backend service for profile recommendation and feedback ingestion",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the match router under /api
app.include_router(match_router, prefix="/api")

# Root health-check endpoint
@app.get("/", tags=["health"])
def health_check():
    return {"status": "alive"}