from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from kosmos.api.routes import router as api_router

app = FastAPI(title="Algorithm Learning Tool API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")

# Mount static files for frontend
static_dir = os.path.join(os.path.dirname(__file__), "web/static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Root route serves index.html
@app.get("/")
async def read_root():
    try:
        with open(os.path.join(static_dir, "index.html"), "r") as f:
            html_content = f.read()

        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not built yet")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
