from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import buildings, inspections, windows, pages, dashboard

app = FastAPI(
    title="Window Inspection System",
    description="드론/스마트폰 촬영 영상 기반 건물 창문 오염도 분석 시스템",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

app.include_router(pages.router)
app.include_router(buildings.router)
app.include_router(inspections.router)
app.include_router(windows.router)
app.include_router(dashboard.router)


@app.on_event("startup")
def on_startup():
    init_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
