"""FastAPI 应用入口：打分台后端。

启动：img-iter-web（pyproject scripts）或 python -m img_iter_agent.web.app
默认 http://localhost:8765 —— 前后端解耦：服务静态前端 + JSON 接口。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from .routers import (
    config as config_router,
)
from .routers import (
    loops as loops_router,
)
from .routers import (
    overview as overview_router,
)
from .routers import (
    scoring as scoring_router,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="img-iter-agent 打分台", version="0.1.0")

    # 路由
    app.include_router(overview_router.router, prefix="/api", tags=["overview"])
    app.include_router(loops_router.router, prefix="/api", tags=["loops"])
    app.include_router(scoring_router.router, prefix="/api", tags=["scoring"])
    app.include_router(config_router.router, prefix="/api", tags=["config"])

    # 图片代理：处理 reference(绝对路径) vs output(相对 run 目录) 的差异
    @app.get("/api/static/img")
    def serve_img(path: str, loop: str | None = None) -> FileResponse:
        """返回一张图。path 可以是绝对路径，或相对 loop 目录的路径（此时 loop 必填）。"""
        p = Path(path)
        if not p.is_absolute() and loop is not None:
            p = get_settings().run_dir(loop) / path
        p = p.resolve()
        # 安全校验：必须在 data_root 或 benchmarks 下
        settings = get_settings()
        allowed_roots = [settings.data_root.resolve()]
        try:
            p.relative_to(settings.data_root.resolve())
        except ValueError:
            # 允许 benchmarks 在 data_root 下，这里 data_root 已覆盖；
            # 若仍越界则拒绝
            ok = False
            for root in allowed_roots:
                try:
                    p.relative_to(root)
                    ok = True
                    break
                except ValueError:
                    continue
            if not ok:
                raise HTTPException(status_code=403, detail="路径越界")
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(str(p))

    # 健康检查
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "data_root": str(get_settings().data_root)}

    # 静态前端（SPA）：未匹配 API 的都回 index.html
    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()


def main() -> None:
    """脚本入口：img-iter-web。"""
    import uvicorn

    uvicorn.run(
        "img_iter_agent.web.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
