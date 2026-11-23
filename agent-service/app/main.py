import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger,configure_logging
from app.core.config import settings
from app.core.exceptions import ApplicationError

from app.modules.chat_thread.exceptions import ChatThreadNotFoundError
from app.infra.checkpointer import close_checkpointer, init_checkpointer

from app.agents.insurance_advisor import init_insurance_agent
from app.agents.orchestrator import init_insurance_orchestrator
# 日志配置
configure_logging(settings.logging.level)

# 获取日志记录
logger = get_logger(__name__)

# 生命周期管理（预留）
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"【{settings.app.name}】应用启动中...")
    from app.infra.database import check_database, close_database
    from app.rag import close_rag  # 导入模块就会初始化

    try:
        # 1.检查业务数据库连接
        await check_database()
        # 2.初始化checkpointer
        checkpointer = await init_checkpointer()
        # 3.初始化保险顾问Agent（现在替换为graph了）,并保存在FastAPI的全局state当中
        app.state.agent = init_insurance_orchestrator(checkpointer)
        yield
    finally:
        logger.info("应用关闭中... ")
        # 4.关闭RAG资源
        close_rag()
        # 5.关闭Checkpointer连接池
        await close_checkpointer()
        # 6.关闭SQLAlchemy连接池
        await close_database()


# 初始化fastapi
app=FastAPI(
    title=settings.app.name,
    debug=settings.app.debug,
    lifespan=lifespan
)

from app.modules.product.router import router as product_router
app.include_router(product_router)

from app.modules.chat_thread.router import router as chat_thread_router
app.include_router(chat_thread_router)

from app.modules.chat.router import router as chat_router
app.include_router(chat_router)

# 测试接口,健康检查
@app.get("/healthy",summary="健康检查接口")
async def healthy()->dict[str, str]:
    logger.info("健康监测接口正常调用.")
    return {"status": "ok"}

# 统一处理异常信息，发现是ChatThreadNotFoundError，返回对应的错误状态码和错误信息
@app.exception_handler(ApplicationError)
async def handle_application_error(
    request: Request,
    exc: ApplicationError, # 捕获到的异常（这里用自定义异常基类捕获，所有业务异常都会匹配到）
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, # 异常中自带status
        content={
            "code": exc.code,        # 异常中自带code
            "message": exc.message,  # 异常中自带message
        },
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=False,
        loop= "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto",
    )