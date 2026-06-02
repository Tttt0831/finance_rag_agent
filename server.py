"""
server.py - 文档上传 Web 服务（FastAPI）
提供 REST API 接口，可独立部署或与 Gradio 并行运行
"""

import os
import shutil
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from services.vector_store import vector_store_service
from utils.logger import get_logger
from utils.path import get_file_md5
from config import settings

logger = get_logger(__name__)

app = FastAPI(
    title="财智助手 - 文档管理 API",
    description="金融知识库文档上传与管理接口",
    version="1.0.0",
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls"}


@app.get("/health")
def health_check():
    """健康检查接口。"""
    return {"status": "ok", "service": "finance-agent-api"}


@app.post("/upload", summary="上传文档到知识库")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    上传一个或多个文档，自动加载到向量知识库。
    支持格式：PDF、TXT、Markdown、CSV、Excel
    """
    results = []

    for file in files:
        # 检查文件类型
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"不支持的文件类型 {ext}，支持: {list(ALLOWED_EXTENSIONS)}",
            })
            continue

        # 保存文件
        save_path = os.path.join(settings.upload_dir, file.filename)
        try:
            content = await file.read()
            with open(save_path, "wb") as f:
                f.write(content)

            # 入库
            result = vector_store_service.add_file(save_path)
            results.append({
                "filename": file.filename,
                "status": result["status"],
                "chunks": result.get("chunks", 0),
                "md5": result.get("md5", ""),
            })
            logger.info(f"文件处理完成: {file.filename} -> {result['status']}")

        except Exception as e:
            logger.error(f"文件处理失败: {file.filename}, {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e),
            })

    return JSONResponse(content={"results": results})


@app.get("/knowledge/search", summary="搜索知识库")
def search_knowledge(q: str, k: int = 4):
    """
    在知识库中搜索相关文档片段。

    Args:
        q: 搜索查询
        k: 返回结果数量（1-10）
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")
    k = max(1, min(k, 10))

    docs = vector_store_service.similarity_search(q, k=k)
    return {
        "query": q,
        "count": len(docs),
        "results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ],
    }


@app.get("/uploaded-files", summary="获取已上传文件列表")
def list_uploaded_files():
    """列出 uploads 目录中的所有文件。"""
    files = []
    for filename in os.listdir(settings.upload_dir):
        if filename.startswith("."):
            continue
        filepath = os.path.join(settings.upload_dir, filename)
        files.append({
            "filename": filename,
            "size_kb": round(os.path.getsize(filepath) / 1024, 2),
        })
    return {"files": files, "count": len(files)}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
