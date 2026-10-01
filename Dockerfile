FROM python:3.12-slim

WORKDIR /app

# 运行依赖在**构建阶段**一次性装齐：pip 安装失败（网络错误、版本无法解析、
# 包损坏等）会以非零退出直接中断 docker build，绝不带到运行时；
# 容器启动时不再联网安装，冷启动直接使用镜像内依赖。
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && python -c "import fastapi, uvicorn, pydantic; from ortools.sat.python import cp_model"

# 应用代码进镜像后，构建期再做字节码编译 + 完整导入校验：
# 应用自身或其依赖的任何导入/语法问题都在构建阶段明确报错。
COPY app /app/app
RUN python -m compileall -q app && python -c "import app.main"

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /data

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
