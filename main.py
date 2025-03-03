import os
import re
import uuid
import time
import json
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from llms import get_llm, get_model_type  # ✅ Let `llms.py` decide the model to use
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
# Load environment variables
load_dotenv()

# ✅ 让 llms.py 选择默认的模型类型（openai / ollama）
# ✅ Let `llms.py` determine the default model type (openai / ollama)
llm_type = get_model_type()  
logging.info(f"Current model type: {llm_type}")

# 访问 API 密钥
# Access API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
print(langchain_api_key)

# 设置 API 相关的环境变量
# Set API-related environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# 设置日志
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# prompt 模板文件
# Prompt template files
PROMPT_TEMPLATE_TXT_SYS = "prompt_template_system.txt"
PROMPT_TEMPLATE_TXT_USER = "prompt_template_user.txt"

# ✅ **删除这行手动设置模型类型的代码**（自动读取 `llms.py` 的 `DEFAULT_LLM_TYPE`）
# ✅ **Remove manual model type setting (automatically use `DEFAULT_LLM_TYPE` from `llms.py`)**
# llm_type = "openai"  ❌ **This line makes switching ineffective, removed!**

# API 服务器端口
# API server port
PORT = 8012

# 申明全局变量
# Declare global variable
graph = None

# 定义消息类
# Define message class
class Message(BaseModel):
    role: str
    content: str

# 定义请求和响应数据结构
# Define request and response data structures
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None

# 定义 chatbot 的状态
# Define chatbot state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 创建 LLM 任务执行的 graph
# Create the graph for LLM task execution
def create_graph(llm) -> StateGraph:
    try:
        graph_builder = StateGraph(State)

        # 定义 chatbot 处理消息
        # Define chatbot message processing
        def chatbot(state: State) -> dict:
            return {"messages": [llm.invoke(state["messages"])]}

        # 配置 graph 结构
        # Configure graph structure
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)

    except Exception as e:
        raise RuntimeError(f"Failed to create graph: {str(e)}")

# 可视化 graph 并保存
# Visualize the graph and save it
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        logger.info(f"Graph visualization saved as {filename}")
    except IOError as e:
        logger.info(f"Warning: Failed to save graph visualization: {str(e)}")

# FastAPI 生命周期管理
# FastAPI lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    try:
        logger.info("Initializing model and defining Graph...")
        llm = get_llm(llm_type)  # ✅ **Use `get_model_type()` to select the model**
        graph = create_graph(llm)
        save_graph_visualization(graph)
        logger.info("Initialization completed")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

    yield
    logger.info("Shutting down...")

# FastAPI 启动
# Start FastAPI
app = FastAPI(lifespan=lifespan)

# 聊天接口
# Chat API endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not graph:
        logger.error("Service not initialized")
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        logger.info(f"Received chat request: {request}")

        query_prompt = request.messages[-1].content
        logger.info(f"User question: {query_prompt}")

        config = {"configurable": {"thread_id": request.userId + "@@" + request.conversationId}}
        logger.info(f"User session info: {config}")

        prompt_template_system = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_SYS)
        prompt_template_user = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_USER)
        prompt = [
            {"role": "system", "content": prompt_template_system.template},
            {"role": "user", "content": prompt_template_user.template.format(query=query_prompt)}
        ]

        # 流式响应
        # Streaming response
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                async for message_chunk, metadata in graph.astream({"messages": prompt}, config, stream_mode="messages"):
                    chunk = message_chunk.content
                    yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {'content': chunk},'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {},'finish_reason': 'stop'}]})}\n\n"
            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 非流式响应
        # Non-streaming response
        else:
            events = graph.stream({"messages": prompt}, config)
            for event in events:
                for value in event.values():
                    result = value["messages"][-1].content

            response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=result),
                        finish_reason="stop"
                    )
                ]
            )
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        logger.error(f"Error processing chat request:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 运行 FastAPI 服务器
# Run FastAPI server
if __name__ == "__main__":
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
