import gradio as gr
import requests
import json
import logging
import re

# 设置日志模版
# Configure logging template
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 后端服务接口地址
# Backend service API URL
url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 默认流式输出 True or False
# Default streaming output: True or False
stream_flag = False

def send_message(user_message, history):
    # 封装请求的参数
    # Prepare request parameters
    data = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": "123",
        "conversationId": "123"
    }

    # 等待LLM产生token前的等待状态
    # Display waiting message before LLM generates a response
    history = history + [["user", user_message], ["assistant", "Generating response..."]]
    yield history

    # 对deepseek-r1模型进行格式化处理
    # Format response for deepseek-r1 model
    def format_response(full_text):
        # 精确替换 <think> 和 </think>
        # Replace <think> and </think> with formatted text
        formatted_text = full_text
        formatted_text = re.sub(r'<think>', '**Thinking process**:\n', formatted_text)
        formatted_text = re.sub(r'</think>', '\n\n**Final response**:\n', formatted_text)
        return formatted_text.strip()

    # 流式输出
    # Streaming output
    if stream_flag:
        assistant_response = ""
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            logger.info(f"Received an empty string, skipping...")
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    # 实时格式化响应
                                    # Format response in real time
                                    formatted_content = format_response(content)
                                    logger.info(f"Received data: {formatted_content}")
                                    assistant_response += formatted_content
                                    updated_history = history[:-1] + [["assistant", assistant_response]]
                                    yield updated_history
                                if response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    logger.info(f"JSON data reception completed")
                                    break
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parsing error: {e}")
                                yield history[:-1] + [["assistant", "Error parsing response, please try again later."]]
                                break
                        else:
                            logger.info(f"Invalid JSON format: {json_str}")
                    else:
                        logger.info(f"Received an empty line")
                else:
                    logger.info("Streaming response ended unexpectedly")
                    yield history[:-1] + [["assistant", "Did not receive a complete response."]]
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            yield history[:-1] + [["assistant", "Request failed, please try again later."]]

    # 非流式输出
    # Non-streaming output
    else:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_json = response.json()
        assistant_content = response_json['choices'][0]['message']['content']
        # 格式化响应
        # Format response
        formatted_content = format_response(assistant_content)
        logger.info(f"Non-streaming output, formatted content: {formatted_content}")
        updated_history = history[:-1] + [["assistant", formatted_content]]
        yield updated_history


# Gradio 前端界面
# Gradio frontend interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chat Dialog")

    with gr.Row():
        with gr.Column(scale=8):
            message = gr.Textbox(label="Enter Message", placeholder="Type your message here")
        with gr.Column(scale=2):
            send = gr.Button("Send")

    # 发送按钮点击后，处理模型的响应
    # Handle model response when the send button is clicked
    send.click(send_message, [message, chatbot], chatbot)
    message.submit(send_message, [message, chatbot], chatbot)
    send.click(lambda: "", None, message)
    message.submit(lambda: "", None, message)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
