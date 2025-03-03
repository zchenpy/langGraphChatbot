# apiTest.py
import requests
import json
import logging
from llms import get_model_type 


# Set up the logging template
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}


# Default to non-streaming output: True or False
stream_flag = False


input_text = "I want a budget-friendly adventure under $2,000. What do you recommend?"
#input_text = "Do you have a luxury vacation with an overwater bungalow?"
#input_text = "I love wildlife! Which package do you suggest?"
#input_text = "Can you recommend a trip for seeing the Northern Lights?"
#input_text = "I'm a food lover! What tour includes a wine tasting experience?"


# Encapsulate request parameters
data = {
    "messages": [{"role": "user", "content": input_text}],
    "stream": stream_flag,
    "userId": "456",
    "conversationId": "456"
}


# Print or log model information
model_info = {
    "model_type": get_model_type(),  # Get the model type from llms.py
    "api_endpoint": url,
    "streaming": stream_flag
}
logger.info(f"Model Information: {model_info}")


# Process streaming output
if stream_flag:
    full_response = ""
    try:
        with requests.post(url, stream=True, headers=headers, data=json.dumps(data)) as response:
            for line in response.iter_lines():
                if line:
                    json_str = line.decode('utf-8').strip("data: ")
                    # Check if the string is empty or invalid
                    if not json_str:
                        logger.info("Received an empty string, skipping...")
                        continue
                    # Ensure the string is in valid JSON format
                    if json_str.startswith('{') and json_str.endswith('}'):
                        try:
                            data = json.loads(json_str)
                            if 'delta' in data['choices'][0]:
                                delta_content = data['choices'][0]['delta'].get('content', '')
                                full_response += delta_content
                                logger.info(f"Streaming output, response part is: {delta_content}")
                            if data['choices'][0].get('finish_reason') == "stop":
                                logger.info("Finished receiving JSON data")
                                logger.info(f"Complete response is: {full_response}")
                        except json.JSONDecodeError as e:
                            logger.info(f"JSON parse error: {e}")
                    else:
                        logger.info(f"Invalid JSON format: {json_str}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

# Process non-streaming output
else:
    # Send POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # logger.info(f"Received raw response content: {response.json()}\n")
    content = response.json()['choices'][0]['message']['content']
    logger.info(f"Non-streaming output, response content is: {content}\n")
