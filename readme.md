Reference: https://github.com/NanGePlus/LangGraphChatBot

1.  You need create a .env to set up your api keys like:
(openai api):
OPENAI_API_KEY=abcd 

(langsmith api)
LANGCHAIN_API_KEY=abcd

2. cd to 01_CHATBOT folder
   Create a virtual environment:
   python3 -m venv .venv

   Activate the virtual environment:
   source .venv/bin/activate

   Install the dependencies:
   pip install -r requirements.txt

3. To change the default model, modify the following line in llms.py:
   Set and switch LLM types
   Default to OpenAI (can be changed to "ollama")
   DEFAULT_LLM_TYPE = "openai"  # Change "openai" to "ollama" if needed
   
   Setup your local deepseek model: check reference video

4. Run the main:
   open terminal
   python3 main.py

5. Run the webUI:
   open another terminal
   python3 webUI.py
   http://127.0.0.1:7860/