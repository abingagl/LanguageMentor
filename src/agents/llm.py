import json
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def is_openai_model(model_name):
    return any(keyword in model_name.lower() for keyword in ['openai', 'gpt'])

class ChatModelManager:
    def __init__(self):
        self.llm = self.load_model()

    def load_model(self):
        config = load_config()
        chatbot_config = config.get('llm_config', {})
        
        model_name = chatbot_config.get('model', 'llama3.1:8b-instruct-q8_0')
        max_tokens = chatbot_config.get('max_tokens', 8192)
        temperature = chatbot_config.get('temperature', 0.8)
        
        chat_model = ChatOpenAI if is_openai_model(model_name) else ChatOllama
        
        return chat_model(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
