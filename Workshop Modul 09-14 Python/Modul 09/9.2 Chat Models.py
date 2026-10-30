
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# GPT melalui OpenRouter
llm_oai = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["OPENROUTER_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# Model melalui Xkiro dengan format Anthropic
llm_ant = ChatAnthropic(
    model="qwen/qwen3.8-max:free",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["XKIRO_KEY"],
    base_url="https://api.xkiro.com",
)

# Kedua model menerima jenis pesan yang sama
messages = [
    SystemMessage(
        content="You are a concise technical assistant."
    ),
    HumanMessage(
        content="What is LangChain Expression Language?"
    ),
]

# Invoke
response = llm_ant.invoke(messages)
print(response.content)

# Stream
for chunk in llm_ant.stream(messages):
    print(chunk.content, end="", flush=True)
print()

# Batch
responses = llm_ant.batch([
    [HumanMessage(content="What is RAG?")],
    [HumanMessage(content="What is a vector database?")],
])

for r in responses:
    print(r.content[:80])