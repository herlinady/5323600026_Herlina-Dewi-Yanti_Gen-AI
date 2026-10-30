from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["OPENROUTER_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


# Pydantic schema for structured output
class ModelComparison(BaseModel):
    models: List[str] = Field(
        description="Model names compared"
    )
    best_for_cost: str = Field(
        description="Most cost-effective model"
    )
    best_for_quality: str = Field(
        description="Highest quality model"
    )
    recommendation: str = Field(
        description="Overall recommendation"
    )


# Structured output
structured_llm = llm.with_structured_output(ModelComparison)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI model comparison expert."),
    (
        "human",
        "Compare these models for a startup on a budget: {models}",
    ),
])

chain = prompt | structured_llm

result: ModelComparison = chain.invoke({
    "models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"
})

print(f"Best for cost: {result.best_for_cost}")
print(f"Best for quality: {result.best_for_quality}")
print(f"Recommendation: {result.recommendation}")