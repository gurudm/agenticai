from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os

os.environ["ANTHROPIC_API_KEY"] = os.environ.get("CLAUDEKEY")
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """you are a senior banking analyst at a top tier investment bank.
you analyse financial documents and provide precise, structured insights.
Always respond in this exact JSON format :
{{
    "summary": "A brief summary",
    "key_metrics": ["metric1", "metric2"],
    "risk_factors": ["risk1", "risk2"],
    "confidence": "high/medium/low",
    "confidence_score": 1 to 10
}}
Avoid speculations, Return only JSON and no extra text"""),
("human", "{input}")
])


chain = prompt | llm

# # Run the chain 
# result = chain.invoke({
#     "input": """Q3 revenue was $2.3B, up 12% YoY. 
# Non-performing loans increased to 3.2%. 
# Capital adequacy ratio is 11.8%."""
# })

# print(result.content)

scenarios = [
    "Revenue down 5%, NPL ratio 2.1%, CAR 14.2%",
    "Revenue up 20%, NPL ratio 5.8%, CAR 9.1%",
    "Revenue flat, NPL ratio 1.8%, CAR 16.5%"
]

for scenario in scenarios:
    result = chain.invoke({"input": scenario})
    print(f"Scenario: {scenario}")
    print(f"Analysis: {result.content}\n")
    print("-" * 50)