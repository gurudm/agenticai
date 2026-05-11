import anthropic
import os

system = """ you are a senior banking analyst at a top tier investment bank.
you analyse financial documents and provide precise, structured insights.
Always respond in this exact JSON format :
{
    "summary": "A brief summary",
    "key_metrics": ["metric1", "metric2"],
    "risk_factors": ["risk1", "risk2"],
    "confidence": "high/medium/low"
}
Avoid speculations, Return only JSON and no extra text"""

#temperature
model="claude-sonnet-4-20250514"
max_tokens=1024
temperature=0

client = anthropic.Anthropic(api_key=os.environ.get("CLAUDEKEY"))
user_input = """Q3 revenue was $2.3B, up 12% YoY. 
Non-performing loans increased to 3.2%. 
Capital adequacy ratio is 11.8%."""

response = client.messages.create(
    model= model,
    max_tokens= max_tokens,
    temperature= temperature,
    system= system,
    messages= [
        {"role": "user", "content": user_input}
    ]
)

print(response.content[0].text)