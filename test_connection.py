from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY")
)

response = client.chat.completions.create(
   model="accounts/fireworks/models/gpt-oss-20b",
    messages=[{"role": "user", "content": "Reply with just: connection successful"}]
)
print(response.choices[0].message.content)