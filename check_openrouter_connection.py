
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found in your .env file."
    )

client = OpenAI(
    base_url="https://openai.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="openai/gpt-5-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "In one or two sentences, explain what a "
                "what are my rights as a woman"
            ),
        }
    ],
    max_tokens=500,
)

choice = response.choices[0]
message = choice.message

print("Openai connection successful!")
print("Model:", response.model)
print("Finish reason:", choice.finish_reason)
print("\nModel response:")

if message.content:
    print(message.content)
else:
    print("No text content was returned.")

    if getattr(message, "refusal", None):
        print("Refusal:", message.refusal)

    if choice.finish_reason == "length":
        print("The output-token limit may have been reached.")

    print("\nFull response for debugging:")
    print(response.model_dump_json(indent=2))
