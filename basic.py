### This is a simple example of how to use the OpenAI API to generate a response
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)
print(response.output_text)


### This is a simple example of how to use the Claude API to generate a response
import anthropic
client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    temperature=1,
    system="You are a world-class poet. Respond only with short poems.",
    messages=[{
            "role": "user",
            "content": [{
                    "type": "text",
                    "text": "Why is the ocean salty?"
                }]
        }]
)
print(message.content)
