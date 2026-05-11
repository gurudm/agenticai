import anthropic
import os


client = anthropic.Anthropic(api_key=os.environ.get('CLAUDEKEY'))
conversation_history = []

print("Banking assistant is ready, Type quit to exit.\n")

while True:
    user_input = input('You: ')
    if user_input.lower() == 'quit':
        break

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    response = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=1024,
        system="You are a helpful banking assistance who explains financial concepts clearly",
        messages=conversation_history
    )

    assistant_message = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    print(f"\nAssistant: {assistant_message}\n")

