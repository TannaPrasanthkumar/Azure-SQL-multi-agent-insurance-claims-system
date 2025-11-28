import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_AISERVICES_ENDPOINT")
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("AZURE_AISERVICES_APIKEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Initialize conversation history with system message
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Provide clear, accurate, and friendly responses.",
    }
]

print("=" * 70)
print("🤖 Azure AI Chat Assistant")
print("=" * 70)
print("Connected successfully! Start chatting below.")
print("Type 'exit', 'quit', or 'bye' to end the conversation.")
print("=" * 70)
print()

# Interactive chat loop
while True:
    # Get user input
    user_input = input("You: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\n👋 Goodbye! Have a great day!")
        break
    
    # Skip empty inputs
    if not user_input:
        print("⚠️  Please enter a message.")
        continue
    
    # Add user message to conversation history
    messages.append({
        "role": "user",
        "content": user_input
    })
    
    try:
        # Get AI response
        print("\n🤔 Assistant is thinking...\n")
        
        response = client.chat.completions.create(
            messages=messages,
            max_completion_tokens=2000,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )
        
        # Extract assistant's message
        assistant_message = response.choices[0].message.content
        
        # Add assistant's response to conversation history
        messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Display the response
        print(f"Assistant: {assistant_message}")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please try again.\n")
        # Remove the last user message if there was an error
        messages.pop()