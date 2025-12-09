import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4"
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize memory - this stores conversation history automatically
memory = ConversationBufferMemory()

# Create conversation chain (combines LLM + memory)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False  # Set to True to see what's happening under the hood
)

def main():
    print("🤖 Chatbot started! Type 'quit' to exit.")
    print("   Type 'history' to see conversation memory.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if user_input.lower() == 'history':
            print("\n📜 Conversation History:")
            print(memory.buffer)
            print()
            continue
        
        if not user_input:
            continue
        
        # Get response - memory is handled automatically!
        response = conversation.predict(input=user_input)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()