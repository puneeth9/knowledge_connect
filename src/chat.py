from src.vector_store import clear_and_reindex
from src.chain import create_rag_chain


def display_chunks(sources):
    """Display retrieved chunks in a formatted way."""
    print("\n" + "=" * 60)
    print("📄 RETRIEVED CHUNKS")
    print("=" * 60)

    for i, doc in enumerate(sources):
        clean_content = " ".join(doc.page_content.split())
        
        print(f"\n┌─ Chunk {i} ─────────────────────────────────────────")
        print(f"│ Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"├─────────────────────────────────────────────────────")
        print(f"│ {clean_content}")
        print(f"└─────────────────────────────────────────────────────")


def display_sources(sources):
    """Display source documents used."""
    print("\n📖 Sources used:")
    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get('source', 'Unknown')
        print(f"  {i}. {source}")
        print(f"     Preview: {doc.page_content[:100]}...")
    print()


def process_input(user_input, chain, last_sources):
    """
    Process a single user input and return the result.
    
    Returns:
        tuple: (action, chain, last_sources)
        - action: 'quit', 'continue', or 'response'
    """
    # Handle quit
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye! 👋")
        return 'quit', chain, last_sources

    # Handle reindex
    if user_input.lower() == 'reindex':
        vector_store = clear_and_reindex()
        if vector_store:
            chain = create_rag_chain(vector_store)
        return 'continue', chain, last_sources
    
    # Handle sources
    if user_input.lower() == 'sources' and last_sources:
        display_sources(last_sources)
        return 'continue', chain, last_sources
    
    # Handle empty input
    if not user_input:
        return 'continue', chain, last_sources
    
    # Process question
    result = chain.invoke({"question": user_input})
    answer = result["answer"]
    last_sources = result.get("source_documents", [])

    display_chunks(last_sources)
    
    print(f"\nAssistant: {answer}")
    if last_sources:
        print(f"  (Based on {len(last_sources)} source(s) - type 'sources' to see them)")
    print()
    
    return 'response', chain, last_sources


def chat_loop(vector_store):
    """Main chat loop."""
    chain = create_rag_chain(vector_store)
    
    print("\n🤖 Chatbot ready! Ask questions about your documents.")
    print("   Commands: 'quit', 'sources', 'reindex'\n")
    
    last_sources = []
    
    while True:
        user_input = input("You: ").strip()
        
        action, chain, last_sources = process_input(user_input, chain, last_sources)
        
        if action == 'quit':
            break
