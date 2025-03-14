from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


template ="""
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model                                               # chaining prompt and model using pipelining

# storing conversation history

def handle_conversation():
    context = ""
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        result = chain.invoke({"context": context, "question": user_input})   # embeding question and context inside of the prompt
        print("Bot: ",result)
        context += f"\nUser: {user_input}\nAI: {result}"                            # conversation history

if __name__ == "__main__":
    handle_conversation()