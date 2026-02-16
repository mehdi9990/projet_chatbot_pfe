from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import search_vector_store
from memory import store_conversation
from web_search import search_wikipedia
import tiktoken

# ===============================
# MODELS
# ===============================
llama_model = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")
qwen_model = OllamaLLM(model="qwen:32b", base_url="http://127.0.0.1:11434")

# ===============================
# PROMPT
# ===============================
template = """
You are an expert assistant.

If restaurant reviews are provided, use them to answer.
If not, answer normally.

Reviews:
{reviews}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
llama_chain = prompt | llama_model
qwen_chain = prompt | qwen_model

# ===============================
# UTILITAIRES
# ===============================
def format_documents(docs):
    return "\n\n".join([doc.page_content for doc in docs]) if docs else ""

def count_tokens(text: str, model_name: str = "llama3.2") -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

# ===============================
# CLASSIFIER
# ===============================
def classify_context(question: str) -> str:
    keywords = [
        "restaurant", "pizza", "food", "delivery", "menu",
        "review", "crust", "cheese", "waiter", "service"
    ]
    for word in keywords:
        if word in question.lower():
            return "restaurant"
    # fallback to LLM
    classifier_prompt = f"""
Classify this question into ONE word only:
restaurant or general.

Question: {question}
Category:
"""
    try:
        response = qwen_model.invoke(classifier_prompt)
        return response.strip().lower()
    except:
        return "general"

# ===============================
# MAIN LOOP
# ===============================
def main():
    print("🚀 Multi-LLM Intelligent Chatbot Ready")

    while True:
        print("\n-------------------------------")
        question = input("Ask your question (q to quit): ").strip()
        if question.lower() == "q":
            break

        # 1️⃣ CLASSIFY QUESTION
        context = classify_context(question)
        print(f"🧠 Detected context: {context}")

        # 2️⃣ RESTAURANT → MINIVERSE
        if context == "restaurant":
            print("🔎 Searching MiniVerse...")
            docs = search_vector_store(question)
            if docs:
                formatted = format_documents(docs)
                result = llama_chain.invoke({"reviews": formatted, "question": question})
                store_conversation(question, result, context)
                num_tokens = count_tokens(result)
                print("\n🤖 Answer (MiniVerse):\n", result)
                print(f"🔢 Tokens used: {num_tokens}")
                continue

        # 3️⃣ QWEN
        print("🤖 Trying Qwen...")
        result = qwen_chain.invoke({"reviews": "", "question": question})
        if result and len(result.strip()) > 40:
            store_conversation(question, result, context)
            num_tokens = count_tokens(result)
            print("\n🤖 Answer (Qwen):\n", result)
            print(f"🔢 Tokens used: {num_tokens}")
            continue

        # 4️⃣ LLAMA FALLBACK
        print("🦙 Trying LLaMA...")
        result = llama_chain.invoke({"reviews": "", "question": question})
        if result and len(result.strip()) > 40:
            store_conversation(question, result, context)
            num_tokens = count_tokens(result)
            print("\n🤖 Answer (LLaMA):\n", result)
            print(f"🔢 Tokens used: {num_tokens}")
            continue

        # 5️⃣ WEB FALLBACK (Wikipedia)
        print("🌐 Searching Web (Wikipedia)...")
        web_result = search_wikipedia(question)
        store_conversation(question, web_result, context)
        num_tokens = count_tokens(web_result)
        print("\n🌍 Web Answer:\n", web_result)
        print(f"🔢 Tokens used: {num_tokens}")

if __name__ == "__main__":
    main()
