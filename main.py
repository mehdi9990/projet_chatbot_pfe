from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import search_vector_store
from memory import store_conversation
from web_search import search_web

# ===============================
# MODELS
# ===============================
llama_model = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")
qwen_model = OllamaLLM(model="qwen:7b", base_url="http://127.0.0.1:11434")

# ===============================
# PROMPT TEMPLATE
# ===============================
template = """
You are an expert assistant.

If reviews are provided, use them to answer.
If no reviews are relevant, answer using your own knowledge.

Reviews:
{reviews}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# ===============================
# CHAINS
# ===============================
llama_chain = prompt | llama_model
qwen_chain = prompt | qwen_model

# ===============================
# HELPER: Convert Documents to Text
# ===============================
def format_documents(docs):
    return "\n\n".join([doc.page_content for doc in docs]) if docs else ""

# ===============================
# CLASSIFIER
# ===============================
def classify_context(question: str) -> str:
    classifier_prompt = f"""
Classify this question into ONE single word category:
restaurant, technical, banking, general.

Question: {question}
Category:
"""
    try:
        response = qwen_model.invoke(classifier_prompt)
        return response.strip().lower() if response else "general"
    except Exception:
        return "general"

# ===============================
# MAIN LOOP
# ===============================
def main():
    print("📂 Loading existing vector database...")
    
    while True:
        print("\n-------------------------------")
        question = input("Ask your question (q to quit): ").strip()

        if question.lower() == "q":
            break

        # ===============================
        # 1️⃣ SMART VECTOR SEARCH
        # ===============================
        print("🔎 Searching in Vector DB...")
        reviews_docs = search_vector_store(question)

        if reviews_docs:
            print("📚 Relevant documents found.")
            formatted_reviews = format_documents(reviews_docs)
            try:
                result = llama_chain.invoke({
                    "reviews": formatted_reviews,
                    "question": question
                })
            except Exception as e:
                print("⚠️ LLaMA model failed:", e)
                result = None

            context = classify_context(question)
            store_conversation(question, result or "No answer", context)
            print("\n🤖 Answer:\n", result or "No answer")
            continue

        # ===============================
        # 2️⃣ FALLBACK TO QWEN
        # ===============================
        print("🤖 Using Qwen fallback...")
        try:
            result = qwen_chain.invoke({
                "reviews": "",
                "question": question
            })
        except Exception as e:
            print("⚠️ Qwen model failed:", e)
            result = None

        if result and len(result.strip()) > 30:
            context = classify_context(question)
            store_conversation(question, result, context)
            print("\n🤖 Answer:\n", result)
            continue

        # ===============================
        # 3️⃣ FALLBACK LLaMA if Qwen fails
        # ===============================
        if not result:
            print("🤖 Fallback to LLaMA...")
            try:
                result = llama_chain.invoke({
                    "reviews": "",
                    "question": question
                })
            except Exception as e:
                print("⚠️ LLaMA fallback failed:", e)
                result = "No answer available"

            context = classify_context(question)
            store_conversation(question, result, context)
            print("\n🤖 Answer:\n", result)
            continue

        # ===============================
        # 4️⃣ WEB FALLBACK
        # ===============================
        print("🌐 Searching Web...")
        try:
            web_result = search_web(question) or "No web result found."
        except Exception as e:
            print("⚠️ Web search failed:", e)
            web_result = "No web result found."

        context = classify_context(question)
        store_conversation(question, web_result, context)
        print("\n🌍 Web Answer:\n", web_result)

if __name__ == "__main__":
    main()
