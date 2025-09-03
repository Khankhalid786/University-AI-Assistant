# 🎓 University AI Assistant Chatbot

This is an intelligent, multi-functional chatbot designed to serve as a comprehensive information hub for a university. Built with Python and powered by the Google Gemini API, it can handle a wide range of queries by routing them to the appropriate backend pipeline.

---

## ✨ Key Features

This bot is more than just a simple Q&A machine. It has a sophisticated architecture that includes:

-   **🤖 Smart Intent Routing:** A core classifier analyzes every user query to determine the user's goal (e.g., asking a question, registering a student, deleting a faculty member).

-   **📚 Document Q&A (RAG):** For questions about university policies, rules, and handbooks, the bot uses Retrieval-Augmented Generation (RAG). It searches a vector database of university documents to find the most relevant information and generates a precise, source-cited answer.

-   **🗃️ Database Q&A (Text-to-SQL):** For specific data queries like "Who teaches CS101?" or "What courses are in the Physics department?", the bot dynamically generates and executes SQL queries on the university database to fetch live information.

-   **✍️ Data Management:** The bot can perform C.R.U.D. (Create, Read, Update, Delete) operations through natural language. Users with the correct permissions can conversationally register new students, add faculty, or remove records.

-   **🗣️ Conversational Memory:** The bot remembers the context of the conversation, allowing for natural follow-up questions.

---

## 🛠️ Tech Stack

-   **Core Language:** Python
-   **LLM & Embeddings:** Google Gemini API (`gemini-2.5-flash`, `embedding-001`)
-   **AI/NLP Framework:** LangChain
-   **Vector Database:** ChromaDB
-   **Structured Database:** SQLite
-   **Data Validation:** Pydantic
-   **Development Environment:** Google Colab

---

## 🚀 How to Use

1.  Clone the repository.
2.  Install the required dependencies from `requirements.txt`.
3.  Set up your `GOOGLE_API_KEY` in the environment.
4.  Run the main script to start the chatbot in your terminal.
