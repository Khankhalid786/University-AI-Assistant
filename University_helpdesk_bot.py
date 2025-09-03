!pip install chromadb
!pip install langchain-google-genai
!pip install -U langchain-chroma
!pip install --upgrade --quiet google-genai PyPDF2
!pip install -qU "langchain-chroma>=0.1.2"
!pip install -U langchain-google-genai
!pip install -U langchain-community pymupdf

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from google.colab import userdata
from google import genai

# Set API key for LangChain Google Generative AI integration
client = genai.Client(api_key=userdata.get('GOOGLE_API_KEY'))
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create / load Chroma vector store
vector_store = Chroma(
    collection_name="help_desk",
    embedding_function=embeddings,
    persist_directory="/content/drive/MyDrive/projects/vdb",
)

import fitz  # PyMuPDF
import os
from langchain_core.documents import Document

def read_pdf_page_by_page(pdf_path):
    data = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip() != "":
                    data.append({
                        "text": text,
                        "page": page_num + 1,
                        "path": pdf_path
                    })
        return data
    except:
        return []

def add_document_to_vdb(pdf_path, vector_store):
    data = []
    langchain_docs = []

    try:
        if os.path.isdir(pdf_path):
            for filename in os.listdir(pdf_path):
                if filename.endswith(".pdf"):
                    full_path = os.path.join(pdf_path, filename)
                    data.extend(read_pdf_page_by_page(full_path))
        else:
            data.extend(read_pdf_page_by_page(pdf_path))
    except:
        return vector_store, []

    if len(data) > 0:
        file_id = 0

        for doc in data:
            file_name = os.path.basename(doc["path"])
            langchain_docs.append(
                Document(
                    page_content=doc["text"],
                    metadata={
                        "page": doc["page"],
                        "file_name": file_name
                    },
                    id=f"{file_name}_{file_id}"
                )
            )
            file_id += 1

        vector_store.add_documents(documents=langchain_docs)

    print("Data added successfully")
    return vector_store, langchain_docs

all_docs = []
pdf_files = [
    "/content/drive/MyDrive/document/Academics.pdf",
    "/content/drive/MyDrive/document/Cafeteria Menu & Rules.pdf",
    "/content/drive/MyDrive/document/Course Catalogue.pdf",
    "/content/drive/MyDrive/document/Department Syllabus.pdf",
    "/content/drive/MyDrive/document/Emergency Contacts.pdf",
    "/content/drive/MyDrive/document/Exam Rules.pdf",
    "/content/drive/MyDrive/document/Hostel Rules.pdf",
    "/content/drive/MyDrive/document/Library Handbook.pdf",
    "/content/drive/MyDrive/document/Scholarship.pdf",
    "/content/drive/MyDrive/document/Sports & Extracurricular.pdf",
    "/content/drive/MyDrive/document/Transportation Guides fares.pdf"

]


for pdf in pdf_files:
    _, docs = add_document_to_vdb(pdf, vector_store)
    all_docs.extend(docs)

import sqlite3

def setup_database():
    conn = None
    try:
        # --- 1. Connect to the database ---
        conn = sqlite3.connect('university.db')
        cursor = conn.cursor()

        # --- 2. Create and Populate Tables ---

        # -- Table 1: faculty --
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faculty (
                faculty_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            );
        ''')

        faculty_data = [
            ('Dr. Alan Turing', 'alan.turing@university.edu'),
            ('Dr. Marie Curie', 'marie.curie@university.edu'),
            ('Dr. Evelyn Reed', 'evelyn.reed@university.edu')
        ]

        cursor.executemany(
            'INSERT OR IGNORE INTO faculty (name, email) VALUES (?, ?)',
            faculty_data
        )

        # -- Table 2: departments --
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS departments (
                department_id INTEGER PRIMARY KEY AUTOINCREMENT,
                department_name TEXT NOT NULL UNIQUE,
                head_id INTEGER,
                FOREIGN KEY (head_id) REFERENCES faculty(faculty_id)
            );
        ''')

        departments_data = [
            ('Computer Science', 1),
            ('Physics', 2),
            ('History', 3)
        ]

        cursor.executemany(
            'INSERT OR IGNORE INTO departments (department_name, head_id) VALUES (?, ?)',
            departments_data
        )

        # -- Table 3: courses --
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                course_id TEXT PRIMARY KEY,
                course_name TEXT NOT NULL,
                department_name TEXT,
                credits INTEGER,
                FOREIGN KEY (department_name) REFERENCES departments(department_name)
            );
        ''')

        courses_data = [
            ('CS101', 'Introduction to Python', 'Computer Science', 3),
            ('PHY201', 'Classical Mechanics', 'Physics', 4),
            ('HIS305', 'Modern European History', 'History', 3),
            ('CS303', 'Data Structures and Algorithms', 'Computer Science', 4)
        ]

        cursor.executemany(
            'INSERT OR IGNORE INTO courses VALUES (?, ?, ?, ?)',
            courses_data
        )

        # -- Table 4: students --
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                major TEXT,
                enrollment_year INTEGER,
                FOREIGN KEY (major) REFERENCES departments(department_name)
            );
        ''')

        students_data = [
            ('Alice Johnson', 'Computer Science', 2023),
            ('Bob Smith', 'Physics', 2022),
            ('Charlie Brown', 'History', 2024)
        ]

        cursor.executemany(
            'INSERT OR IGNORE INTO students (name, major, enrollment_year) VALUES (?, ?, ?)',
            students_data
        )

        # --- 3. Commit the transaction ---
        conn.commit()
        print("Database 'university.db' with 4 tables is ready. 🗃️✨")

    except sqlite3.Error as e:
        print(f"A database error occurred: {e}")
    finally:
        if conn:
            conn.close()

import json
import sqlite3


def register_student(user_query):
    """
    Register a student via LLM extraction, then ask interactively for missing info.
    Returns only a success message, never raw JSON.
    """
    conn = sqlite3.connect("university.db")
    cursor = conn.cursor()

    prompt =f"""
    You are an intelligent data parsing assistant for a university system.
    Your goal is to identify and extract a person's name, major, and enrollment year from a query.

    First, understand the difference between a 'role' and a 'name':
    - A 'role' is a category like 'Student', 'Faculty', or 'Professor'.
    - A 'name' is the personal identifier, such as 'Alice' or 'John Doe'.

    Your primary instruction is to extract only the 'name'. The 'role' must be ignored.

    Return a JSON object with these keys:
    - "name": The person's name, excluding their role. Use null if not found.
    - "major": The academic field of study. Use null if not found.
    - "enrollment_year": The four-digit year. Use null if not found.

Now extract from this query: "{user_query}"
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.0}
    )

    try:
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
    except Exception:
        parsed = {"name": None, "major": None, "enrollment_year": None}

    name = parsed.get("name")
    major = parsed.get("major")
    year = parsed.get("enrollment_year")

    # Ask interactively if any info is missing
    if not name:
        name = input("Enter student name: ")
    if not major:
        major = input("Enter student major: ")

    # Year validation
    year_provided = bool(year)
    valid_year = False

    while not valid_year:
        try:
            if year:
                year_int = int(year)
                if 2000 <= year_int <= 2030:
                    valid_year = True
                    year = year_int
                else:
                    print("❌ Invalid year, must be between 2000 and 2030.")
                    year = None
            if not year:
                year_input = input("Enter enrollment year: ")
                year_int = int(year_input)
                if 2000 <= year_int <= 2030:
                    valid_year = True
                    year = year_int
                else:
                    print("❌ Invalid year, must be between 2000 and 2030.")
        except ValueError:
            print("❌ Please enter a valid number.")

    cursor.execute(
        "INSERT INTO students (name, major, enrollment_year) VALUES (?, ?, ?)",
        (name, major, year)
    )
    conn.commit()
    conn.close()

    return "✅ Student registered successfully!"



def delete_student(user_query):
    """
    Deletes a student. Tries to extract the name from the query first,
    otherwise, lists all students and asks for the name interactively.
    """
    conn = sqlite3.connect("university.db")
    cursor = conn.cursor()

    prompt = f"""
    You are an intelligent data parsing assistant. Your goal is to extract the
    name of a student to be deleted from a query.

    - A 'role' is a category like 'Student'.
    - A 'name' is the personal identifier, such as 'Alice' or 'John Doe'.
    - Your primary instruction is to extract only the 'name'. The 'role' must be ignored.

    Return a JSON object with one key:
    - "name": The name of the student to delete. Use null if not found.

    Analyze the query: "{user_query}"
    """
    parsed = {"name": None}

    name_to_delete = parsed.get("name")

    if not name_to_delete:
        print("Bot: Sure, which student would you like to delete? Here are the current students:")
        cursor.execute("SELECT student_id, name FROM students ORDER BY name")
        all_students = cursor.fetchall()

        if not all_students:
            conn.close()
            return "There are no students in the database to delete."

        for student in all_students:
            print(f"  ID: {student[0]}, Name: {student[1]}")

        name_to_delete = input("Enter the exact name of the student to delete: ")

    cursor.execute("DELETE FROM students WHERE name = ?", (name_to_delete,))
    conn.commit()

    # Check if a row was actually deleted and return the appropriate message
    if cursor.rowcount > 0:
        conn.close()
        return f"✅ Student '{name_to_delete}' has been deleted."
    else:
        conn.close()
        return f"⚠️ No student found with the name '{name_to_delete}'."



def register_faculty(user_query):
    """
    Register a faculty via LLM extraction, then ask interactively for missing info.
    Returns only a success message, never raw JSON.
    """
    conn = sqlite3.connect("university.db")
    cursor = conn.cursor()

    prompt = f"""
    You are an intelligent data parsing assistant for a university system.
    Your goal is to identify and extract a person's name and email from a query.

    First, understand the difference between a 'role' and a 'name':
    - A 'role' is a category like 'Faculty' or 'Professor'.
    - A 'name' is the personal identifier, such as 'Alice' or 'Dr. John Doe'.

    Your primary instruction is to extract only the 'name'. The 'role' must be ignored.

    Return a JSON object with these keys:
    - "name": The person's name, excluding their role. Use null if not found.
    - "email": The person's email address. Use null if not found.
    - **Use null**: If any piece of information is not mentioned in the query, its value in the JSON object must be null. Do not guess or make up information.

Now extract from this query: "{user_query}"
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.0}
    )

    try:
        text = response.text.strip()
        # Clean up response (remove code blocks if present)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
    except Exception as e:
        parsed = {"name": None, "email": None}

    name = parsed.get("name")
    email = parsed.get("email")

    # Ask interactively if any info is missing
    if not name:
        name = input("Enter faculty name: ")

    # Email validation
    while not email or "@" not in email or "." not in email:
        # Only show the error if the user has already entered something invalid.
        if email is not None:
             print("❌ Invalid email format! Please enter a valid email.")
        email = input("Enter faculty email: ")

    cursor.execute(
        "INSERT INTO faculty (name, email) VALUES (?, ?)",
        (name, email)
    )
    conn.commit()
    conn.close()

    return "✅ Faculty registered successfully!"


def delete_faculty(user_query):
    """
    Deletes a faculty member. Tries to extract the name from the query first,
    otherwise, lists all faculty and asks for the name interactively.
    """
    conn = sqlite3.connect("university.db")
    cursor = conn.cursor()

    prompt = f"""
    You are an intelligent data parsing assistant. Your goal is to extract the
    name of a faculty member to be deleted from a query.

    - A 'role' is a category like 'Faculty' or 'Professor'.
    - A 'name' is the personal identifier, such as 'Alice' or 'Dr. John Doe'.
    - Your primary instruction is to extract only the 'name'. The 'role' must be ignored.

    Return a JSON object with one key:
    - "name": The name of the faculty member to delete. Use null if not found.

    Analyze the query: "{user_query}"
    """

    # response = client.models.generate_content(...)
    # For demonstration, we'll simulate the LLM output
    parsed = {"name": None} # Simulating that the name was not in the query

    name_to_delete = parsed.get("name")

    # If the LLM couldn't find a name, start the interactive process
    if not name_to_delete:
        print("Bot: Sure, which faculty member would you like to delete? Here are the current faculty:")
        cursor.execute("SELECT faculty_id, name FROM faculty ORDER BY name")
        all_faculty = cursor.fetchall()

        if not all_faculty:
            conn.close()
            return "There are no faculty members in the database to delete."

        for faculty in all_faculty:
            print(f"  ID: {faculty[0]}, Name: {faculty[1]}")

        name_to_delete = input("Enter the exact name of the faculty member to delete: ")

    # Perform the deletion
    cursor.execute("DELETE FROM faculty WHERE name = ?", (name_to_delete,))
    conn.commit()

    # Check if a row was actually deleted and return the appropriate message
    if cursor.rowcount > 0:
        conn.close()
        return f"✅ Faculty member '{name_to_delete}' has been deleted."
    else:
        conn.close()
        return f"⚠️ No faculty member found with the name '{name_to_delete}'."

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Classifier(BaseModel):
    """Classifies the user query as 'structured', 'unstructured','chit_chat','register_student','register_faculty','delete_student' and delete_faculty"""
    label: Literal["structured", "unstructured","chit_chat","register_student","register_faculty","delete_student","delete_faculty"] = Field(
        ...,
        description="Classify the user query into 'structured' for database questions, 'unstructured' for general document questions, 'chit_chat' for general hii hello questions, 'register_student and register_faculty for adding new people', 'delete_student and delete_faculty for removing people.'")

  # # --- PIPELINE 1: Unstructured (Text-to-RAG for Documents) ---

def get_unstructured_response(user_query: str, past_conversation: str) -> str:
    """
    Handles questions by searching documents and citing the sources (RAG pipeline).
    """

    # 1. Create a retriever to search the vector store.
    retriever = vector_store.as_retriever(search_kwargs={"k": 25})

    # 2. Find the relevant documents based on the user's query.
    retrieved_docs = retriever.invoke(user_query)

    # 3. If no documents are found, return a helpful message.
    if not retrieved_docs:
        return "I couldn't find any specific information about that in the university documents."

    # 4. Combine the content of the retrieved documents into a single "context".
    context = ""
    context = ""
    for i, doc in enumerate(retrieved_docs):
        source_info = f"Source {i+1} (from {doc.metadata.get('file_name', 'N/A')}, page {doc.metadata.get('page', 'N/A')}):"
        context += f"{source_info}\n{doc.page_content}\n\n---\n\n"

    # 5. Create the final prompt for the AI with instructions to cite sources.
    prompt = f"""You are a helpful university assistant. Answer the user's question based ONLY on the context provided below.
Your answer must be concise. After each piece of information, cite the source number in brackets, like this [file_name , page X].

If the answer cannot be found in the context, reply with:
"No information found in the provided documents."

Conversation history:
{past_conversation}



Context:
{context}

Question: {user_query}
"""

    # 6. Generate the final answer using the AI model.
    final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

    return final_response.text

import json
# # --- PIPELINE 2: STRUCTURED (Text-to-SQL for Database) ---
def get_database_response(user_query: str , past_conversation: str) -> str:
    """Handles specific data questions by querying the university database."""
    db_schema = """
    CREATE TABLE faculty (faculty_id INTEGER PRIMARY KEY, name TEXT, email TEXT);
    CREATE TABLE departments (department_id INTEGER PRIMARY KEY, department_name TEXT, head_id INTEGER);
    CREATE TABLE courses (course_id TEXT PRIMARY KEY, course_name TEXT, department_name TEXT, credits INTEGER);
    CREATE TABLE students (student_id INTEGER PRIMARY KEY, name TEXT, major TEXT, enrollment_year INTEGER);
    """

    prompt = f"""Given the database schema below, generate a valid SQLite query
to answer the user's question. Respond with ONLY the SQL query.


Conversation_History: {past_conversation}
Schema: {db_schema}
Question: '{user_query}'
SQL Query:"""

    try:

        sql_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        sql_query = sql_response.candidates[0].content.parts[0].text.strip()


        sql_query = (
            sql_query.replace("```sql", "")
            .replace("```", "")
            .replace("SQLQuery:", "")
            .replace("SQL Query:", "")
            .strip()
        )


    # Split into lines and find the first SQL statement
        lines = sql_query.split('\n')
        sql_lines = []
        found_statement = False
        for line in lines:
            if line.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
              found_statement = True
            if found_statement:
               sql_lines.append(line)
        sql_query = '\n'.join(sql_lines).strip()


        conn = sqlite3.connect("university.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        conn.close()

        if not rows:
            return "I couldn't find any data in the database for that query."

        result_context = f"Query result: {[dict(zip(col_names, row)) for row in rows]}"
        final_prompt = f"""You are a helpful assistant. A database was queried to answer the user's question.
Convert the following raw data into a clear, natural language answer.

Data: {result_context}
User Question: '{user_query}'
Answer:"""


        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt
        )
        return final_response.candidates[0].content.parts[0].text

    except Exception as e:
        return "I had trouble querying the database. Please rephrase your question."

def route_query(user_query: str) -> str:
    """
    Classifies a user query using a JSON schema and routes it to the correct function.
    """
    prompt = """You are a query classifier for a university chatbot.
Your job is to decide if the user query should be routed to one of the following categories and respond in a valid JSON format based on the provided schema.

- `structured`: if the user is asking for specific information about students, faculty, courses, departments, credits, or contact details.
- `unstructured`: if the user is asking about general information, rules, policies, procedures, campus life, or handbook details.
- `chit_chat`: if the user is making greetings, farewells, expressing thanks, casual conversation, or asking about you as the chatbot, and is not seeking university related information.
- `register_student`: if the user wants to enroll, register, or take admission as a student.
- `register_faculty`: if the user wants to add/register/join as a faculty/professor/teacher.
- `delete_student`: if the user wants to delete, remove, unenroll, or drop a student.
- `delete_faculty`: if the user wants to delete or remove a faculty member, professor, or teacher.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"**Query:** {user_query}",
            config={
                "temperature": 0.0,
                "system_instruction": prompt,
                "response_mime_type": "application/json",
                "response_schema": Classifier
            }
        )
        parsed_json = json.loads(response.text)
        result = Classifier(**parsed_json)


        if result.label == "structured":
            return "This is a structured query."
        elif result.label == "unstructured":
            return "This is an unstructured query."
        elif result.label == "chit_chat":
            chit_chat_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"You are a friendly university assistant chatbot. Respond politely and conversationally to this message:\nUser: {user_query}",
                config={"temperature": 0.7}
            )
            return chit_chat_response.text
        elif result.label == "register_student":
            return register_student(user_query)
        elif result.label == "register_faculty":
            return register_faculty(user_query)

        elif result.label == "delete_student":
            return delete_student(user_query)
        elif result.label == "delete_faculty":
            return delete_faculty(user_query)

        else:
            return "Sorry, I'm not sure how to handle that request."

    except Exception as e:
        print(f"An error occurred in route_query: {e}")
        return "Sorry, I encountered an error."

import string
from langchain.memory import ConversationBufferMemory

chat_buffer_memory = ConversationBufferMemory(return_messages=True)

def university_chatbot(user_query: str):
    """
    The main chatbot function that routes and answers the user's query.
    """

    # Otherwise, proceed with the usual routing and response generation
    chat_buffer_memory.chat_memory.add_user_message(user_query)


    # ---conversation history as context ---
    past_conversation = "\n".join(
        [f"User: {m.content}" if m.type=="human" else f"Assistant: {m.content}"
         for m in chat_buffer_memory.chat_memory.messages]
    )

    # 2. Determine the route
    selected_route = route_query(user_query)

    # 3. Call the appropriate pipeline
    if selected_route == "structured":
        answer = get_database_response(user_query, past_conversation)
    elif selected_route == "unstructured":
        answer = get_unstructured_response(user_query, past_conversation)
    else:
        # If routing returns a chit-chat answer string, return it directly
        answer = selected_route

    return answer

def chatbot_loop():
    """The main loop to interact with the chatbot."""
    print("\n" + "="*50)
    print("🎓 Welcome to the University AI Assistant! 🎓")
    print("You can ask about courses, faculty, or university policies.")
    print("Type 'exit' to quit.")
    print("="*50)

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break
        response = university_chatbot(user_query)
        print(f"Bot: {response}")

if __name__ == "__main__":

     setup_database()
     chatbot_loop()













import sqlite3

def view_table(table_name):
    conn = sqlite3.connect("university.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    print(f"\nTable: {table_name}")
    for row in rows:
        print(row)
    conn.close()


view_table("students")
view_table("faculty")

# !mv /content/vdb/* /content/drive/MyDrive/projects/vdb

# !mkdir /content/vdb
# !mv /content/drive/MyDrive/projects/vdb/* /content/vdb