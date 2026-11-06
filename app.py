import streamlit as st
import uuid

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="MEMORA AI",
    page_icon="🧠",
    layout="wide"
)


# ==================================================
# 2. CUSTOM CSS - FONT AND BACKGROUND COLOURS
# ==================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(135deg, #101827, #202d44);
    color: #ffffff !important;
}

/* Main content text */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 {
    color: #f8fafc !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111b2d !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Chat message background */
[data-testid="stChatMessage"] {
    background-color: #26354d !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

/* Chat response text */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
    color: #ffffff !important;
}

/* Chat input container */
[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 2px solid #817cf5 !important;
    border-radius: 12px !important;
}

/* Chat input typed text */
.stApp [data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    caret-color: #111111 !important;
    font-size: 16px !important;
}

/* Chat input placeholder */
.stApp [data-testid="stChatInput"] textarea::placeholder {
    color: #666666 !important;
    -webkit-text-fill-color: #666666 !important;
    opacity: 1 !important;
}

/* Other input fields */
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    caret-color: #111111 !important;
}

/* Select box */
.stApp [data-baseweb="select"] * {
    color: #111111 !important;
}

/* Buttons */
.stButton button {
    background-color: #635bdb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Button hover */
.stButton button:hover {
    background-color: #817cf5 !important;
    color: #ffffff !important;
}

/* Download button */
.stDownloadButton button {
    background-color: #635bdb !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Horizontal line */
hr {
    border-color: #52627b !important;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. MEMORY TYPES
# ==================================================

MEMORY_TYPES = {
    "Buffer Memory":
        "Stores the complete conversation history.",

    "Summary Memory":
        "Creates a short summary of previous conversations.",

    "Buffer Window Memory":
        "Keeps only the latest six messages.",

    "Vector Store Memory":
        "Retrieves relevant past information using embeddings.",

    "Entity Memory":
        "Remembers important names, people and facts."
}


# ==================================================
# 4. CREATE MEMORY
# ==================================================

def create_memory():

    return {
        "history": InMemoryChatMessageHistory(),
        "summary": "",
        "entities": "",
        "vector_store": None,
        "vector_count": 0,
        "messages": []
    }


# ==================================================
# 5. SESSION STATE
# ==================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())


if "memory_data" not in st.session_state:

    st.session_state.memory_data = {
        name: create_memory()
        for name in MEMORY_TYPES
    }


# ==================================================
# 6. LOAD LOCAL MODELS
# ==================================================

@st.cache_resource
def load_models():

    chat_model = ChatOllama(
        model="gemma3:1b",
        temperature=0.7
    )

    embedding_model = OllamaEmbeddings(
        model="embeddinggemma"
    )

    return chat_model, embedding_model


# ==================================================
# 7. SIDEBAR
# ==================================================

st.sidebar.title("MEMORA AI")

st.sidebar.caption(
    "Local AI Chatbot with LangChain Memory"
)

selected_memory = st.sidebar.selectbox(
    "Select Memory Type",
    list(MEMORY_TYPES.keys())
)

st.sidebar.info(
    MEMORY_TYPES[selected_memory]
)


# New Conversation

if st.sidebar.button(
    "New Conversation",
    use_container_width=True
):

    st.session_state.session_id = str(uuid.uuid4())

    st.session_state.memory_data = {
        name: create_memory()
        for name in MEMORY_TYPES
    }

    st.rerun()


# Clear Current Chat

if st.sidebar.button(
    "Clear Current Chat",
    use_container_width=True
):

    st.session_state.memory_data[selected_memory] = (
        create_memory()
    )

    st.rerun()


st.sidebar.markdown("---")

st.sidebar.subheader("About MEMORA AI")

st.sidebar.write(
    "MEMORA AI is a local AI chatbot that uses "
    "LangChain memory techniques to remember "
    "and retrieve conversation information."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Powered by Ollama | Gemma 3 | LangChain"
)


# ==================================================
# 8. INITIALIZE MODELS
# ==================================================

try:

    llm, embeddings = load_models()

except Exception as error:

    st.error(
        "Unable to initialize AI models. "
        "Please check your Ollama installation."
    )

    st.code(str(error))

    st.stop()


# ==================================================
# 9. ACTIVE MEMORY
# ==================================================

memory = st.session_state.memory_data[selected_memory]


# ==================================================
# 10. MAIN TITLE
# ==================================================

st.title("MEMORA AI")

st.markdown(
    "An AI That Remembers, Understands & Responds."
)

st.markdown("---")

st.markdown(
    f"**Active Memory:** {selected_memory}"
)


# ==================================================
# 11. LANGCHAIN PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are MEMORA AI, a helpful and intelligent
        AI assistant.

        Answer the user's questions clearly and accurately.

        Use the supplied memory context when relevant.

        If the requested information is not available
        in memory, do not invent personal facts.

        Memory Context:
        {context}
        """
    ),

    MessagesPlaceholder(
        variable_name="history"
    ),

    (
        "human",
        "{input}"
    )

])

chain = prompt | llm


# ==================================================
# 12. GET MEMORY CONTEXT
# ==================================================

def get_memory_context(user_input):

    history = memory["history"].messages

    context = ""

    selected_history = []


    # BUFFER MEMORY

    if selected_memory == "Buffer Memory":

        selected_history = history

        context = (
            "Use the complete conversation history."
        )


    # SUMMARY MEMORY

    elif selected_memory == "Summary Memory":

        selected_history = []

        context = (
            "Previous conversation summary:\n"
            + memory["summary"]
        )


    # BUFFER WINDOW MEMORY

    elif selected_memory == "Buffer Window Memory":

        selected_history = history[-6:]

        context = (
            "Use only the recent conversation messages."
        )


    # VECTOR STORE MEMORY

    elif selected_memory == "Vector Store Memory":

        selected_history = []

        if memory["vector_store"] is None:

            memory["vector_store"] = InMemoryVectorStore(
                embedding=embeddings
            )

        if memory["vector_count"] > 0:

            results = memory["vector_store"].similarity_search(
                user_input,
                k=3
            )

            memories = "\n".join(
                document.page_content
                for document in results
            )

            context = (
                "Relevant previous conversation information:\n"
                + memories
            )


    # ENTITY MEMORY

    elif selected_memory == "Entity Memory":

        selected_history = []

        context = (
            "Previously remembered entities and facts:\n"
            + memory["entities"]
        )


    return context, selected_history


# ==================================================
# 13. UPDATE MEMORY
# ==================================================

def update_memory(user_input, answer):

    history = memory["history"]

    history.add_messages([

        HumanMessage(content=user_input),

        AIMessage(content=answer)

    ])


    # SUMMARY MEMORY

    if selected_memory == "Summary Memory":

        summary_prompt = f"""
        Update the previous conversation summary.

        Keep important names, facts and details.

        Previous Summary:
        {memory["summary"]}

        New Conversation:
        User: {user_input}
        Assistant: {answer}

        Return a concise updated summary.
        Only include information supported by the conversation.
        """

        result = llm.invoke(summary_prompt)

        memory["summary"] = str(result.content)


    # VECTOR STORE MEMORY

    elif selected_memory == "Vector Store Memory":

        if memory["vector_store"] is None:

            memory["vector_store"] = InMemoryVectorStore(
                embedding=embeddings
            )

        memory["vector_store"].add_texts([

            f"User: {user_input}\nAssistant: {answer}"

        ])

        memory["vector_count"] += 1


    # ENTITY MEMORY

    elif selected_memory == "Entity Memory":

        entity_prompt = f"""
        Extract important entities and facts.

        Preserve previously known information.

        Previous Information:
        {memory["entities"]}

        New Conversation:
        User: {user_input}
        Assistant: {answer}

        Return a concise list of important facts.

        Only include facts supported by the conversation.
        """

        result = llm.invoke(entity_prompt)

        memory["entities"] = str(result.content)


# ==================================================
# 14. DISPLAY CHAT HISTORY
# ==================================================

for message in memory["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================================
# 15. CHAT INPUT AND RESPONSE
# ==================================================

user_input = st.chat_input(
    "Ask MEMORA AI anything..."
)


if user_input:

    with st.chat_message("user"):

        st.markdown(user_input)


    try:

        with st.chat_message("assistant"):

            with st.spinner("MEMORA AI is thinking..."):

                context, history = get_memory_context(
                    user_input
                )

                response = chain.invoke({

                    "context": context,

                    "history": history,

                    "input": user_input

                })

                answer = response.content

                st.markdown(answer)


        # Save conversation messages

        memory["messages"].append({

            "role": "user",

            "content": user_input

        })

        memory["messages"].append({

            "role": "assistant",

            "content": answer

        })


        # Update selected memory

        try:

            update_memory(
                user_input,
                answer
            )

        except Exception as memory_error:

            st.warning(
                "Response generated, but memory update failed."
            )

            st.caption(str(memory_error))


    except Exception as error:

        st.error(
            "Unable to generate a response. "
            "Check whether Ollama is running and "
            "the required models are installed."
        )

        st.code(str(error))


# ==================================================
# 16. DOWNLOAD CONVERSATION
# ==================================================

if memory["messages"]:

    chat_text = "\n\n".join(

        f"{message['role'].upper()}: "
        f"{message['content']}"

        for message in memory["messages"]

    )

    st.sidebar.download_button(

        label="Download Conversation",

        data=chat_text,

        file_name="MEMORA_AI_Conversation.txt",

        mime="text/plain",

        use_container_width=True

    )


# ==================================================
# 17. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#cbd5e1;
        padding:10px;
    ">
        MEMORA AI | Local AI Chatbot with LangChain Memory
    </div>
    """,
    unsafe_allow_html=True
)
