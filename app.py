import streamlit as st
import uuid

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MEMORA AI",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #101827, #202d44);
    color: #ffffff;
}

/* Headings */
h1, h2, h3, p, label {
    color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111b2d;
}

/* Chat input box */
[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 2px solid #817cf5 !important;
    border-radius: 12px !important;
}

/* Typed text */
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    caret-color: #111111 !important;
    font-size: 16px !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #555555 !important;
    -webkit-text-fill-color: #555555 !important;
    opacity: 1 !important;
}

/* Other text inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    caret-color: #111111 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #26354d;
    border-radius: 12px;
    padding: 12px;
}

/* Buttons */
.stButton button {
    background-color: #635bdb;
    color: #ffffff !important;
    border-radius: 8px;
    border: none;
}

.stButton button:hover {
    background-color: #817cf5;
    color: #ffffff !important;
}

/* Download button */
.stDownloadButton button {
    background-color: #635bdb;
    color: #ffffff !important;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MEMORY TYPES
# --------------------------------------------------

MEMORY_TYPES = {
    "Buffer Memory":
        "Stores the complete conversation history.",

    "Summary Memory":
        "Creates a summary of previous conversations.",

    "Buffer Window Memory":
        "Keeps only the latest conversation messages.",

    "Vector Store Memory":
        "Uses embeddings to retrieve relevant past information.",

    "Entity Memory":
        "Remembers important names, facts and entities."
}


# --------------------------------------------------
# CREATE MEMORY
# --------------------------------------------------

def create_memory():

    return {
        "history": InMemoryChatMessageHistory(),
        "summary": "",
        "entities": "",
        "vector_store": None,
        "vector_count": 0,
        "messages": []
    }


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())


if "memory_data" not in st.session_state:

    st.session_state.memory_data = {
        name: create_memory()
        for name in MEMORY_TYPES
    }


if "active_memory" not in st.session_state:

    st.session_state.active_memory = "Buffer Memory"


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("MEMORA AI")

st.sidebar.caption(
    "Local AI Chatbot with LangChain Memory"
)

selected_memory = st.sidebar.selectbox(
    "Select Memory Type",
    list(MEMORY_TYPES.keys())
)

st.session_state.active_memory = selected_memory

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
    "different memory techniques to remember "
    "conversation information."
)

st.sidebar.caption(
    "Powered by Ollama, Gemma 3 and LangChain"
)


# --------------------------------------------------
# LOAD LOCAL MODELS
# --------------------------------------------------

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


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

try:

    llm, embeddings = load_models()

except Exception as error:

    st.error(
        "Unable to initialize the AI models. "
        "Please check your Ollama installation."
    )

    st.caption(str(error))

    st.stop()


# --------------------------------------------------
# ACTIVE MEMORY
# --------------------------------------------------

memory = st.session_state.memory_data[selected_memory]


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

st.title("MEMORA AI")

st.caption(
    "An AI That Remembers, Understands & Responds."
)

st.markdown(
    f"**Active Memory:** {selected_memory}"
)

st.markdown("---")


# --------------------------------------------------
# LANGCHAIN PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        "You are MEMORA AI, a helpful AI assistant. "
        "Answer clearly and accurately. "
        "Use the provided memory context when relevant.\n\n"
        "Memory Context:\n{context}"
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


# --------------------------------------------------
# GET MEMORY CONTEXT
# --------------------------------------------------

def get_memory_context(user_input):

    history = memory["history"].messages

    context = ""

    selected_history = []


    # 1. BUFFER MEMORY

    if selected_memory == "Buffer Memory":

        selected_history = history

        context = (
            "Use the complete conversation history."
        )


    # 2. SUMMARY MEMORY

    elif selected_memory == "Summary Memory":

        selected_history = []

        context = (
            "Previous Conversation Summary:\n"
            + memory["summary"]
        )


    # 3. BUFFER WINDOW MEMORY

    elif selected_memory == "Buffer Window Memory":

        selected_history = history[-6:]

        context = (
            "Use only the recent conversation messages."
        )


    # 4. VECTOR STORE MEMORY

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
                "Relevant Previous Conversation Information:\n"
                + memories
            )


    # 5. ENTITY MEMORY

    elif selected_memory == "Entity Memory":

        selected_history = []

        context = (
            "Previously Remembered Entities and Facts:\n"
            + memory["entities"]
        )


    return context, selected_history


# --------------------------------------------------
# UPDATE MEMORY
# --------------------------------------------------

def update_memory(user_input, answer):

    history = memory["history"]

    history.add_messages([

        HumanMessage(content=user_input),

        AIMessage(content=answer)

    ])


    # SUMMARY MEMORY UPDATE

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
        """

        result = llm.invoke(summary_prompt)

        memory["summary"] = str(result.content)


    # VECTOR STORE MEMORY UPDATE

    elif selected_memory == "Vector Store Memory":

        if memory["vector_store"] is None:

            memory["vector_store"] = InMemoryVectorStore(
                embedding=embeddings
            )

        memory["vector_store"].add_texts([

            f"User: {user_input}\nAssistant: {answer}"

        ])

        memory["vector_count"] += 1


    # ENTITY MEMORY UPDATE

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


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in memory["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

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


        # Save successful conversation

        memory["messages"].append({

            "role": "user",

            "content": user_input

        })

        memory["messages"].append({

            "role": "assistant",

            "content": answer

        })


        # Update selected memory

        update_memory(
            user_input,
            answer
        )


    except Exception as error:

        st.error(
            "Unable to generate a response. "
            "Please check whether Ollama is running "
            "and the required models are installed."
        )

        st.caption(str(error))


# --------------------------------------------------
# DOWNLOAD CHAT
# --------------------------------------------------

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


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "MEMORA AI | Local AI Chatbot with LangChain Memory"
)
