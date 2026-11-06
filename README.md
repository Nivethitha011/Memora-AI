# MEMORA AI – Local AI Chatbot with LangChain Memory

## Overview

MEMORA AI is a locally running AI chatbot developed using Python, Streamlit, Ollama, and LangChain. It provides an interactive conversational interface that generates intelligent responses while maintaining conversation context through LangChain memory.

The application uses a locally hosted Large Language Model (LLM), allowing users to interact with an AI assistant without depending on external AI APIs.

<img width="904" height="384" alt="image" src="https://github.com/user-attachments/assets/12dc72c9-2605-4086-8a6f-66f6f5e4e3c9" />
<img width="869" height="344" alt="image" src="https://github.com/user-attachments/assets/6a555c6e-7dbf-41a2-b4ca-e3abda5fc615" />
<img width="912" height="256" alt="image" src="https://github.com/user-attachments/assets/7a31ccc9-25ff-4fe7-98a5-4a94b1eb8b68" />
<img width="904" height="364" alt="image" src="https://github.com/user-attachments/assets/039402fa-785c-48ec-a2bc-9d3cb90fa93c" />
<img width="895" height="336" alt="image" src="https://github.com/user-attachments/assets/60bc4aa7-4d9e-4169-aa5b-953daadabb91" />
<img width="950" height="433" alt="image" src="https://github.com/user-attachments/assets/12ffb496-ec1d-4dc1-b78e-08e1c0afa87d" />
<img width="225" height="436" alt="image" src="https://github.com/user-attachments/assets/fc15df93-3260-4f2e-8259-749ca9df3421" />


## Project Objectives

- Develop a simple and interactive AI chatbot.
- Integrate a locally running Large Language Model.
- Implement conversational memory using LangChain.
- Maintain context across multiple messages.
- Provide a user-friendly chatbot interface.
- Understand how memory improves conversational AI applications.

## Features

- Local AI inference using Ollama.
- Gemma 3 language model integration.
- Conversation memory using LangChain.
- Context-aware responses.
- Interactive chat interface.
- Real-time conversational interaction.
- Clear conversation history functionality.
- Session-based conversation management.
- Local processing without external AI API keys.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Streamlit | Web-based user interface |
| Ollama | Local LLM execution |
| Gemma 3 | Language model |
| LangChain | LLM integration and prompt management |
| LangChain Core | Conversation history management |

## Project Structure

```text
MEMORA-AI/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## LangChain Memory Implementation

MEMORA AI uses LangChain's `RunnableWithMessageHistory` and `InMemoryChatMessageHistory` to maintain conversational context.

### Memory Workflow

1. The user sends a message.
2. The message is passed to the LangChain chat chain.
3. Previous conversation messages are retrieved.
4. The conversation history is included in the prompt.
5. Ollama generates a response using the Gemma 3 model.
6. The user message and AI response are stored in memory.
7. The next message uses the previous conversation context.

Memory is maintained separately for each conversation session.

The application uses in-memory storage, so conversation history is temporary and is not permanently stored after the application session ends.

## Testing

The following test cases can be used to verify the application.

| Test Case | Description | Expected Result |
|---|---|---|
| Basic Chat | Send a simple greeting | Chatbot generates a response |
| Conversation Memory | Provide a name and ask it later | Chatbot remembers the name |
| Context Awareness | Ask follow-up questions | Chatbot understands previous context |
| General Knowledge | Ask a general question | Chatbot provides a relevant answer |
| New Conversation | Start a new session | Previous conversation memory is not used |
| Clear Chat | Clear the current conversation | Current history is removed |
| Local Model | Run chatbot using Ollama | Responses are generated locally |

## Privacy and Security

- AI inference is performed locally using Ollama.
- No external AI API key is required.
- Conversation history is maintained in application memory.
- Chat history is not automatically persisted to a database.
- Local inference reduces dependence on external AI services.

## Limitations

- Response generation depends on available system hardware.
- Local language models may require sufficient RAM.
- Conversation memory is temporary.
- The chatbot may occasionally generate inaccurate information.
- Model performance depends on the installed LLM.

## Future Enhancements

- Persistent conversation storage using SQLite.
- Conversation summary memory.
- Conversation buffer window memory.
- Support for multiple local LLM models.
- User-specific chat sessions.
- Document-based question answering.
- Improved chatbot personalization.

## Conclusion

MEMORA AI demonstrates how locally running Large Language Models can be integrated with LangChain conversational memory to develop an intelligent chatbot application.

The project combines local AI inference, conversational context management, and an interactive Streamlit interface to provide a practical understanding of memory-based conversational AI systems.
