# Financial Data Q&A System

## Overview

This application allows users to upload PDF documents containing financial data (like Profit & Loss tables), ask financial questions, and get answers. The app uses Google Generative AI and Pinecone for embeddings and retrieval, respectively.

### Features

- Upload financial PDFs via the sidebar.
- Automatically processes the PDF, extracts text, and stores embeddings in a Pinecone Vector Database.
- Ask financial queries in the main chat interface and receive context-based responses.
- Retrieves relevant segments from the document alongside answers.

---

## Tech Stack

1. **Streamlit** - For building the interactive UI.
2. **PyPDF2** - For extracting text from uploaded PDFs.
3. **Google Generative AI (Gemini)** - For embeddings and generating text responses.
4. **Pinecone** - For storing and retrieving vector embeddings.
5. **Python** - Main programming language.

---

## Prerequisites

1. **Python 3.9 or above**.
2. **Google Generative AI API Key**.
3. **Pinecone API Key and Index name should be - test**.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-repository.git
cd your-repositor

pip install -r requirements.txt

Create a .env file in the root directory and add your API keys:
GOOGLE_GENERATIVE_AI_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key

streamlit run backend_with_ui.py
