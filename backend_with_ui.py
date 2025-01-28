from PyPDF2 import PdfReader
import google.generativeai as genai
import re
from pinecone import Pinecone
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("test")

# Function to generate embeddings using Google Gemini API
def embed_with_gemini(content):
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=content)
        return result['embedding']
    except Exception as e:
        print(f"Error in embedding: {e}")
        return None

# Function to generate a response using Google Generative AI
def generate_response(prompt, context):
    try:
        response = model.generate_content(
        f"User query: {prompt}\nContext: {context} and give answer in text also give/create reference table segments from context."
        )
        return response.text
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "Error generating response."

# Function to extract text and metadata from a PDF file
def extract_pages_from_pdf(pdf_file):
    """
    Extract text and metadata from a PDF file.
    """
    pages = []
    pdf_reader = PdfReader(pdf_file)  
    num_pages = len(pdf_reader.pages)

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        cleaned_text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        pages.append({"page": page_num, "content": cleaned_text})

    return pages

# Function to clean extracted text
def clean(text):
    cleanedlist = []
    for item in text:
        cleaned_text = re.sub(r'\s{2,}', ' ', item['content'])  
        cleaned_text = re.sub(r'\n\s+', '\n', cleaned_text)  
        cleaned_text = re.sub(r'\s+\n', '\n', cleaned_text)  
        cleanedlist.append(cleaned_text)
    return cleanedlist

# Function to upsert embeddings into Pinecone
def upsert_embeddings_to_pinecone(listofdata):
    for i, item in enumerate(listofdata):
        embedding = embed_with_gemini(item)
        if embedding:
            index.upsert([(f"id-{i}", embedding, {"content": item})])  
    print("Data successfully inserted into Pinecone.")

# Function to retrieve similar data from Pinecone
def retrieve_from_pinecone(query, index, top_k=1):
    query_embedding = embed_with_gemini(query)
    if query_embedding:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True  
        )
        return results
    else:
        print("Failed to generate query embedding.")
        return None


# Streamlit frontend UI
def main():
    st.title("Financial Data Q&A")

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")
        print(uploaded_pdf)

        if uploaded_pdf:
            if st.button("Upload and Process"):
                
                
                context = extract_pages_from_pdf(uploaded_pdf)
                listofdata = clean(context)

                
                upsert_embeddings_to_pinecone(listofdata)
                st.success("PDF processed and data stored in Pinecone.")

    # Main page for chat interface
    st.header("Ask Financial Questions")

    query = st.text_input("Enter your financial query:")
    if query:
        # Retrieve relevant data from Pinecone
        retrieved_results = retrieve_from_pinecone(query, index, top_k=5)

        if retrieved_results:
            context_for_response = "\n".join([match['metadata']['content'] for match in retrieved_results['matches']])

            # Generate a response using the retrieved context
            response = generate_response(query, context_for_response)

            # Display the response
            st.subheader("Answer:")
            st.write(response)

            


if __name__ == '__main__':
    main()