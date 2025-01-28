from PyPDF2 import PdfReader
import google.generativeai as genai
import re
from pinecone import Pinecone
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
        f"User query: {prompt}\nContext: {context} "
        )
        return response.text
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "Error generating response."

# Function to extract text and metadata from a PDF file
def extract_pages_from_pdf(pdf_path):
    pages = []
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
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

# Main function to execute the pipeline
def main():
    pdf_path = "Sample Financial Statement.pdf"  

    # Extract text and metadata from the PDF
    context = extract_pages_from_pdf(pdf_path)

    # Clean the extracted text
    listofdata = clean(context)

    # Upsert embeddings to Pinecone
    upsert_embeddings_to_pinecone(listofdata)

    # Example query to search for relevant data
    query = "INFOSYS LIMITED  Total other comprehensive income /(loss), net of tax "

    # Retrieve relevant data from Pinecone
    retrieved_results = retrieve_from_pinecone(query, index)

    # Generate the response using the retrieved results as context
    if retrieved_results:
        context_for_response = "\n".join([match['metadata']['content'] for match in retrieved_results['matches']])

        # Call the generate_response function
        response = generate_response(query, context_for_response)

        # Display the response
        print("Generated Response:")
        print(response)


if __name__ == '__main__':
    main()
