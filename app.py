import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# Page configuration
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄"
)

st.title("📄 AI PDF Assistant")
st.write("Upload any PDF and ask questions about its content.")

load_dotenv()


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    # Save uploaded PDF
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")


    # Load PDF
    loader = PyPDFLoader("uploaded_document.pdf")
    documents = loader.load()


    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)


    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Create FAISS
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )


    # Create Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview"
    )


    # Question
    question = st.text_input(
        "Ask a question about your document"
    )


    if question:

        # Retrieve relevant chunks
        results = vectorstore.similarity_search(
            question,
            k=3
        )


        # Combine retrieved text
        context = "\n\n".join(
            result.page_content
            for result in results
        )


        # Prompt
        prompt = f"""
        Answer the question using only the information
        provided in the context.

        Context:
        {context}

        Question:
        {question}

        If the answer is not present in the context, say:
        "I couldn't find the answer in the document."
        """


        # Generate answer
        response = llm.invoke(prompt)


        # Handle Gemini response
        if isinstance(response.content, list):
            answer = response.content[0]["text"]
        else:
            answer = response.content


        # Display answer
        st.subheader("Answer")
        st.write(answer)


        # Display sources
        st.subheader("Sources")

        pages = set()

        for result in results:
            page = result.metadata.get("page", 0) + 1
            pages.add(page)

        for page in sorted(pages):
            st.write(f"📄 Page {page}")