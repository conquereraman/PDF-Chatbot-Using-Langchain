import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
with st.sidebar:
    st.title("Chat App")
    add_vertical_space(5)
    st.write("Made with langchain")


def main():
    load_dotenv()
    st.write("Hello")
    # uploading pdf
    pdf = st.file_uploader("Upload your pdf", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            # st.write("Embeddings Completed")

        # Accept user question
        query = st.text_input("Ask Questions Or Query About Your Document")
        # st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
            # st.write(docs)


if __name__ == '__main__':
    main()
