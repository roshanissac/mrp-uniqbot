from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import re
from langchain_community.document_loaders import DataFrameLoader ,WebBaseLoader   
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
class UniqBot:
    def __init__(self):

        #Loading environment variables from .env
        load_dotenv(override=True)

        self.llm = ChatOpenAI( 
                    model="gpt-4o-mini",
                    temperature=.1,
                    max_tokens=500,
                    verbose=True,
                    model_kwargs={"top_p":0.5}
                    )
        self.embedding=OpenAIEmbeddings(model="text-embedding-3-large")

        vectordb_directory='./chroma_db'

        def load_data_from_dataframe(file_path):
            df = pd.read_csv(file_path)
            loader = DataFrameLoader(df, page_content_column="document")
            return loader.load()
        
        docs=load_data_from_dataframe("Experiments/datasets/ingestion/preprocessed/combined_faqs_preprocessed.csv")

        # split the docs into chunks using recursive character splitter
        def split_docs(documents,chunk_size=1500,chunk_overlap=200,type='csv'):
            separators=None
            if type=="web":
                print("Type is Web...")
                separators=["\n\n\n","\n\n","\n","(?<=\.)",""," "]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=separators)
            chunks = text_splitter.split_documents(documents)
            return chunks

        # store the splitte documnets in docs variable
        chunks = split_docs(documents=docs,chunk_size=1500,chunk_overlap=200)

        # vectordb=Chroma(
        #     collection_name="faqs",
        #     persist_directory=vectordb_directory,
        #     embedding_function=self.embedding


        # )
        vectordb = Chroma.from_documents(collection_name="faqs",documents=chunks, embedding=self.embedding,persist_directory="./chroma_db")
        self.vectordb=vectordb

        self.load_chain()

    def load_chain(self):

        print("Loading Chain")
        # template = """SYSTEM:You are an intelligent assistant helping Toronto Metropolitan University Website visitors on their frequently asked questions in English.

        # Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

        # Do not try to make up an answer:
        # -if the answer to the question cannot be determined from the context alone or if the context is empty,
        # only say "I cannot determine the answer to that"
        # -Use numbered lists when possible

        # Context:
        # =============
        # {context}
        # =============

        # Question: {question}

        # Helpful Answer:"""

        template = """SYSTEM:You are an intelligent assistant helping Toronto Metropolitan University Website visitors on their frequently asked questions in English.

        Question: {question}

        Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

        Do not try to make up an answer:
        -if the answer to the question cannot be determined from the context alone or if the context is empty,
        say "I cannot determine the answer to that"
        -Use numbered lists when possible

        Always finish with a new line that says "\nYours Truly - UniQBot 1.0"
        =============
        {context}
        =============

        Question: {question}

        Helpful Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        
        memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key='answer'
        )

        # retriever = self.vectordb.as_retriever(search_kwargs={"k": 3,"score_threshold":.40}, search_type="similarity_score_threshold")
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 3}, search_type="similarity")

        qa = ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question = True,
        verbose=False,
        chain_type="stuff",
        combine_docs_chain_kwargs={'prompt': prompt},
        )

        self.qa=qa

    def reset_memory(self):
        self.qa.memory.clear()


    def ask(self,question: str):

        phrases_to_check = ["i cannot determine the answer to that", "i do not know the answer to that", "i can help you with a variety of tasks", "i am uniq-bot 1.0"]
        # Define a list of greeting keywords
        greetings = [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b", r"\bwhat's up\b", 
            r"\bhowdy\b", r"\bgood morning\b", r"\bgood afternoon\b", r"\bgood evening\b"
        ]

        # Combine all conclusion into a single regular expression
        greeting_pattern = re.compile("|".join(greetings), re.IGNORECASE)

        # Define a list of greeting keywords
        conclusion = [
            r"\bthanks\b", r"\bthank you\b", r"\bthank you so much\b",r"\bbye\b"
        ]

        # Combine all conclusion into a single regular expression
        conclusion_pattern = re.compile("|".join(conclusion), re.IGNORECASE)


        result = self.qa.invoke({"question": question})


        output=result["answer"].strip()

        print("Output...")
        print(output)

    
        if greeting_pattern.search(question):
            return '\n Hello there! How can I help you today?'
        if conclusion_pattern.search(question):
            return '\n You are welcome!'
        if any(phrase in result["answer"].lower() for phrase in phrases_to_check):
            print("OOD")
            return '\nPlease ask a relevant question to UniQ-Bot.'
        else:
            try:
                url_link=result["source_documents"][0].metadata['question_url']
                url=f"<br>\n\nView this link for more information: {url_link}"
            except KeyError:
                url=""
            return output+url