from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

class UniqBot:
    def __init__(self,load_vectordb:bool=True):

        #Loading environment variables from .env
        load_dotenv()

        self.llm = ChatOpenAI( 
                    model="gpt-4o-mini",
                    temperature=.1,
                    max_tokens=500,
                    verbose=True,
                    model_kwargs={"top_p":0.5}
                    )
        self.embedding=OpenAIEmbeddings(model="text-embedding-3-large")

        vectordb_directory='chroma_db/'

        vectordb=Chroma(
            collection_name="faqs",
            persist_directory=vectordb_directory,
            embedding_function=self.embedding


        )
        self.vectordb=vectordb

        self.load_chain()

    def load_chain(self):
        template = """SYSTEM:You are an intelligent assistant helping Toronto Metropolitan University Website visitors on their frequently asked questions in English.

        Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

        Do not try to make up an answer:
        -if the answer to the question cannot be determined from the context alone or if the context is empty,
        only say "I cannot determine the answer to that"
        -Use numbered lists when possible

        Context:
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
        greetings = ["Hello", "Hi", "Good Morning", "Good Evening",""]

        result = self.qa({"question": question})

        output=result["answer"].strip()

        if any(phrase in result["answer"].lower() for phrase in phrases_to_check):
            return '\nPlease ask a relevant question to UniQ-Bot.'
        else:
            try:
                url_link=result["source_documents"][0].metadata['question_url']
                url=f"<br>\n\nView this link for more information: {url_link}"
            except KeyError:
                url=""
            return output+url