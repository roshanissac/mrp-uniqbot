# UniQ-Bot: An Advanced LLM-Powered Intelligent Conversational Assistant for Swift and Accurate Frequently Answered Questions (FAQs) Assistance in Universities
In this project, I evaluated 4 open-source LLMs and embedding models by creating RAG(Retrieval
Augmented Generation) pipeline and employed RAGAS quantitative metrics to evaluate them. I also
developed an intelligent conversational assistant called UniQ-Bot that aims to address the challenge faced
by current and prospective students who spend considerable time navigating the Toronto Metropolitan
University(TMU) website in search of answers to common queries. Despite the existence of a Fre-
quently Asked Questions (FAQs) section on the university’s website, the process of locating relevant infor-
mation swiftly remains cumbersome. So, to enhance user experience and streamline information retrieval
UniQ-Bot is helpful.
The selected open source LLMs for the evaluation include Llama-2-13b-chat-hf, Meta-Llama-3-70B-
Instruct,Mixtral-8x7B-Instruct-v0.1, and Nous-Hermes-2-Mixtral-8x7B-DPO and the embed-
ding models all-mpnet-base-v2,e5-small-v2, bge-small-en-v1.5, snowflake-arctic-embed-s.The
chosen framework for building the RAG workflow is the LangChain framework, which facilitates the
seamless integration between LLMs, Embedding models, and vector database. To enhance the efficiency of
information storage and retrieval, ChromaDB vector database is employed. This database will store em-
beddings that contribute to the contextual understanding and responsiveness of RAG-based systems. UniQ-
Bot is developed using OpenAI’s ChatGPT-4o-mini LLM and the embedding model text-embedding-3-
large

# Data
For the project, we used a custom data set web-scraped from the Toronto Metropolitan University website. We used the Tool ParseHub to scrape the FAQs from the corresponding web pages of the university. Not all FAQs were selected for this Project, but FAQS from around 32 departments where taken. The dataset consists of 984 rows of FAQs from different departments of the university.

The data stored under *Experiments/datasets/* folder,
1. FAQs Data-*Experiments/datasets/ingestion/preprocessed/combined_faqs_preprocessed.csv*
2. Test/Evaluation Data- *Experiments/datasets/evaluation/test/eval_data.csv*
3. Final Results Data- *Experiments/datasets/evaluation/results/eval_results.csv*

# Exploratory Data Analysis(EDA)
Exploratory Data Analysis of the FAQs data is done in the notebook **EDA.ipynb** under *Experiments/* folder

# Experimentation
Please follow the below steps to replicate the experiments,
1. Generate Test Data
Follow the steps in the notebook **Generate_Test_Data.ipynb** under *Experiments/* folder
2. Run Experiments
You have to make sure you have 


## Step 
