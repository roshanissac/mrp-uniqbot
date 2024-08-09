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

# Pre-requisites
Go to the *Experiments/* folder folder of the project and Install the packages/requirements by executing the below command.
   ```
   pip  install -r requirements.txt
   ```
# Exploratory Data Analysis(EDA)
Exploratory Data Analysis of the FAQs data is done in the notebook **EDA.ipynb** under *Experiments/* folder

# Experimentation
Please follow the below steps to replicate the experiments,You have to make sure you have Hugging Face Key  and Open AI key(Stored as Colab Secrets) to do the below steps.
## 1. Generate Test Data
Follow the steps in the notebook **Generate_Test_Data.ipynb** under *Experiments/* folder
## 2. Run Experiments
1. To run the experiments,make sure you uploaded the FAQs and Test dataset to appropriate folder in the Google Drive.
2. Follow the steps in the notebook **Experiments_Hugging_Face_API.ipynb** under *Experiments/* folder
3. Download the ouput *csv files generated from the notebook. Here we have stored these files under *Experiments/datasets/evaluation/results/files* folder.
## 3. Analyze and Visualize Results
Follow the steps in the notebook **Results.ipynb** under *Experiments/* folder

# Running UniQ-Bot
UniQ-Bot requires Open AI key to run  and should be stored in *.env* file.Please follow below steps.
## 1. Ingestion Pipeline(Optional)

Follow the steps in the notebook **RAG_UniQBot.ipynb** under *chatbot/* folder to store FAQs data into the ChromaDB vector Store and to test the RAG pipeline. The vector embeddings will be stored under folder *chroma_db/*.

## 2. Run chatbot UI

Follow the below steps to run the chatbot gradio UI interface.
1. Go to the *chatbot/* folder
2. Execute the below command
   ```
   python main.py
   ```
3. You can open the chat Interface by executing below URL in the browser.
  ```
  http://127.0.0.1:7860/
  ```

# References
1. RAGAS library:[link](https://docs.ragas.io/en/stable/getstarted/index.html)
2. Langchain :[link](https://python.langchain.com/v0.2/docs/introduction/)

