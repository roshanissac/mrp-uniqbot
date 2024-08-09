import gradio as gr
import random
from uniq_bot import UniqBot

bot=UniqBot()

def predict(message, history):

   
    response=bot.ask(message)
    return response

gr.ChatInterface(predict,

chatbot=gr.Chatbot(placeholder="<strong>Welcome to UniQBot!</strong><br>Ask Me Anything related to Toronto Metropolitan Universities FAQs",
min_width=200,
height=400,
),



).launch()

