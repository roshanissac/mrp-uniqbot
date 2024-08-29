import gradio as gr
import random
from uniq_bot import UniqBot

bot=UniqBot()

def predict(message, history):
    response=bot.ask(message)
    return response

gr.ChatInterface(predict,title="UniQ-Bot FAQ Assistant!").launch()

