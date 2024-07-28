import gradio as gr
import random
from uniq_bot import UniqBot


def predict(message, history):

    bot=UniqBot()
    response=bot.ask(message)
    return response

gr.ChatInterface(predict).launch()