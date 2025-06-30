from random import random

import gradio as gr

def greet_number(name):
    greeting = "Hello " + name
    lucky_number = random.randint(1, 100)
    return greeting, lucky_number

# Custom theme with a blue background
demo = gr.Interface(
    fn=greet_number,
    inputs=gr.Textbox(label="Name", placeholder="Enter Name Here"),
    outputs=[gr.Textbox(label="Greeting"), gr.Number(label="Your Lucky Number")],
    title="GreetLucky",
    css=".gradio-container {background: rgb(0, 166, 228)}"
)
demo.launch()