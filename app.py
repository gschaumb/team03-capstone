import gradio as gr

def greet(name):
    return "Good to meet you, " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()