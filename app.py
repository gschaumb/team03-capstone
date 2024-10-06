import gradio as gr

def greet(name):
    return "It is very nice to meet you, Dr. " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
