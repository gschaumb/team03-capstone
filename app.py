import gradio as gr
from main import process_data  # Import function from main.py

def create_interface():
    input_text = gr.Textbox(label="Input Text")
    output_text = gr.Textbox(label="Output Text")

    return gr.Interface(fn=process_data, inputs=input_text, outputs=output_text)
    
create_interface().launch()
