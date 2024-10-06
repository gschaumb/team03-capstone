import gradio as gr
from main import process_data  # Import function from main.py

def create_interface():
    input_text = gr.Textbox(label="Input Text")
    output_text = gr.Textbox(label="Output Text")
    
    def process_text(input):
        return input.upper()

    return gr.Interface(fn=process_text, inputs=input_text, outputs=output_text)
    
create_interface().launch()
