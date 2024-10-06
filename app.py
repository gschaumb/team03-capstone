import gradio as gr
from main import process_data  # Import function from main.py

def create_interface():
    # Define the Gradio interface here
    input_text = gr.inputs.Textbox(label="Input Text")
    output_text = gr.outputs.Textbox(label="Output Text")

    # Create the interface and link it to the process_data function
    interface = gr.Interface(fn=process_data, inputs=input_text, outputs=output_text, title="Text Processor")
    return interface

if __name__ == "__main__":
    # Launch the Gradio app
    create_interface().launch()
