import gradio as gr
import logging
from main import graph, AgentState

# Configure logging for hosted environment
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to process user input and run through the LangGraph workflow
def process_user_input(user_input):
    logger.debug("Received user input: %s", user_input)

    # Instantiate initial state
    state = AgentState()
    state.messages.append({"sender": "User", "content": user_input})

    try:
        # Execute the graph workflow with the given state
        logger.debug("Starting graph workflow execution with initial state: %s", state.__dict__)
        
        events = graph.stream(state)

        # Iterate through the workflow events and log each event
        for event in events:
            logger.debug("Event in workflow: %s", event)

        # Log the final state after processing the entire workflow
        logger.debug("Final state after workflow execution: %s", state.__dict__)

        # Extract the response from the state and provide feedback to the user
        integration_result = state.data.get('integration_result', "Could not generate a response.")
        logger.debug("Integration result: %s", integration_result)
        return integration_result, state

    except Exception as e:
        # Log any errors that occur during the workflow
        logger.error("Error occurred during processing of user input: %s", e)
        return "An error occurred during processing. Please try again.", None

# Creating the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Legal & Financial Analysis Chatbot")
    gr.Markdown(
        """
        This chatbot helps answer questions related to legal and financial data.
        It uses advanced language models and multi-agent collaboration to provide you with the best possible insights.
        """
    )
    
    # Input field for user query
    with gr.Row():
        user_input = gr.Textbox(lines=2, placeholder="Ask about Enron's legal and financial issues...")
        submit_button = gr.Button("Submit")
    
    # Output areas for the main response and intermediate details
    with gr.Row():
        output_response = gr.Textbox(label="Response", lines=5)
        intermediate_info = gr.Textbox(label="Intermediate State Information", lines=10)
    
    # Define action to be performed on clicking Submit button
    def interface_action(user_query):
        response, state = process_user_input(user_query)
        
        # Logging state information directly for intermediate inspection
        if state is not None:
            try:
                state_debug_info = (
                    f"Messages: {state.messages}\n"
                    f"Data: {state.data}\n"
                    f"Sender: {state.sender}"
                )
            except Exception as e:
                state_debug_info = f"Error fetching intermediate state: {e}"
                logger.error(state_debug_info)
        else:
            state_debug_info = "State could not be fetched due to an error during processing."

        return response, state_debug_info

    # Connect user input, button, and output display
    submit_button.click(fn=interface_action, inputs=user_input, outputs=[output_response, intermediate_info])

# Launch Gradio Interface
if __name__ == "__main__":
    demo.launch()
