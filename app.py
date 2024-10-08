import gradio as gr
from main import graph, AgentState
import logging

# Configure logging to print to stdout for hosted environments
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initializing the Agent State
agent_state = AgentState()

def process_user_input(user_input):
    # Add user input to state
    logger.debug("Received user input: %s", user_input)
    agent_state.messages.append({'sender': 'User', 'content': user_input})

    try:
        logger.debug("Starting graph workflow execution with initial state: %s", agent_state.__dict__)
        # Execute graph
        events = graph.stream(agent_state.__dict__)  # Streaming state through the graph

        # Collect intermediate states
        intermediate_states = []
        for event in events:
            # Capture and log each intermediate state
            intermediate_state_info = {
                "state_data": event['data'],
                "messages": event['messages'],
                "current_sender": event['sender']
            }
            logger.debug("Intermediate state: %s", intermediate_state_info)
            intermediate_states.append(intermediate_state_info)

        # Final response (assuming IntegrationNode is the endpoint that gives a response)
        final_response = agent_state.data['integration_result']['message']
        if not final_response:
            final_response = "No relevant information found."

        logger.debug("Final response: %s", final_response)

    except Exception as e:
        logger.error("Error occurred during processing of user input: %s", e)
        final_response = "An error occurred while processing your request. Please try again."

    return final_response, intermediate_states

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Agent Collaboration Demo")

    user_input = gr.Textbox(placeholder="Ask a question about the Enron case...", label="Your Query")
    output = gr.Textbox(label="Agent Response")
    state_output = gr.JSON(label="Intermediate State Data", value={})

    def update_state_ui(user_query):
        response, intermediate_states = process_user_input(user_query)
        return response, intermediate_states

    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=update_state_ui, inputs=user_input, outputs=[output, state_output])

# Launch Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
