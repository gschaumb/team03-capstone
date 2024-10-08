import gradio as gr
from main import graph, AgentState
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initializing the Agent State with clear structure
agent_state = {
    'messages': [],
    'sender': '',
    'perception_1': {'status': None, 'data': None},
    'perception_2': {'status': None, 'data': None},
    'integration_result': {'status': None, 'message': None}
}

def process_user_input(user_input):
    # Add user input to state
    logger.debug("Received user input: %s", user_input)
    
    if user_input:
        agent_state['messages'].append({'sender': 'User', 'content': user_input})
    else:
        logger.error("User input is empty.")
        return "Please provide a valid input.", []

    try:
        logger.debug("Starting graph workflow execution with initial state: %s", agent_state)
        # Execute graph
        events = graph.stream(agent_state)  # Streaming state through the graph

        # Collect intermediate states
        intermediate_states = []
        for event in events:
            # Update the state explicitly to avoid overwriting
            agent_state.update({
                'perception_1': event.get('perception_1', agent_state['perception_1']),
                'perception_2': event.get('perception_2', agent_state['perception_2']),
                'integration_result': event.get('integration_result', agent_state['integration_result'])
            })

            # Log and capture intermediate states
            intermediate_state_info = {
                "state_data": event.get('data', {}),
                "messages": event.get('messages', agent_state['messages']),
                "current_sender": event.get('sender', agent_state['sender'])
            }
            logger.debug("Intermediate state: %s", intermediate_state_info)
            intermediate_states.append(intermediate_state_info)

        # Final response (from IntegrationNode)
        final_response = agent_state['integration_result'].get('message', "No relevant information found.")
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
if __
