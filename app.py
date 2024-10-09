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
    logger.debug("Received user input: %s", user_input)
    
    if user_input:
        agent_state['messages'].append({'sender': 'User', 'content': user_input})
    else:
        logger.error("User input is empty.")
        return "Please provide a valid input.", []
    
    try:
        logger.debug("Starting graph workflow execution with initial state: %s", agent_state)
        
        # Execute the graph workflow
        events = graph.stream(agent_state)  # Stream through the state graph

        intermediate_states = []
        for event in events:
            logger.debug("Received event with state data: %s", event)

            # Update the agent_state with intermediate results
            agent_state['perception_1'] = event.get('perception_1', agent_state['perception_1'])
            agent_state['perception_2'] = event.get('perception_2', agent_state['perception_2'])
            agent_state['integration_result'] = event.get('integration_result', agent_state['integration_result'])

            logger.debug("Updated agent state after event: %s", agent_state)

            intermediate_states.append({
                "state_data": event.get('data', {}),
                "messages": agent_state['messages'],
                "current_sender": agent_state.get('sender', "")
            })

        # Check if the integration result has a valid message
        if not agent_state.get('integration_result', {}).get('message'):
            logger.error("Integration result message is empty or missing.")
            final_response = "No relevant information found."
        else:
            final_response = agent_state['integration_result']['message']

        logger.debug("Final response to be returned: %s", final_response)
        return final_response, intermediate_states
    
    except Exception as e:
        logger.error("Error occurred during processing of user input: %s", e)
        return "An error occurred while processing your request. Please try again.", []

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
