import gradio as gr
from main import perception_node_1, perception_node_2, integration_node, AgentState
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initializing the Agent State
agent_state = {
    "messages": [],
    "sender": "",
    "perception_1": {"status": None, "data": None},
    "perception_2": {"status": None, "data": None},
    "integration_result": {"status": None, "message": None},
}


# Function to process_user_input (TBD changes to keep or reset state for each query for future persistence functionality)
def process_user_input(user_input):
    logger.debug("Received user input: %s", user_input)

    if user_input:
        # Append the new message to the state
        agent_state["messages"].append({"sender": "User", "content": user_input})

        # Reset perceptions and integration result for the new query
        agent_state["perception_1"] = {"status": None, "data": None}
        agent_state["perception_2"] = {"status": None, "data": None}
        agent_state["integration_result"] = {"status": None, "message": None}
    else:
        logger.error("User input is empty.")
        return "Please provide a valid input.", []

    try:
        logger.debug("Starting agent workflow with initial state: %s", agent_state)

        # Run through perception nodes
        current_state = perception_node_1(agent_state)
        current_state = perception_node_2(current_state)

        # Run the integration node
        current_state = integration_node(current_state)

        # Update the global agent state with the final state
        agent_state.update(current_state)

        # Check if the integration result has a valid message
        final_response = (
            current_state["integration_result"]["message"]
            or "No relevant information found."
        )

        logger.debug("Final response to be returned: %s", final_response)
        return final_response, current_state

    except Exception as e:
        logger.error("Error occurred during processing of user input: %s", e)
        return "An error occurred while processing your request. Please try again.", {}


# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# Enron Agentic RAG Demo")

    user_input = gr.Textbox(
        placeholder="Ask a question about the Enron case...", label="Your Query"
    )
    output = gr.Textbox(label="Agent Response")
    state_output = gr.JSON(label="Agent Intermediate State Data", value={})

    def update_state_ui(user_query):
        response, intermediate_states = process_user_input(user_query)
        return response, intermediate_states

    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=update_state_ui, inputs=user_input, outputs=[output, state_output]
    )

# Launch Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
