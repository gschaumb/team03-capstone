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


# Function to process user input
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
        return "Please provide a valid input.", [], {}

    try:
        logger.debug("Starting agent workflow with initial state: %s", agent_state)

        # Run through perception nodes
        current_state = perception_node_1(agent_state)
        current_state = perception_node_2(current_state)

        # Run the integration node
        current_state = integration_node(current_state)

        # Get the summaries generated by the integration agent
        summaries = current_state["integration_result"]["message"] or [
            "No relevant information found."
        ]
        logger.debug("Final response to be returned: %s", summaries)

        # Provide shortened labels for each summary in the dropdown
        dropdown_choices = ["Summary 1", "Summary 2", "Summary 3"]
        return dropdown_choices, summaries, current_state

    except Exception as e:
        logger.error("Error occurred during processing of user input: %s", e)
        return "An error occurred while processing your request.", [], {}


# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# Enron Agentic RAG Demo")

    user_input = gr.Textbox(
        placeholder="Ask a question about the Enron case...", label="Your Query"
    )
    output = gr.Textbox(label="Agent Response")
    summary_dropdown = gr.Dropdown(choices=[], label="Select a Summary for Evaluation")
    state_output = gr.JSON(label="Agent Intermediate State Data", value={})

    def update_state_ui(user_query):
        dropdown_choices, summaries, intermediate_states = process_user_input(
            user_query
        )
        return dropdown_choices, intermediate_states

    def evaluate_summary(summary_choice, summaries):
        index = (
            int(summary_choice.split(" ")[-1]) - 1
        )  # Extract index from "Summary 1", "Summary 2", etc.
        return summaries[index]

    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=update_state_ui, inputs=user_input, outputs=[summary_dropdown, state_output]
    )

    # The dropdown will now display shortened choices, and return the actual selected summary
    summary_dropdown.change(
        fn=evaluate_summary, inputs=[summary_dropdown, state_output], outputs=output
    )

# Launch Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
