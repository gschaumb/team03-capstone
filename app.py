import gradio as gr
from main import graph, AgentState

# Initialize shared state
state = AgentState()

# Gradio Function for Handling User Input
def process_user_input(user_input, conversation_history=[]):
    # Update conversation history with user input
    conversation_history.append(f"User: {user_input}")

    # Update agent state with the new input
    state.messages.append({'sender': 'User', 'content': user_input})

    # Execute the graph based on input and workflow logic
    events = graph.stream({'messages': state.messages}, {'recursion_limit': 50})

    # Intermediate state information
    perception_agent_responses = []
    integration_response = ""

    # Process events and update conversation history
    for event in events:
        sender = event.get("sender", "Agent")
        content = event.get("content", "Processing...")

        # Collect intermediate data based on agent types
        if "PerceptionNode" in sender:
            perception_agent_responses.append(f"{sender}: {content}")
        elif "IntegrationNode" in sender:
            integration_response = content
        
        # Append to conversation history
        conversation_history.append(f"{sender}: {content}")

    # Append intermediate state information to the conversation
    if perception_agent_responses:
        perception_info = "\n".join(perception_agent_responses)
        conversation_history.append(f"Intermediate Perception Agent Results:\n{perception_info}")

    if integration_response:
        conversation_history.append(f"Intermediate Integration Agent Result:\n{integration_response}")

    # Return updated conversation
    return "\n".join(conversation_history), conversation_history

# Define the Gradio UI Elements
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="Ask a question about Enron")
    conversation_history = gr.State([])

    user_input.submit(process_user_input, inputs=[user_input, conversation_history], outputs=[chatbot, conversation_history])

# Launch the Gradio App
demo.launch()
