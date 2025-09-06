import streamlit as st
from openai import OpenAI, AuthenticationError

# Set the title of the app
st.title("Simple Chatbot 🤖")

# --- USER AUTHENTICATION ---
# Check for the API key in Streamlit's secrets
try:
    # This section is modified to include the base_url
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        base_url="https://api.chatanywhere.tech/v1"  # This line redirects requests
    )
except (FileNotFoundError, KeyError):
    st.error("🔑 OPENAI_API_KEY not found in secrets. Please add it to your Streamlit Cloud settings.")
    st.stop()

# --- CHAT LOGIC ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept and process user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Create the chat completion stream
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            # Stream the response to the placeholder
            for chunk in stream:
                full_response += chunk.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        # Handle authentication errors specifically
        except AuthenticationError:
            st.error("Authentication Error: The API key is invalid. Please check your Streamlit Cloud secrets.")
            st.stop()
        # Handle other potential API errors
        except Exception as e:
            st.error(f"An API error occurred: {e}")
            st.stop()
            
    # Add the final assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
