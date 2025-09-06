import streamlit as st
import openai

# Set the title of the app
st.title("Simple Chatbot 🤖")

# Securely fetch the API key from Streamlit's secrets management
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create a .streamlit/secrets.toml file.")
    st.stop()
except KeyError:
    st.error("OPENAI_API_KEY not found in secrets. Please add it to your secrets file.")
    st.stop()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Create the chat completion request
            for response in openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except openai.error.AuthenticationError:
            st.error("Authentication Error: Your API key is invalid or has been revoked. Please check your secrets file.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
