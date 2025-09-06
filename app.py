import streamlit as st
from openai import OpenAI, AuthenticationError

# Set the title of the app
st.title("Simple Chatbot 🤖")

# --- USER AUTHENTICATION ---
try:
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        base_url="https://api.chatanywhere.tech/v1"
    )
except (FileNotFoundError, KeyError):
    st.error("🔑 OPENAI_API_KEY not found in secrets. Please add it to your Streamlit Cloud settings.")
    st.stop()

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            for chunk in stream:
                # --- THIS IS THE SAFETY CHECK ---
                # Check if the chunk and its 'choices' list are valid before access
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta:
                    full_response += chunk.choices[0].delta.content or ""
                    message_placeholder.markdown(full_response + "▌")
            
            # Final check to ensure a response was received
            if not full_response:
                full_response = "Sorry, I received an empty response. Please try again."
            
            message_placeholder.markdown(full_response)
            
        except AuthenticationError:
            st.error("Authentication Error: The API key is invalid. Please check your Streamlit Cloud secrets.")
            st.stop()
        except Exception as e:
            st.error(f"An API error occurred: {e}")
            st.stop()
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
