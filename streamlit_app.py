import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="🦙💬 Code Llama Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Code Llama Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    st.subheader('Model parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=0, max_value=512, value=250, step=1)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=128, step=8)
    
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
# Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = """
    You are a coding assistant.
    """
    #You must follow the following rules strictly in generating your response:
    #1. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. 
    #2. Most importantly don't repeat yourself excessively.
    #3. If you don't know, say you don't know and don't make up stuff.
    #4. Make your response concise, to the point and relevant to the question being asked.
    #5. Whenever you need to generate code please encapsulate the code using ```{code goes here}```
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    #output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
    prompt = """
    <s>[INST]<<SYS>>
    Write code to solve the following coding problem and wrap your code answer, particularly using ``` and ``` before and after the code answer, respectively:
    <</SYS>>[/INST]</s>

    <s>[INST]
    {prompt_input}
    [/INST]</s>
    Assistant:
    """
    output = replicate.run('replicate/codellama-7b-python:0135c7cd0ffb4917e53580562c8d528a46b64102546c27333e8ff8298a85798f',
                           #input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                           input={"prompt": prompt,
                                  "temperature":temperature, "top_p":top_p, "top_k":top_k, "max_length":max_length, "repetition_penalty":1.15})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.write(full_response)
            placeholder.write(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
