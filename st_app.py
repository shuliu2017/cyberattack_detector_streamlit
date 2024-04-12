## build a streamlit app to show all the results

# Path: edge-iiot/st_app.py
import io
import pandas as pd
import dill
import streamlit as st
from  helper import data_type
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

trained_model = dill.load(open('dt_model.dill', 'rb'))

# @st.cache_data(experimental_allow_widgets=True) 
def main():
    
    ## write a front end view for the app
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">CyberSecurity Detector!</h2>
    </div>
    """
    ## display the front end aspect
    st.markdown(html_temp,unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        st.write('Hello Human👋, I am a Cyberattack Detector developed by the LYL team.')
    
    file = st.file_uploader("Upload file", type=["csv"])
    if file is not None:
        try:
            ## read the uploaded csv file
            
            X = pd.read_csv(file, low_memory=False, dtype=data_type)
            selected_features = ['icmp.checksum', 'icmp.seq_le', 'http.content_length', 'http.request.method', 'http.referer',
                                 'http.request.version', 'http.response', 'tcp.ack', 'tcp.ack_raw', 'tcp.checksum', 'tcp.len', 'tcp.seq',
                                 'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn', 'tcp.connection.synack', 'tcp.flags', 'tcp.flags.ack',
                                 'udp.stream', 'udp.time_delta', 'dns.qry.name', 'dns.qry.name.len', 'dns.qry.qu', 'dns.retransmission', 'dns.retransmit_request',
                                 'mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
                                 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.topic', 'mqtt.topic_len', 'mqtt.ver']
            X = X[selected_features]
            st.subheader('Data')
            with st.chat_message("assistant"):
                st.write('Let us take a look of the data')
            st.write(X.head())
        except Exception as e:
            st.write(str(e))

    st.subheader('EDA')
        
    if st.button('Analyze data'):
        with st.chat_message("assistant"):
            st.write('Correlations of numerical features')
        # selected_numerical_features = ['icmp.seq_le', 'icmp.checksum', 'http.content_length', 'tcp.connection.rst', 'tcp.ack', 'mqtt.topic_len']
        # correlation_matrix = X[selected_numerical_features].corr()
        numerical_features = ['icmp.checksum', 'icmp.seq_le', 'tcp.ack', 'tcp.ack_raw', 
                      'tcp.checksum', 'tcp.len', 'tcp.seq', 'udp.stream', 'dns.qry.name',
                      'http.content_length',  'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn',
                      'tcp.connection.synack', 'tcp.flags', 'tcp.flags.ack',
                      'udp.time_delta', 'dns.qry.qu', 'dns.retransmission', 'dns.retransmit_request',
                      'mqtt.conflag.cleansess', 'mqtt.hdrflags', 'mqtt.len',  'mqtt.msgtype', 'mqtt.proto_len',
                      'mqtt.topic_len', 'mqtt.ver', 'mqtt.conflags']
        correlation_matrix = X[numerical_features].corr()
        
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig)
        plt.clf()
        
        with st.chat_message("assistant"):
            st.write('Counts of categorical features')
        # selected_categorical_features = ['mqtt.protoname', 'http.response', 'http.request.method', 'mqtt.conack.flags']
        categorical_features = ['http.request.method', 'http.referer', 'http.response', 'mqtt.conack.flags', 'mqtt.protoname',
                                'mqtt.topic', 'http.request.version', 'dns.qry.name.len']
        for i, col in enumerate(categorical_features):
            fig, ax = plt.subplots(1, 1)
            X[col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
            plt.clf() 
  
    st.subheader('Prediction') 
    if st.button('Predict'):
        try:
            with st.chat_message("assistant"): 
                st.write('Start predicting')
            predictions = trained_model.predict(X)
            
            prediction_df = pd.DataFrame({'Attack_type': predictions})
            df_transposed = prediction_df.T
            with st.chat_message("assistant"): 
                st.write('Prediction results')
            st.write(df_transposed)
            
            with st.chat_message("assistant"): 
                st.write('Normal vs. Attack')
            normal_count = (predictions == 'Normal').sum()
            attack_count = len(predictions) - normal_count
            ax = pd.DataFrame({'Normal': [normal_count], 'Attack': [attack_count]}).plot(kind='bar')
            fig = ax.figure
            st.pyplot(fig)
            plt.clf()
            
            
            with st.chat_message("assistant"): 
                st.write('Attack Types')
            ax2 = pd.DataFrame({'Attack_type':predictions}).Attack_type.value_counts().sort_values(ascending=True).plot(kind='barh')
            fig2 = ax2.figure
            st.pyplot(fig2)
            plt.clf()
            
            with st.chat_message("assistant"): 
                st.write('Attack Types Percentage')
            ax3 = pd.DataFrame({'Attack_type':predictions}).Attack_type.value_counts().sort_values(ascending=True).plot(kind='pie', autopct='%.2f%%')
            fig3 = ax3.figure
            st.pyplot(fig3)
            plt.clf()
            
                        
            if attack_count == 0:
                with st.chat_message("assistant"):
                    st.write('Great! Your network is safe!')
            else:
                with st.chat_message("assistant"):
                    st.write('Danger! Your network is under cyberattack!')
                                    
                st.write('Here is somethings you can do. You can also ask me for help!')
                st.markdown(f"[{'Cyberattack'}]({'https://en.wikipedia.org/wiki/Cyberattack'})", unsafe_allow_html=True)
                st.markdown(f"[{'Denial-of-service attack'}]({'https://en.wikipedia.org/wiki/Denial-of-service_attack'})", unsafe_allow_html=True)
                st.markdown(f"[{'Man-in-the-middle attack'}]({'https://en.wikipedia.org/wiki/Man-in-the-middle_attack'})", unsafe_allow_html=True)
                st.markdown(f"[{'Code injection'}]({'https://en.wikipedia.org/wiki/Code_injection'})", unsafe_allow_html=True)
                st.markdown(f"[{'Malware'}]({'https://en.wikipedia.org/wiki/Malware'})", unsafe_allow_html=True)
                st.markdown(f"[{'Ransomware'}]({'https://en.wikipedia.org/wiki/Ransomware'})", unsafe_allow_html=True)
                st.markdown(f"[{'TCP/IP stack fingerprinting'}]({'https://en.wikipedia.org/wiki/TCP/IP_stack_fingerprinting'})", unsafe_allow_html=True)
                st.markdown(f"[{'Port scanner'}]({'https://en.wikipedia.org/wiki/Port_scanner'})", unsafe_allow_html=True)

            
            # st.write(cr)
            
        except Exception as e:
            st.write(str(e))
            

    st.subheader("Chat with me")

    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
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
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
    


if __name__ == '__main__':
    main()
    
    
