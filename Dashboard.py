from API import pd
from API import np
import altair as alt
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

id=st.text_input("Your ID", key="ID")
if len(id) != 0:
       
        pred=requests.post(url = "http://127.0.0.1:8000/predict/{}".format(id)).json()
        if pred==0:
            st.write("Your credit has been denied")
        else: st.write("Your credit has been accepted")
            
        score=requests.post(url = "http://127.0.0.1:8000/score/{}".format(id)).json()
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = score,
            mode = "gauge+number",
            title = {'text': "Your score"},
            gauge = {'axis': {'range': [None, 1]},
                     'bar': 
                     {'color': "darkblue"},
                     'steps' : [
                         {'range': [0, 0.5], 'color': "red"},
                         {'range': [0.5, 1], 'color': "green"}]}))

        st.plotly_chart(fig, use_container_width=True)
    
        localfi=requests.post(url = "http://127.0.0.1:8000/localfi/{}".format(id)).json()
        localfi = pd.DataFrame.from_dict(localfi)
        localfi=localfi.sort_values(by='val', ascending=False)
        bar_chart = alt.Chart(localfi, title="Feature importance for one client").mark_bar().encode(
            y='val',
            x=alt.X('col',sort='-y'),
        )
 
        st.altair_chart(bar_chart, use_container_width=True)
        st.bar_chart(data=localfi, x='col', y='val', use_container_width=True)
        
        gfi=requests.post(url = "http://127.0.0.1:8000/gfi/{}".format(id)).json()
        gfi = pd.DataFrame.from_dict(gfi)
        bar_chart2 = alt.Chart(gfi, title="Feature importance for all clients").mark_bar().encode(
            y='val',
            x=alt.X('col',sort='-y'),
        )       
        st.altair_chart(bar_chart2, use_container_width=True)
        st.bar_chart(data=gfi, x='col', y='val', use_container_width=True)
        data = pd.read_csv('clients_list_predict.csv')
        
        class_0 = data[data['Predicted'] == 0]
        class_1 = data[data['Predicted'] == 1] 
        option = st.selectbox('Which parameter do you want to see?', localfi['col'])
        fig = px.histogram(data, x=option, color="Predicted")
        st.plotly_chart(fig, use_container_width=True)
