import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage

from pages import expAnalysis
# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('cover.jpg')
display = np.array(display)
st.image(display)
st.title("FedCSIS 2022 Challenge: Predicting the Costs of Forwarding Contracts")
st.text("Expense prediction by Machine Learning")

# col1 = st.columns(1)
# col1, col2 = st.columns(2)
# col1.image(display, width = 400)
# col2.title("Data Storyteller Application")

# Add all your application here
app.add_page("Calculate Expense", expAnalysis.app)
# app.add_page("Detect Disaster Type", detectDisaster.app)


# The main app
app.run()
