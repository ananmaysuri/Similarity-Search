from streamlit_utils import *
import streamlit as st
import io
from PIL import Image
st.set_page_config(
    page_title = 'Similarity Search',
    page_icon = '🔍'
)
st.header("Similarity Search - Major Project Testing")
imageText = st.text_input("Input Query Text")
N = st.text_input("Input Page Number")
if imageText:
    with st.spinner(text = 'Getting Images from Google and Sorting with CLIP ...'):
        
        imgSimScore, upSplashImages = getSortedQuery(imageText, N)
        images = [linkToImage(img) for img, score in imgSimScore]
        simScore = [f'Sim Score: {score:.2f}' for img, score in imgSimScore]
        upSplashImages = [linkToImage(img) for img in upSplashImages]
        upSplashIx = [i+1 for i in range(len(upSplashImages))]
        col1, col2 = st.beta_columns(2)
        col1.header("Similarity Search")
        col1.image(images, width = 300, caption = simScore)
        col2.header("Images from Google")
        col2.image(upSplashImages, width = 300, caption = upSplashIx)
