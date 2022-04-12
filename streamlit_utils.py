import torch
import clip
from PIL import Image
import os
import re
from tqdm import tqdm, trange
import random
import requests
import numpy as np
import streamlit as st
from serpapi import GoogleSearch
global model, preprocess, device
device = 'cpu'
model, preprocess = clip.load("ViT-B/32", device = 'cpu')

def getImagesFromGoogle(total: int, query_text: str):
    num_page = 1
    imgs_total = total
    query = query_text
    regUrls = []
    search = GoogleSearch({"q": query, "tbm": "isch", "num": imgs_total})
    for image_result in search.get_dict()['images_results']:
        link = image_result["original"]
        regUrls.append(link)
    return regUrls

def getImagesFromUnsplash(total: int, query_text: str):
    '''
    Images from query text
    '''
    num_page = 1
    imgs_total = total
    query = query_text
    UPSPLASH_API_KEY = "8_cSFjMpnN14fMUOZF_UJG0RsM9Y7s4dtVfBgCx5_rk"
    url = f"https://api.unsplash.com/search/photos?query={query_text}&page={num_page}&per_page={imgs_total}"
    headers = {
        "Authorization": f"Bearer Client-ID {UPSPLASH_API_KEY}",
    }
    req = requests.get(url, headers = headers)
    resp = req.json()
    regUrls = [r['urls']['regular'] for r in resp['results']]
    return regUrls

def linkToImage(link):
    '''
    Image URL to PIL.Image
    '''
    content = requests.get(link, stream = True)
    content = content.raw
    img = Image.open(content)
    return img

@st.cache(show_spinner=False)
def getImageTextSimScore(link, text):
    '''
    Compute similarity score from image feature
    and text feature
    '''
    image = preprocess(linkToImage(link)).unsqueeze(0).to(device)
    tokenizedText = clip.tokenize(text).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenizedText)
    
    simScore = torch.matmul(text_features, image_features.T)[0][0]
    return simScore.item()

@st.cache(show_spinner=False)
def getSortedQuery(text, N):
    '''
    Get images from text and sort
    using similarity score
    '''
    upSplashImages = getImagesFromGoogle(N, text)
    
    imgSimScore = []
    for ix, img in enumerate(tqdm(upSplashImages)):
        imgSimScore.append((img, getImageTextSimScore(img, text)))
    
    imgSimScore = sorted(imgSimScore, key = lambda x: x[1], reverse=True)
    return imgSimScore, upSplashImages
