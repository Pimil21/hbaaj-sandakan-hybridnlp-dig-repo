# Install required packages
"""
pip install pypdf2 pdfplumber spacy transformers nltk pandas numpy textblob geopandas
python -m spacy download en_core_web_lg
pip install torch torchvision torchaudio
# ADD to Step 1 (01_environment_setup.py)
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# PDF Processing
import PyPDF2
import pdfplumber

# NLP Libraries
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Transformers for emotion detection
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

print("Environment setup complete!")
