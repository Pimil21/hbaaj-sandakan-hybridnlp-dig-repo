# Emotion Mapping and Sentiment Analysis using NLP and GIS
### The Case Study of the Sandakan-Ranau Death Marches (1945)

**MSc Thesis Repository** **Author:** Hajar Al Jabbar
**Institution:** Universiti Teknologi MARA (UiTM), Faculty of Built Environment  
**Year:** 2026

---

## 📖 Overview

This repository contains the computational framework developed for my Master of Science thesis. The research integrates **Natural Language Processing (NLP)** and **Geographic Information Systems (GIS)** to extract, classify, and visualize emotional narratives from historical texts regarding the Sandakan-Ranau Death Marches.

The pipeline transforms unstructured historical PDF documents into a 4D Space-Time Cube (STC), utilizing a **Hybrid Fusion Strategy** that combines Transformer-based Deep Learning (BERT) with domain-specific Lexicon matching.

## 📂 Repository Structure

The scripts are numbered sequentially to represent the processing pipeline:

```text
.
├── scripts/
│   ├── 01_environment_setup.py          # Library initialization & NLTK downloads
│   ├── 02_pdf_text_extraction.py        # OCR and text extraction from historical PDFs
│   ├── 03_text_preprocessing.py         # Text normalization & cleaning
│   ├── 04_ner_extraction.py             # Spatiotemporal Entity Extraction (NER)
│   ├── 05_hybrid_emotion_analysis.py    # BERT + Lexicon Fusion Engine
│   ├── 06a_statistical_dashboard.py     # Dash app for statistical correlation
│   ├── 06b_heatmap_dashboard.py         # Dash app for KDE Heatmaps
│   ├── 06c_streamlit_dashboard.py       # Streamlit app for interactive POI exploration
│   └── 07_mapbox_4d_viz.py              # 4D Space-Time Cube generator (HTML output)
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation