#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os
from helpers.extract_paragraphs import extract_paragraphs
from helpers.classify_and_rewrite import classify_and_rewrite_clauses
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Carregar vetores e DataFrame histórico uma única vez
@st.cache_resource
def load_resources():
    vectordb = Chroma(persist_directory="data/combined_chroma_dir_ing_v4", embedding_function=OpenAIEmbeddings())
    df_historical = pd.read_csv("data/Clausulas_Historicas_Ing_Paragrafos_Revisadas_vf.xlsx")  # ou .xlsx
    return vectordb, df_historical

vectordb, df_historical = load_resources()

# Interface do usuário
st.title("📄 NDA Clause Rewriter")

uploaded_file = st.file_uploader("📤 Faça o upload de um arquivo NDA (.docx)", type=["docx"])

if uploaded_file:
    paragraphs = extract_paragraphs(uploaded_file)

    if st.button("Classificar e Reescrever Cláusulas"):
        with st.spinner("🔍 Processando..."):
            df_resultado = classify_and_rewrite_clauses(
                new_paragraphs=paragraphs,
                vectordb=vectordb,
                df_historical=df_historical,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                top_original=3,
                top_standard=3,
                classification_threshold=0.2,
                rewrite_threshold=0.65,
                fuzzy_cutoff=0.75,
                similarity_cutoff = 0.3
            )
            st.success("✅ Concluído!")

            st.dataframe(df_resultado)

            st.download_button("📥 Baixar resultado", df_resultado.to_excel(index=False).encode('utf-8'), file_name="output.xlsx")

