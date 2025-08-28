#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import os
import io
import tempfile
from Helpers.extract_paragraphs import extract_paragraphs
from Helpers.classify_and_rewrite import classify_and_rewrite_clauses
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Embedding compartilhado
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Menu para escolher idioma
st.title("📄 NDA Clause Rewriter")
idioma = st.selectbox("Selecione o idioma do NDA:", ["Inglês", "Português"])

# Carregar o banco vetorial e dataset correto com base no idioma
@st.cache_resource
def load_resources(idioma):
    if idioma == "Inglês":
        vectordb = Chroma(persist_directory="Data/chroma_db_ing", embedding_function=embeddings)
        df = pd.read_excel("Data/Clausulas_Historicas_Ing_Paragrafos_Revisadas_vf.xlsx")
    else:
        vectordb = Chroma(persist_directory="Data/chroma_db_pt", embedding_function=embeddings)
        df = pd.read_excel("Data/Clausulas_Historicas_Pt_Paragrafos_Revisadas_vf.xlsx")
    return vectordb, df

vectordb, df_historical = load_resources(idioma)

uploaded_file = st.file_uploader("📤 Faça o upload de um arquivo NDA (.docx)", type=["docx"])

if uploaded_file:
    paragraphs = extract_paragraphs(uploaded_file)

    if st.button("Classificar e Reescrever Cláusulas") and idioma == "Inglês":
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
            st.write(f"🔎 {len(df_resultado)} cláusulas processadas.")
            st.success("✅ Concluído!")

            st.dataframe(df_resultado)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_resultado.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Baixar resultado",
                data=output,
                file_name="output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
