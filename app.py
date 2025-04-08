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

# Carregar vetores e DataFrame histórico uma única vez
@st.cache_resource
def load_resources():
    vectordb = Chroma(persist_directory="Data/chroma_db_ing", embedding_function=OpenAIEmbeddings())
    df_historical = pd.read_excel("Data/Clausulas_Historicas_Ing_Paragrafos_Revisadas_vf.xlsx")  # ou .xlsx
    return vectordb, df_historical

vectordb, df_historical = load_resources()

# Interface do usuário
st.title("📄 NDA Clause Rewriter")

uploaded_file = st.file_uploader("📤 Faça o upload de um arquivo NDA (.docx)", type=["docx"])


if uploaded_file:
    # ✅ Salva o arquivo enviado pelo usuário como arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name  # <-- Caminho do arquivo para passar à função

    # ✅ Agora sim: extrai os parágrafos do .docx salvo
    paragraphs = extract_paragraphs(tmp_path)

    st.write("Parágrafos extraídos:", paragraphs)
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
            st.write(f"🔎 {len(df_resultado)} cláusulas processadas.")
            st.success("✅ Concluído!")

            st.dataframe(df_resultado)

            # ✅ Criar arquivo Excel em memória usando BytesIO
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_resultado.to_excel(writer, index=False)
            output.seek(0)  # volta o ponteiro para o início

            # ✅ Botão para baixar
            st.download_button(
                label="📥 Baixar resultado",
                data=output,
                file_name="output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
