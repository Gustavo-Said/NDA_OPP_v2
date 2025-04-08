#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from docx import Document
import io

def extract_paragraphs(uploaded_file):
    try:
        # ✅ Converte o arquivo de upload para um buffer de memória
        doc = Document(io.BytesIO(uploaded_file.read()))

        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return paragraphs

    except Exception as e:
        print(f"Erro ao processar o arquivo .docx: {e}")
        return []