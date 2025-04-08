#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from docx import Document

# Assuming the existing functions are defined as provided
def extract_paragraphs(doc):
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return paragraphs
