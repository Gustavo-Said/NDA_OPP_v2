#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from docx import Document

# Assuming the existing functions are defined as provided
def extract_paragraphs(doc):
    print(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return paragraphs

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return []
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return []

