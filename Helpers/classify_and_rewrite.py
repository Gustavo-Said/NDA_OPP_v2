#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openai
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS  # or Chroma/Pinecone
import csv
from docx import Document
from langchain_chroma import Chroma  # or FAISS, Pinecone, etc.
import pandas as pd
import logging
import faiss
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from openai import OpenAI
import json
from collections import defaultdict
import re
import difflib

def classify_and_rewrite_clauses(
    new_paragraphs,
    vectordb,
    df_historical,
    openai_api_key,
    top_original,
    top_standard,
    classification_threshold=0.2,
    rewrite_threshold=0.65,
    fuzzy_cutoff=0.75,
    similarity_cutoff = 0.3
):
    """
    1) For each clause in `new_paragraphs`, classify it by searching entire vectordb (any source).
    2) If classification found, gather up to top_original (3) from source="original" with same classification,
       then fuzzy-match these originals to the historical DataFrame to find their 'Final Clause'.
    3) Also gather top_standard (5) from source="standard" with the same classification.
    4) Pass everything to GPT to produce a final rewriting if needed.
    5) Return a DataFrame with columns ["new_clause", "final_version", "classification"].
    """

    # Setup the rewriting prompt
    rewrite_prompt = PromptTemplate(
        input_variables=["new_clause", "historical_pairs", "standard_refs", "classification_label", "similarity_score"],
        template="""
You are a legal AI. We have received a paragraph of NDA (below).
We also have 2 reference modifications from our history and a standard clause.

# Steps

1. **Compare clauses**: compare the received clause with the standard and original references.

2. **Identify Differences**: Determine whether there are significant differences in legally relevant terms, meaning, or structure. Focus on clauses related to:
- Definition of "Affiliates", "Persons" and "Representatives" must closely match ours
- Eligible Court must be Rio de Janeiro
- Governing Law must be Brazilian Law 
- Duration must be one or two years

3. **Identify Similarities**: Determine where are the similarities between the clause being revised and the original historical clause given as reference.

4. **Modify clauses**: Adjust clauses that differ notably to align with the NDA standard. Use the historical references to learn how should the modification be done.

Approximate similarity to best clause in history = {similarity_score}
Your changes may include deleting the clause as a whole or parts of it when deemed necessary.

4. **Produce revised NDA**: Provide the revised clause without commentary or reasoning, ensuring a professional format for legal review. Be as concise and brief as possible. Try to keep the wording of the original clause as close as possible.

The final clause should be presented coherently, maintaining a professional style without additional commentary. Even if several changes are made, try to keep the revised clause as close to the original as possible. Use the historical clauses only as a general reference, without using specific names or references.

New paragraph:
{new_clause}

Top historical examples of how it was previously done:
{historical_pairs}

Standard clauses
{standard_refs}

Notes:
When rewriting, keep an eye on the context of the reference text, and if it is similar to the clause being revised. Look to keep the new clause as close as possible to the original.
"""
        )

    llm_model = ChatOpenAI(
        model_name="ft:gpt-4o-mini-2024-07-18:opportunity::BDYO66Nj",
        openai_api_key=openai_api_key,
        temperature=0.0,
        top_p=0.9
    )
    chain_rewrite = LLMChain(llm=llm_model, prompt=rewrite_prompt)

    results = []

    for new_clause in new_paragraphs:
        # --- Step 1: Classification
        docs_and_scores = vectordb.similarity_search_with_score(new_clause, k=1)
        if not docs_and_scores:
            # If no match, mark unclassified
            results.append({
                "new_clause": new_clause,
                "classification": "unclassified",
                "final_version": new_clause
            })
            continue

        best_doc, best_score = docs_and_scores[0]
        sim = 1 - best_score
        if sim >= classification_threshold:
            classification_label = best_doc.metadata.get("classification", "unclassified")
        else:
            classification_label = "unclassified"

        # If unclassified => skip rewriting
        if classification_label == "unclassified":
            results.append({
                "new_clause": new_clause,
                "classification": classification_label,
                "final_version": new_clause
            })
            continue

        # --- Step 2A: Retrieve top original from the DB

        filter_ori = {
            "$and": [
                {"classification": {"$eq": classification_label}},
                {"source": {"$eq": "original"}}
            ]
        }
        docs_original = vectordb.similarity_search_with_score(
            new_clause,
            k=30,  # increase to ensure enough good results
            filter=filter_ori
        )
        filter_rev = {
            "$and": [
                {"classification": {"$eq": classification_label}},
                {"source": {"$eq": "revised"}}
            ]
        }
        docs_revised = vectordb.similarity_search_with_score(
            new_clause,
            k=30,  # increase to ensure enough good results
            filter=filter_rev
        )
        all_docs_and_scores = docs_original + docs_revised
        
        # Filter out below similarity cutoff
        filtered_docs = [(doc, score) for doc, score in all_docs_and_scores if 1 - score >= similarity_cutoff]
        
        # Sort and select top-N
        filtered_docs.sort(key=lambda x: x[1])  # ascending distance
        top_docs = filtered_docs[:top_original]

        # Build the historical pairs string
        historical_pairs_str = ""
        if top_docs:
            for rank, (doc, score_val) in enumerate(top_docs, start=1):
                sim_val = 1 - score_val
                matched_text = doc.page_content
                source_type = doc.metadata.get("source", "")
                
                # Fuzzy matching logic
                if source_type == "original":
                    # Try to find a fuzzy match in the Initial Clause column
                    all_inits = df_historical["Initial Clause"].dropna().tolist()
                    best_match = difflib.get_close_matches(matched_text, all_inits, n=1, cutoff=fuzzy_cutoff)
                    if best_match:
                        row = df_historical[df_historical["Initial Clause"] == best_match[0]]
                        revised_text = row["Final Clause"].iloc[0] if not row.empty else "[Revised not found]"
                        matched_text = best_match[0]  # optionally use matched version
                    else:
                        revised_text = "[No fuzzy match found]"
                    
                    historical_pairs_str += (
                        f"\nOriginal #{rank} (sim={sim_val:.2f}): {matched_text}"
                        f"\nRevised #{rank}: {revised_text}\n"
                    )
                
                elif source_type == "revised":
                    # Try to find a fuzzy match in the Final Clause column
                    all_revised = df_historical["Final Clause"].dropna().tolist()
                    best_match = difflib.get_close_matches(matched_text, all_revised, n=1, cutoff=fuzzy_cutoff)
                    if best_match:
                        row = df_historical[df_historical["Final Clause"] == best_match[0]]
                        original_text = row["Initial Clause"].iloc[0] if not row.empty else "[Original not found]"
                        matched_text = best_match[0]  # optionally use matched version
                    else:
                        original_text = "[No fuzzy match found]"
                
                    historical_pairs_str += (f"\nRevised #{rank} (sim={sim_val:.2f}): {matched_text}"f"\nOriginal #{rank}: {original_text}\n"
                                            )
        else:
            historical_pairs_str = "No historical original clauses found for this classification."

        # --- Step 2B: Retrieve top standard
        filter_std = {
            "$and": [
                {"classification": {"$eq": classification_label}},
                {"source": {"$eq": "standard"}}
            ]
        }
        all_standard_docs_and_scores = vectordb.similarity_search_with_score(
            new_clause,
            k=20,
            filter=filter_std
        )
        # 2) Filter out below similarity_cutoff
        filtered_standard_docs = []
        for doc, distance in all_standard_docs_and_scores:
            sim = 1 - distance
            if sim >= 0.2:
                filtered_standard_docs.append((doc, distance))

        # 3) Now sort them by best similarity (lowest distance)
        filtered_standard_docs.sort(key=lambda x: x[1])  # distance ascending

        # 4) Finally, take the top_original from the filtered set
        top_standard_docs_and_scores = filtered_standard_docs[:top_standard]
        standard_refs_str = ""
        if top_standard_docs_and_scores:
            for rank, (doc, score_val) in enumerate(top_standard_docs_and_scores, start=1):
                sim_val = 1 - score_val
                std_text = doc.page_content
                standard_refs_str += f"\nStandard #{rank} (sim={sim_val:.2f}): {std_text}\n"
        else:
            standard_refs_str = "-"

        # --- Step 3: Rewriting logic
        if sim < rewrite_threshold:
            # rewrite
            response = chain_rewrite.run(
                new_clause=new_clause,
                historical_pairs=historical_pairs_str.strip(),
                standard_refs=standard_refs_str.strip(),
                classification_label=classification_label,
                similarity_score=f"{sim:.2f}"
            )
            final_version = response.strip()
        else:
            # keep as-is if similarity is high
            final_version = new_clause

        results.append({
            "new_clause": new_clause,
            "classification": classification_label,
            "final_version": final_version,
            "histórico": historical_pairs_str,
            "padrão": standard_refs_str
        })

    # convert to DataFrame
    df_out = pd.DataFrame(results, columns=["new_clause", "final_version", "classification","histórico","padrão"])
    return df_out

