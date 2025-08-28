# Helpers/classify_and_rewrite_pt.py
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def classify_and_rewrite_clauses_pt(new_paragraphs, vectordb, openai_api_key, similarity_threshold=0.25, rewrite_threshold=0.65):
    from langchain import PromptTemplate
    classification_results = []

    for paragraph in new_paragraphs:
        docs_and_scores = vectordb.similarity_search_with_score(paragraph, k=3)
        if not docs_and_scores:
            classification_results.append({
                "new_paragraph": paragraph,
                "classification": "unclassified",
                "reference_text": None,
                "similarity": 0.0
            })
            continue

        best_doc, best_score = docs_and_scores[0]
        similarity = 1 - best_score

        if similarity >= similarity_threshold:
            classification_results.append({
                "new_paragraph": paragraph,
                "classification": best_doc.metadata.get("classification", "unclassified"),
                "reference_text": best_doc.page_content,
                "similarity": similarity
            })
        else:
            classification_results.append({
                "new_paragraph": paragraph,
                "classification": "unclassified",
                "reference_text": None,
                "similarity": similarity
            })

    df_classified = pd.DataFrame(classification_results)

    llm_finetuned = ChatOpenAI(
        model_name="ft:gpt-4o-2024-08-06:opportunity::B8aZlbiC",  # substitua se necessário
        openai_api_key=openai_api_key,
        temperature=0.0
    )

    rewrite_prompt = PromptTemplate(
        input_variables=["new_paragraph", "standard_paragraph"],
        template="""
Compare a cláusula de um NDA recebido "{new_paragraph}" com o seu treinamento e modifique caso seja significativamente diferente ou pertinente a seções específicas para corresponder de perto ao padrão. Use o "{standard_paragraph}" como um possível guia.

Revise as seguintes etapas e garanta clareza em sua avaliação:

# Etapas

1. **Compare cláusulas**: compare a cláusula recebida com a padrão.

2. **Identifique diferenças**: determine se há diferenças significativas em termos juridicamente relevantes, significado ou estrutura. Concentre-se em cláusulas relacionadas a:
- Definição de "Afiliados"
- Definição de "Pessoas" ou "Representantes"
- Tribunal elegível
- Duração

3. **Modifique cláusulas**: ajuste cláusulas que diferem notavelmente ou pertencem às categorias especificadas para se alinharem ao NDA padrão. Caso não hajam diferenças significativas, mantenha a escrita original.

4. **Produza NDA revisado**: forneça a cláusula revisada sem comentários ou raciocínio, garantindo um formato profissional para revisão legal. Seja o mais sucinto e breve possível.

A cláusula final deve ser apresentada de forma coerente, mantendo um estilo profissional sem comentários adicionais.
"""
    )

    chain_rewrite = LLMChain(llm=llm_finetuned, prompt=rewrite_prompt)

    final_results = []
    for _, row in df_classified.iterrows():
        new_para = row["new_paragraph"]
        classification = row["classification"]
        ref_text = row["reference_text"]
        sim = row["similarity"]

        if classification == "unclassified" or ref_text is None:
            final_version = new_para
        else:
            if sim < rewrite_threshold:
                response = chain_rewrite.run(
                    new_paragraph=new_para,
                    standard_paragraph=ref_text
                )
                final_version = response.strip()
            else:
                final_version = new_para

        final_results.append({
            "new_paragraph": new_para,
            "classification": classification,
            "final_version": final_version
        })

    return pd.DataFrame(final_results)