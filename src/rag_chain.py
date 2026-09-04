from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


MODEL_NAME = "google/flan-t5-base"

_tokenizer = None
_model = None


# ==========================================
# LOAD MODEL ONLY ONCE
# ==========================================

def load_llm():

    global _tokenizer
    global _model

    if _tokenizer is None or _model is None:

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        _model.eval()

    return _tokenizer, _model


# ==========================================
# BUILD CONTEXT
# ==========================================

def build_context(results):

    context_parts = []

    for index, result in enumerate(results):

        page = result.metadata.get(
            "page",
            "Unknown"
        )

        text = result.page_content.strip()

        context_parts.append(
            f"[SOURCE {index + 1} | PAGE {page}]\n"
            f"{text}"
        )

    return "\n\n".join(context_parts)


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(question, results):

    if not results:

        return (
            "I could not find this information "
            "in the uploaded document."
        )

    context = build_context(results)

    prompt = f"""
You are a precise document question-answering assistant.

Your job is to answer the user's question using ONLY the
information in the provided document context.

STRICT RULES:

- Do not use outside knowledge.
- Do not guess.
- Do not invent facts.
- Do not combine unrelated information.
- If the answer is directly available, give the exact answer.
- Preserve numbers, percentages, dates, names and technical terms.
- If the context does not contain enough information, respond exactly:
"I could not find this information in the uploaded document."

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    tokenizer, model = load_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    if not answer:

        return (
            "I could not find this information "
            "in the uploaded document."
        )

    return answer