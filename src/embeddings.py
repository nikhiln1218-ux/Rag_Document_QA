from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings():
    """
    Create local sentence-transformer embeddings.
    No OpenAI API credits are required.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings