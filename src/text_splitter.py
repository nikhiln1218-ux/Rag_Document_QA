from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(pages):
    """
    Split extracted PDF pages into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:

        page_text = page["text"]

        page_chunks = splitter.split_text(page_text)

        for chunk in page_chunks:

            if chunk.strip():

                chunks.append({
                    "text": chunk.strip(),
                    "page": page["page"]
                })

    return chunks