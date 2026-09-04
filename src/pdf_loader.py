import fitz


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file page by page.
    """

    document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number + 1,
                "text": text.strip()
            })

    document.close()

    return pages