
import streamlit as st
import os

from src.pdf_loader import extract_text_from_pdf
from src.text_splitter import split_documents
from src.vector_store import (
    create_vector_store,
    search_documents
)
from src.rag_chain import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "document_name" not in st.session_state:
    st.session_state["document_name"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    .source-box {
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 DocuMind AI")

    st.caption(
        "AI-Powered Document Question Answering System"
    )

    st.divider()

    st.subheader("📄 Document")

    if st.session_state["document_name"]:

        st.success(
            st.session_state["document_name"]
        )

        st.write(
            f"Chunks: {len(st.session_state['chunks'])}"
        )

    else:

        st.info(
            "No document processed yet."
        )

    st.divider()

    st.subheader("⚙️ Options")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state["chat_history"] = []

        st.success("Chat cleared.")

    if st.button(
        "🔄 Reset Document",
        use_container_width=True
    ):

        st.session_state["vector_store"] = None
        st.session_state["chunks"] = []
        st.session_state["document_name"] = None
        st.session_state["chat_history"] = []

        st.success("Document reset.")

        st.rerun()

    st.divider()

    st.caption(
        "Local Embeddings + FAISS + Local LLM"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your PDF using '
    'Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

st.header("📤 Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a PDF such as a resume, policy, manual, or report."
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:

    st.info(
        f"Selected document: **{uploaded_file.name}**"
    )

    if st.button(
        "📖 Process Document",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        upload_dir = "uploads"

        os.makedirs(
            upload_dir,
            exist_ok=True
        )

        file_path = os.path.join(
            upload_dir,
            uploaded_file.name
        )

        try:

            with open(
                file_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

        except Exception as e:

            st.error(
                f"Could not save PDF: {e}"
            )

            st.stop()

        st.session_state["document_name"] = (
            uploaded_file.name
        )

        # ----------------------------------------------------
        # STEP 1 — PDF TEXT EXTRACTION
        # IMPORTANT:
        # Pass uploaded_file, NOT file_path
        # ----------------------------------------------------

        with st.spinner(
            "📄 Extracting text from PDF..."
        ):

            try:

                # Reset file position before reading
                uploaded_file.seek(0)

                pages = extract_text_from_pdf(
                    uploaded_file
                )

            except Exception as e:

                st.error(
                    f"PDF extraction failed: {e}"
                )

                st.stop()

        if not pages:

            st.error(
                "No text could be extracted from this PDF."
            )

            st.stop()

        st.success(
            f"✅ PDF text extracted successfully "
            f"from {len(pages)} pages."
        )

        # ----------------------------------------------------
        # STEP 2 — TEXT CHUNKING
        # ----------------------------------------------------

        with st.spinner(
            "✂️ Splitting document into chunks..."
        ):

            try:

                chunks = split_documents(
                    pages
                )

            except Exception as e:

                st.error(
                    f"Text splitting failed: {e}"
                )

                st.stop()

        if not chunks:

            st.error(
                "No text chunks were created."
            )

            st.stop()

        st.session_state["chunks"] = chunks

        st.success(
            f"✅ Document split successfully "
            f"into {len(chunks)} chunks."
        )

        # ----------------------------------------------------
        # STEP 3 — CREATE VECTOR DATABASE
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Creating embeddings and FAISS vector database..."
        ):

            try:

                vector_store = create_vector_store(
                    chunks
                )

            except Exception as e:

                st.error(
                    f"Vector database creation failed: {e}"
                )

                st.stop()

        st.session_state["vector_store"] = (
            vector_store
        )

        st.success(
            "✅ Embeddings created and "
            "vector database is ready!"
        )

        st.balloons()


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state["vector_store"] is not None:

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Document",
            st.session_state["document_name"]
        )

    with col2:

        st.metric(
            "🧩 Chunks",
            len(st.session_state["chunks"])
        )

    with col3:

        st.metric(
            "🔎 Retrieval",
            "Top 5"
        )


# ============================================================
# QUESTION SECTION
# ============================================================

st.divider()

st.header("💬 Ask Your Document")

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What is Nikhil's current percentage?"
    )
)


# ============================================================
# ASK AI
# ============================================================

if st.button(
    "🤖 Ask AI",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # Validate document
    # --------------------------------------------------------

    if st.session_state["vector_store"] is None:

        st.warning(
            "⚠️ Please upload and process a document first."
        )

        st.stop()

    # --------------------------------------------------------
    # STEP 1 — SEMANTIC SEARCH
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching relevant information..."
    ):

        try:

            results = search_documents(
                st.session_state["vector_store"],
                question,
                k=5
            )

        except Exception as e:

            st.error(
                f"Document search failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        st.warning(
            "No relevant information was found "
            "in the document."
        )

        st.stop()

    # --------------------------------------------------------
    # STEP 2 — SHOW RETRIEVED CONTEXT
    # --------------------------------------------------------

    st.subheader(
        "🔎 Retrieved Information"
    )

    for index, result in enumerate(results):

        page = result.metadata.get(
            "page",
            "Unknown"
        )

        with st.expander(
            f"Result {index + 1} — 📄 Page {page}"
        ):

            st.write(
                result.page_content
            )

    # --------------------------------------------------------
    # STEP 3 — GENERATE ANSWER
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Generating answer..."
    ):

        try:

            answer = generate_answer(
                question,
                results
            )

        except Exception as e:

            st.error(
                f"AI answer generation failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # STEP 4 — DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader(
        "🤖 AI Answer"
    )

    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # STEP 5 — DISPLAY SOURCES
    # --------------------------------------------------------

    st.subheader(
        "📚 Sources"
    )

    source_pages = []

    for result in results:

        page = result.metadata.get(
            "page",
            "Unknown"
        )

        if page not in source_pages:

            source_pages.append(page)

            st.markdown(
                f"""
                <div class="source-box">
                📄 <b>{st.session_state["document_name"]}</b>
                — Page {page}
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # SAVE CHAT HISTORY
    # --------------------------------------------------------

    st.session_state["chat_history"].append(
        {
            "question": question,
            "answer": answer,
            "sources": source_pages
        }
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state["chat_history"]:

    st.divider()

    st.header("🕘 Previous Questions")

    total_questions = len(
        st.session_state["chat_history"]
    )

    for index, chat in enumerate(
        reversed(
            st.session_state["chat_history"]
        )
    ):

        question_number = (
            total_questions - index
        )

        with st.expander(
            f"Q{question_number}: {chat['question']}"
        ):

            st.write("**🤖 Answer:**")

            st.write(
                chat["answer"]
            )

            st.write("**📚 Sources:**")

            for page in chat["sources"]:

                st.write(
                    f"📄 Page {page}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocuMind AI • RAG • Python • LangChain • "
    "FAISS • Hugging Face"
)

