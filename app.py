import streamlit as st
import os
import time
import arxiv
import re
from core import (
    generate_slides,
    generate_slides_from_pdf,
    compile_latex,
    search_arxiv,
    edit_slides,
    generate_pdf_id,
)
import base64
import logging
import subprocess
import platform
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tempfile


def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_pdf_as_images(file_path: str):
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        st.error(f"Failed to open PDF: {e}")
        return

    page_count = doc.page_count
    st.caption(f"Pages: {page_count}")

    # Heuristic: render all if small doc, otherwise let user choose
    render_all_default = page_count <= 15
    render_all = st.checkbox("Render all pages", value=render_all_default)

    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)

    if render_all:
        for i in range(page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            st.image(pix.tobytes("png"), use_container_width=True, caption=f"Page {i+1}")
    else:
        page_num = st.slider("Page", min_value=1, max_value=page_count, value=1)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        st.image(pix.tobytes("png"), use_container_width=True, caption=f"Page {page_num}")

    doc.close()


def get_arxiv_id_from_query(query: str) -> str | None:
    """
    Resolve query to arxiv_id, similar to paper2slides.py get_arxiv_id function.
    If query is already a valid arXiv ID, return it directly.
    Otherwise, perform search and let user select from results.
    """
    # Regex to check for valid arXiv ID format
    arxiv_id_pattern = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
    if arxiv_id_pattern.match(query):
        logging.info(f"Valid arXiv ID provided: {query}")
        return query

    # If not a direct ID, we need to search and let user choose
    # This will be handled by the UI search flow
    return None


def run_generate_step(paper_id: str, api_key: str, model_name: str, pdf_path: str | None = None) -> bool:
    """
    Step 1: Generate slides from arXiv paper or local PDF
    
    Args:
        paper_id: arXiv ID or generated ID for uploaded PDF
        api_key: API key for LLM
        model_name: Model name
        pdf_path: Path to uploaded PDF file (None for arXiv papers)
    """
    logging.info("=" * 60)
    if pdf_path:
        logging.info("GENERATING SLIDES FROM UPLOADED PDF")
    else:
        logging.info("GENERATING SLIDES FROM ARXIV PAPER")
    logging.info("=" * 60)

    if pdf_path:
        success = generate_slides_from_pdf(
            pdf_path=pdf_path,
            paper_id=paper_id,
            use_linter=False,
            use_pdfcrop=False,
            api_key=api_key,
            model_name=model_name,
        )
    else:
        success = generate_slides(
            arxiv_id=paper_id,
            use_linter=False,
            use_pdfcrop=False,
            api_key=api_key,
            model_name=model_name,
        )

    if success:
        logging.info("✓ Slide generation completed successfully")
    else:
        logging.error("✗ Slide generation failed")

    return success


def run_compile_step(paper_id: str, pdflatex_path: str) -> bool:
    """
    Step 2: Compile LaTeX slides to PDF (equivalent to cmd_compile)
    """
    logging.info("=" * 60)
    logging.info("COMPILING SLIDES TO PDF")
    logging.info("=" * 60)

    success = compile_latex(
        tex_file_path="slides.tex",
        output_directory=f"source/{paper_id}/",
        pdflatex_path=pdflatex_path,
    )

    if success:
        logging.info("✓ PDF compilation completed successfully")
    else:
        logging.error("✗ PDF compilation failed")

    return success


def run_full_pipeline(
    paper_id: str,
    api_key: str,
    model_name: str,
    pdflatex_path: str,
    pdf_path: str | None = None,
) -> bool:
    """
    Full pipeline: generate + compile (equivalent to cmd_all, minus opening PDF)
    
    Args:
        paper_id: arXiv ID or generated ID for uploaded PDF
        api_key: API key for LLM
        model_name: Model name
        pdflatex_path: Path to pdflatex compiler
        pdf_path: Path to uploaded PDF file (None for arXiv papers)
    """
    logging.info("=" * 60)
    logging.info("RUNNING FULL PAPER2SLIDES PIPELINE")
    logging.info("=" * 60)

    # Step 1: Generate slides
    if not run_generate_step(paper_id, api_key, model_name, pdf_path):
        logging.error("Pipeline failed at slide generation step")
        return False

    # Step 2: Compile to PDF
    if not run_compile_step(paper_id, pdflatex_path):
        logging.error("Pipeline failed at PDF compilation step")
        return False

    # Step 3: Verify PDF exists (we don't auto-open in webui)
    pdf_output_path = f"source/{paper_id}/slides.pdf"
    if os.path.exists(pdf_output_path):
        logging.info("=" * 60)
        logging.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        return True
    else:
        logging.error("PDF not found after compilation")
        return False


def main():
    st.set_page_config(layout="wide")

    st.title("📄 Paper2Slides")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "arxiv_id" not in st.session_state:
        st.session_state.arxiv_id = None
    if "paper_id" not in st.session_state:
        st.session_state.paper_id = None
    if "uploaded_pdf_path" not in st.session_state:
        st.session_state.uploaded_pdf_path = None
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "arxiv"  # "arxiv" or "upload"
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = (
            "ready"  # ready, generating, compiling, completed, failed
        )
    if "pdflatex_path" not in st.session_state:
        st.session_state.pdflatex_path = "pdflatex"
    if "openai_api_key" not in st.session_state:

        load_dotenv(override=True)
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if "model_name" not in st.session_state:
        load_dotenv(override=True)
        st.session_state.model_name = os.getenv("DEFAULT_MODEL", "gpt-4.1-2025-04-14")

    if "run_full_pipeline" not in st.session_state:
        st.session_state.run_full_pipeline = False

    # Configure logger
    if "logger_configured" not in st.session_state:
        logger = logging.getLogger()
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        st.session_state.logger_configured = True

    # Sidebar for paper search and settings
    with st.sidebar:
        st.header("Paper Input")
        
        # Input mode selection
        input_mode = st.radio(
            "Choose input method:",
            options=["arXiv Paper", "Upload PDF"],
            index=0 if st.session_state.input_mode == "arxiv" else 1,
            key="input_mode_radio"
        )
        st.session_state.input_mode = "arxiv" if input_mode == "arXiv Paper" else "upload"
        
        if st.session_state.input_mode == "arxiv":
            # arXiv search
            query = st.text_input("Enter arXiv ID or search query:", key="query_input")
            
            if st.button("Search Papers", key="search_button"):
                st.session_state.arxiv_id = None
                st.session_state.paper_id = None
                st.session_state.uploaded_pdf_path = None
                st.session_state.pdf_path = None
                st.session_state.messages = []
                st.session_state.pipeline_status = "ready"

                # Check if query is direct arxiv_id or needs search
                direct_id = get_arxiv_id_from_query(query)
                if direct_id:
                    st.session_state.arxiv_id = direct_id
                    st.session_state.paper_id = direct_id
                else:
                    results = search_arxiv(query)
                    if results:
                        st.session_state.search_results = results
                    else:
                        st.warning("No papers found.")

            # Show search results for selection
            if "search_results" in st.session_state:
                st.subheader("Search Results")
                for i, result in enumerate(st.session_state.search_results):
                    if st.button(
                        f"**{result.title[:60]}...** by {result.authors[0].name} et al.",
                        key=f"select_{i}",
                    ):
                        st.session_state.arxiv_id = result.get_short_id()
                        st.session_state.paper_id = result.get_short_id()
                        del st.session_state.search_results
                        st.rerun()
        
        else:
            # PDF upload
            uploaded_file = st.file_uploader(
                "Upload a PDF file",
                type=["pdf"],
                key="pdf_uploader"
            )
            
            if uploaded_file is not None and ("uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Generate a unique ID for this PDF
                paper_id = generate_pdf_id(tmp_path)
                
                # Update session state
                st.session_state.uploaded_pdf_path = tmp_path
                st.session_state.paper_id = paper_id
                st.session_state.arxiv_id = None
                st.session_state.pdf_path = None
                st.session_state.messages = []
                st.session_state.pipeline_status = "ready"
                st.session_state.uploaded_file_name = uploaded_file.name
                
                st.success(f"PDF uploaded successfully! ID: {paper_id}")

        st.header("Pipeline Settings")
        st.session_state.openai_api_key = st.text_input(
            "API Key (OpenAI or DashScope)",
            type="password",
            value=st.session_state.openai_api_key,
        )
        st.caption(
            "If left empty, keys from .env are used: OPENAI_API_KEY > DASHSCOPE_API_KEY."
        )
        st.session_state.model_name = st.text_input(
            "Model Name (e.g., gpt-4.1-2025-04-14 or qwen-plus)",
            value=st.session_state.model_name,
        )
        st.caption(
            "Default model from .env (DEFAULT_MODEL). Can be overridden here."
        )
        st.session_state.pdflatex_path = st.text_input(
            "Path to pdflatex compiler", value=st.session_state.pdflatex_path
        )

        # Pipeline control buttons
        st.header("Pipeline Control")

        # Pipeline execution buttons (only show if paper_id is selected)
        if st.session_state.paper_id:
            if st.session_state.input_mode == "arxiv":
                st.success(f"Selected arXiv: {st.session_state.paper_id}")
            else:
                st.success(f"Selected PDF: {st.session_state.paper_id}")

            # Only allow running if not currently processing
            can_run = st.session_state.pipeline_status in [
                "ready",
                "completed",
                "failed",
            ]

            if st.button(
                "🚀 Run Full Pipeline",
                key="run_full",
                disabled=not can_run,
                help="Generate slides + Compile PDF (equivalent to 'python paper2slides.py all <arxiv_id>')",
            ):
                st.session_state.pipeline_status = "generating"
                st.session_state.pdf_path = None
                st.session_state.run_full_pipeline = True
                st.rerun()

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "📝 Generate Only",
                    key="run_generate",
                    disabled=not can_run,
                    help="Generate slides only (equivalent to 'python paper2slides.py generate <arxiv_id>')",
                ):
                    st.session_state.pipeline_status = "generating"
                    st.session_state.pdf_path = None
                    st.session_state.run_full_pipeline = False
                    st.rerun()

            with col2:
                slides_exist = os.path.exists(
                    f"source/{st.session_state.paper_id}/slides.tex"
                )
                if st.button(
                    "🔨 Compile Only",
                    key="run_compile",
                    disabled=not can_run or not slides_exist,
                    help="Compile existing slides to PDF (equivalent to 'python paper2slides.py compile <paper_id>')",
                ):
                    st.session_state.pipeline_status = "compiling"
                    st.session_state.run_full_pipeline = False
                    st.rerun()

    # Main area for chat and PDF viewer
    col1, col2 = st.columns(2)

    with col1:
        st.header("Interactive Editing")

        # Only allow editing if pipeline is completed and PDF exists
        if (
            st.session_state.pipeline_status == "completed"
            and st.session_state.paper_id
            and os.path.exists(f"source/{st.session_state.paper_id}/slides.tex")
        ):

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Your instructions to edit the slides..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Editing slides..."):
                        slides_tex_path = (
                            f"source/{st.session_state.paper_id}/slides.tex"
                        )
                        with open(slides_tex_path, "r") as f:
                            beamer_code = f.read()

                        new_beamer_code = edit_slides(
                            beamer_code,
                            prompt,
                            st.session_state.openai_api_key,
                            st.session_state.model_name,
                        )

                        if new_beamer_code:
                            with open(slides_tex_path, "w") as f:
                                f.write(new_beamer_code)
                            st.info("Recompiling PDF with changes...")
                            if run_compile_step(
                                st.session_state.paper_id,
                                st.session_state.pdflatex_path,
                            ):
                                st.success("PDF recompiled successfully!")
                                st.session_state.pdf_path = (
                                    f"source/{st.session_state.paper_id}/slides.pdf"
                                )
                                st.rerun()
                            else:
                                st.error("Failed to recompile PDF.")
                        else:
                            st.error("Failed to edit slides.")
        else:
            st.info(
                "Interactive editing will be available after successful pipeline completion."
            )

    with col2:
        st.header("Pipeline Status & Results")

        # Execute pipeline based on status
        if (
            st.session_state.pipeline_status == "generating"
            and st.session_state.paper_id
        ):
            with st.spinner("🔄 Running slide generation..."):
                success = run_generate_step(
                    st.session_state.paper_id,
                    st.session_state.openai_api_key,
                    st.session_state.model_name,
                    st.session_state.uploaded_pdf_path,  # None for arXiv papers
                )

                if success:
                    st.success("✅ Slide generation completed!")
                    # Check if this was part of full pipeline or generate-only
                    if st.session_state.get("run_full_pipeline", False):
                        st.session_state.pipeline_status = "compiling"
                    else:
                        st.session_state.pipeline_status = "completed"
                else:
                    st.error("❌ Slide generation failed!")
                    st.session_state.pipeline_status = "failed"
                st.rerun()

        elif (
            st.session_state.pipeline_status == "compiling"
            and st.session_state.paper_id
        ):
            with st.spinner("🔄 Compiling PDF..."):
                success = run_compile_step(
                    st.session_state.paper_id, st.session_state.pdflatex_path
                )

                if success:
                    st.success("✅ PDF compilation completed!")
                    st.session_state.pipeline_status = "completed"
                    st.session_state.pdf_path = (
                        f"source/{st.session_state.paper_id}/slides.pdf"
                    )
                else:
                    st.error("❌ PDF compilation failed!")
                    st.session_state.pipeline_status = "failed"
                st.rerun()

        # Show PDF if available
        if (
            st.session_state.pdf_path
            and os.path.exists(st.session_state.pdf_path)
            and st.session_state.pipeline_status == "completed"
        ):

            st.subheader("📄 Generated Slides")
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download PDF",
                    f,
                    file_name=f"{st.session_state.paper_id}_slides.pdf",
                    mime="application/pdf",
                )
            display_pdf_as_images(st.session_state.pdf_path)

        elif st.session_state.pipeline_status == "ready":
            st.info("🎯 Select a paper and run the pipeline to generate slides.")
        elif st.session_state.pipeline_status == "failed":
            st.error("❌ Pipeline failed. Check the logs above for details.")
        else:
            st.info("📄 Generated PDF will be displayed here when ready.")


if __name__ == "__main__":
    main()
