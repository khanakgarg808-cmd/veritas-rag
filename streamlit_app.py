import os
import tempfile
import streamlit as st
from ingestion import HybridRetrieverSystem
from generation import StrictRAGGenerator
from fact_checker import FactualityChecker

# Page Configuration
st.set_page_config(
    page_title="Domain RAG & Fact-Checker",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Domain-Specific RAG with Factuality Checker")
st.write("Upload a PDF document, ask any question, and get a strictly verified, audited response.")

# ---------------------------------------------------------------------------
# Sidebar: Setup & File Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Source PDF(s)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    st.divider()
    st.header("2. Status")
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} File(s) Ready")
    else:
        st.warning("Please upload at least one PDF to proceed.")

# ---------------------------------------------------------------------------
# Main RAG & Factuality Logic
# ---------------------------------------------------------------------------
if uploaded_files:
    # Process files and build/update index
    @st.cache_resource(show_spinner="Processing and Indexing PDF Documents...")
    def build_rag_index(files):
        temp_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_paths.append(tmp.name)
        
        # Initialize Hybrid System
        system = HybridRetrieverSystem(pdf_paths=temp_paths)
        
        # Cleanup temporary files
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
                
        return system

    rag_system = build_rag_index(uploaded_files)
    generator = StrictRAGGenerator()
    checker = FactualityChecker()

    # Question Input
    user_query = st.text_input(
        "Ask a question about your uploaded document(s):",
        placeholder="e.g., What are the rules regarding exam attendance?"
    )

    if st.button("Generate & Audit Answer", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context, generating answer, and auditing factuality..."):
                # 1. Retrieve Chunks
                retrieved_chunks = rag_system.retrieve(user_query)

                # 2. Generate Response
                gen_result = generator.generate_response(user_query, retrieved_chunks)
                raw_answer = gen_result["answer"]
                contexts = gen_result["context_texts"]

                # 3. Audit Factuality
                audit_results = checker.audit_answer(raw_answer, contexts)

            # ---------------------------------------------------------------------------
            # Display Results
            # ---------------------------------------------------------------------------
            st.divider()
            
            # Score Overview Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Factuality Score", 
                    value=f"{audit_results['factuality_score']}%"
                )
            with col2:
                if audit_results['contains_hallucinations']:
                    st.error("⚠️ Potential Hallucination Detected!")
                else:
                    st.success("✅ Fully Grounded in Context")

            # Column Split: Generated Answer Audit vs Retrieved Context
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.subheader("🤖 Generated Response Audit")
                st.write(raw_answer)
                st.divider()

                st.subheader("🔬 Sentence-by-Sentence Verification Log")
                for audit in audit_results["sentence_audits"]:
                    if audit["status"] == "VERIFIED":
                        st.success(f"**[VERIFIED]** \"{audit['sentence']}\"\n\n*Reason:* {audit['reason']}")
                    else:
                        st.error(f"**[HALLUCINATION]** \"{audit['sentence']}\"\n\n*Reason:* {audit['reason']}")

            with right_col:
                st.subheader("📚 Retrieved Source Context Chunks")
                for idx, chunk in enumerate(contexts):
                    with st.expander(f"Context Chunk #{idx + 1}", expanded=(idx == 0)):
                        st.write(chunk)

else:
    st.info("👈 Upload your PDF document in the sidebar to begin.")