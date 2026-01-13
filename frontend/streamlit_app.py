"""Streamlit frontend for the Agentic RAG system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline import AgenticPipeline
from backend.baseline_rag import BaselineRAG
from backend.llm import call_groq_chat
from backend.classifier import classify_question
from backend.hallucination_detection import detect_hallucination_complete
from backend.self_eval import self_evaluate_answer_confidence
from backend.logger import log_interaction
from backend import config
from backend.logger import INTERACTIONS_LOG, SUMMARY_LOG


st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline():
    """Initialize and cache the agentic pipeline."""
    return AgenticPipeline()


@st.cache_resource
def get_baseline_rag():
    """Initialize and cache the baseline RAG."""
    return BaselineRAG()


def load_summary_metrics():
    """Load summary metrics from JSON file."""
    if SUMMARY_LOG.exists():
        try:
            with SUMMARY_LOG.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}


def load_recent_interactions(limit: int = 10):
    """Load recent interactions from JSONL file."""
    interactions = []
    if INTERACTIONS_LOG.exists():
        try:
            with INTERACTIONS_LOG.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    if line.strip():
                        interactions.append(json.loads(line))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return interactions


def format_document(doc: str, metadata: dict = None) -> str:
    """Format a document for display."""
    if metadata:
        planet = metadata.get("planet", "Unknown")
        return f"**{planet}**: {doc[:200]}..." if len(doc) > 200 else f"**{planet}**: {doc}"
    return doc[:200] + "..." if len(doc) > 200 else doc


def main():
    # Header with styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="main-header">🚀 Agentic RAG-Based AI System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header"><b>Multi-Agent System with Self-Correction Loop</b><br>'
        'Planner • Retriever • Verifier • Memory Agents</p>',
        unsafe_allow_html=True,
    )

    # Sidebar for mode selection and metrics
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.selectbox(
            "Select Mode",
            ["Agentic RAG", "Baseline RAG", "Plain LLM"],
            help="Agentic RAG uses the full multi-agent pipeline with verification",
        )

        st.divider()
        st.header("📊 System Metrics")
        metrics = load_summary_metrics()
        if metrics:
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Total Questions", metrics.get("total_questions", 0))
            with col_metric2:
                st.metric(
                    "Hallucinations",
                    metrics.get("hallucinations_detected", 0),
                    delta=f"-{metrics.get('hallucinations_detected', 0)}",
                    delta_color="inverse",
                )
            st.metric(
                "Avg Confidence",
                f"{metrics.get('avg_confidence', 0):.2f}",
                help="Average confidence score across all verified answers",
            )
        else:
            st.info("No metrics available yet. Run some queries to see statistics.")

        st.divider()
        st.header("📋 Recent Interactions")
        interactions = load_recent_interactions(limit=5)
        if interactions:
            for interaction in reversed(interactions):
                with st.expander(
                    f"Q: {interaction.get('question', 'N/A')[:40]}...",
                    expanded=False,
                ):
                    st.write(f"**Answer:** {interaction.get('final_answer', 'N/A')[:100]}...")
                    if interaction.get("hallucination_detected"):
                        st.error("🚨 Hallucination detected")
                    else:
                        st.success("✅ Verified")
        else:
            st.info("No recent interactions yet.")

    # Main content area
    st.header("💬 Ask a Question")
    question = st.text_input(
        "Enter your question about planets:",
        placeholder="e.g., How many moons does Mars have?",
        key="question_input",
        label_visibility="collapsed",
    )

    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("🔄 Processing your question through the agentic pipeline..."):
                try:
                    if mode == "Agentic RAG":
                        pipeline = get_pipeline()
                        result = pipeline.run_with_evaluation(question, mode="agentic_rag")

                        # Question Classification
                        if result.classification:
                            st.markdown("---")
                            st.subheader("🔍 Question Classification")
                            col_class1, col_class2 = st.columns(2)
                            with col_class1:
                                st.metric("Question Type", result.classification.question_type.replace("_", " ").title())
                            with col_class2:
                                st.metric("Answerable from Local", "Yes" if result.classification.is_answerable_from_local else "No")
                            if result.classification.detected_planet:
                                st.info(f"🌍 Detected Planet: **{result.classification.detected_planet}**")

                        # Answer section
                        st.markdown("---")
                        st.subheader("📝 Final Answer")
                        
                        if result.used_memory:
                            st.info("✅ **Answered from memory cache** - This answer was previously verified and cached.")
                        
                        st.markdown(f"**{result.final_answer}**")

                        # Enhanced Hallucination Detection
                        if result.hallucination_result:
                            halluc_result = result.hallucination_result
                            if halluc_result["hallucination_detected"]:
                                st.error(
                                    f"""
                                    🚨 **HALLUCINATION DETECTED!**
                                    
                                    **Is Correct:** ❌ No
                                    """
                                )
                                if halluc_result.get("mismatches"):
                                    st.markdown("**Mismatches Found:**")
                                    for mismatch in halluc_result["mismatches"]:
                                        st.warning(
                                            f"- **{mismatch['field']}**: "
                                            f"Answer said `{mismatch['answer_value']}`, "
                                            f"but ground truth is `{mismatch['true_value']}`"
                                        )
                            else:
                                st.success(
                                    f"✅ **Answer is Correct** - No hallucinations detected"
                                )
                        
                        # Verification status (from verifier agent)
                        if result.verification:
                            if result.verification.hallucination_detected:
                                st.error(
                                    f"""
                                    ⚠️ **Verifier Agent Warning**
                                    
                                    **Confidence:** {result.verification.confidence:.2f}
                                    
                                    **Notes:** {result.verification.notes}
                                    """
                                )
                            else:
                                st.success(
                                    f"✅ **Verified by Agent** - Confidence: {result.verification.confidence:.2f}"
                                )
                        
                        # Refusal Behavior (for nonsense questions)
                        if result.classification and result.classification.question_type == "nonsense_or_out_of_scope":
                            if result.hallucination_result:
                                refusal_expected = result.hallucination_result.get("refusal_expected", False)
                                refusal_correct = result.hallucination_result.get("refusal_correct", False)
                                if refusal_expected:
                                    if refusal_correct:
                                        st.success("✅ **Correct Refusal** - System properly refused to answer nonsense question")
                                    else:
                                        st.error("❌ **Incorrect Behavior** - System should have refused but provided an answer")
                        
                        # Self-Evaluation
                        if result.self_eval:
                            st.markdown("---")
                            st.subheader("🤔 Self-Evaluation")
                            col_self1, col_self2 = st.columns(2)
                            with col_self1:
                                st.metric("Confidence", f"{result.self_eval['confidence']:.2f}")
                            with col_self2:
                                st.caption("Self-assessed confidence score")
                            if result.self_eval.get("notes"):
                                with st.expander("📝 Self-Evaluation Notes"):
                                    st.write(result.self_eval["notes"])

                        # Pipeline details in tabs
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "🔍 Retrieved Documents",
                            "🌐 Web Search Results",
                            "📊 Pipeline Details",
                            "🔍 Baseline Comparison",
                            "📋 Trusted Facts"
                        ])

                        with tab1:
                            st.subheader("📚 Local Documents (ChromaDB)")
                            if result.retrieval and result.retrieval.local_results:
                                local_docs = result.retrieval.local_results.documents
                                local_metas = result.retrieval.local_results.metadatas
                                
                                st.caption(f"Retrieved {len(local_docs)} documents from local knowledge base")
                                
                                for idx, (doc, meta) in enumerate(zip(local_docs, local_metas), 1):
                                    with st.expander(f"📄 Document {idx}: {meta.get('planet', 'Unknown Planet')}", expanded=False):
                                        st.markdown(f"**Planet:** {meta.get('planet', 'N/A')}")
                                        st.markdown(f"**Content:**\n{doc}")
                                        st.json(meta)
                            else:
                                st.info("No local documents retrieved.")

                        with tab2:
                            st.subheader("🌐 Web Search Results (Tavily)")
                            if result.retrieval and result.retrieval.used_web:
                                # Use web_results (dicts) for display, web_snippets (strings) for formatted view
                                web_results = result.retrieval.web_results if result.retrieval.web_results else []
                                web_snippets = result.retrieval.web_snippets if result.retrieval.web_snippets else []
                                
                                if web_results:
                                    st.caption(f"Found {len(web_results)} web search results")
                                    
                                    for idx, result_item in enumerate(web_results, 1):
                                        title = result_item.get('title', 'Untitled') if isinstance(result_item, dict) else 'Untitled'
                                        url = result_item.get('url', 'N/A') if isinstance(result_item, dict) else 'N/A'
                                        content = result_item.get('content') or result_item.get('snippet', 'N/A') if isinstance(result_item, dict) else 'N/A'
                                        
                                        with st.expander(f"🌍 Result {idx}: {title[:60]}...", expanded=False):
                                            st.markdown(f"**Title:** {title}")
                                            st.markdown(f"**URL:** {url}")
                                            st.markdown(f"**Content:**\n{content}")
                                            
                                            # Show formatted snippet if available
                                            if idx <= len(web_snippets) and web_snippets[idx-1]:
                                                st.markdown("---")
                                                st.markdown("**Formatted Snippet:**")
                                                st.code(web_snippets[idx-1], language=None)
                                elif web_snippets:
                                    # Fallback: if only snippets available, show them as text
                                    st.caption(f"Found {len(web_snippets)} web search results (formatted)")
                                    for idx, snippet in enumerate(web_snippets, 1):
                                        with st.expander(f"🌍 Result {idx}", expanded=False):
                                            st.markdown(f"**Content:**\n{snippet}")
                                else:
                                    st.info("Web search was performed but no results were returned.")
                            else:
                                st.info("No web search performed. Planner chose 'local_only' strategy.")

                        with tab3:
                            st.subheader("⚙️ Pipeline Execution Details")
                            
                            if result.planner:
                                st.markdown("### 🧠 Planner Agent")
                                col_plan1, col_plan2 = st.columns(2)
                                with col_plan1:
                                    st.metric("Strategy", result.planner.strategy)
                                with col_plan2:
                                    st.metric("Confidence", f"{result.planner.confidence:.2f}")
                            
                            if result.retrieval:
                                st.markdown("### 🔍 Retriever Agent")
                                col_ret1, col_ret2 = st.columns(2)
                                with col_ret1:
                                    st.metric(
                                        "Local Docs",
                                        len(result.retrieval.local_results.documents) if result.retrieval.local_results else 0
                                    )
                                with col_ret2:
                                    st.metric(
                                        "Web Results",
                                        len(result.retrieval.web_snippets) if result.retrieval.used_web else 0
                                    )
                            
                            if result.verification:
                                st.markdown("### ✅ Verifier Agent")
                                col_ver1, col_ver2 = st.columns(2)
                                with col_ver1:
                                    st.metric("Hallucination", "Detected" if result.verification.hallucination_detected else "None")
                                with col_ver2:
                                    st.metric("Confidence", f"{result.verification.confidence:.2f}")
                                
                                if result.verification.trusted_facts:
                                    with st.expander("📋 Trusted Facts Used"):
                                        st.json(result.verification.trusted_facts)

                        with tab4:
                            st.subheader("🔍 Baseline Answer Comparison")
                            if result.baseline:
                                st.markdown("### Initial Baseline Answer")
                                st.info(result.baseline["answer"])
                                
                                st.markdown("### Final Corrected Answer")
                                st.success(result.final_answer)
                                
                                if result.verification and result.verification.hallucination_detected:
                                    st.warning("⚠️ The baseline answer was corrected due to hallucination detection.")
                            else:
                                st.info("No baseline answer available for comparison.")
                        
                        with tab5:
                            st.subheader("📋 Trusted Facts")
                            if result.retrieval:
                                # Local facts
                                st.markdown("### 🗄️ Local Facts (CSV/ChromaDB)")
                                if result.retrieval.local_results.documents:
                                    for idx, (doc, meta) in enumerate(zip(
                                        result.retrieval.local_results.documents,
                                        result.retrieval.local_results.metadatas
                                    ), 1):
                                        with st.expander(f"Fact {idx}: {meta.get('planet', 'Unknown')}", expanded=False):
                                            st.markdown(f"**Planet:** {meta.get('planet', 'N/A')}")
                                            st.markdown(f"**Content:** {doc}")
                                            st.json(meta)
                                else:
                                    st.info("No local facts retrieved.")
                                
                                # Web facts
                                st.markdown("### 🌐 Web Facts (Tavily)")
                                if result.retrieval.web_results:
                                    for idx, web_result in enumerate(result.retrieval.web_results, 1):
                                        with st.expander(f"Web Source {idx}: {web_result.get('title', 'Untitled')[:50]}...", expanded=False):
                                            st.markdown(f"**Title:** {web_result.get('title', 'N/A')}")
                                            st.markdown(f"**URL:** {web_result.get('url', 'N/A')}")
                                            st.markdown(f"**Content:**\n{web_result.get('content', 'N/A')}")
                                else:
                                    st.info("No web facts retrieved.")
                            else:
                                st.info("No trusted facts available.")

                    elif mode == "Baseline RAG":
                        # Classify question
                        classification = classify_question(question)
                        
                        rag = get_baseline_rag()
                        result = rag.answer_question(question)
                        
                        # Self-evaluation
                        context = "\n\n".join(result['documents'][:2])
                        self_eval = self_evaluate_answer_confidence(question, result['answer'], context)
                        
                        # Hallucination detection
                        planet_name = classification.detected_planet
                        halluc_result = detect_hallucination_complete(
                            question=question,
                            question_type=classification.question_type,
                            final_answer=result['answer'],
                            planet_name=planet_name,
                        )
                        
                        # Format trusted facts
                        trusted_facts = {
                            "local": [
                                {"planet": meta.get("planet", "Unknown"), "content": doc}
                                for doc, meta in zip(result['documents'], result['metadatas'])
                            ],
                            "web": [],
                        }
                        sources = [meta.get("planet", "local") for meta in result['metadatas']]
                        
                        # Log
                        log_interaction(
                            question=question,
                            initial_answer=result['answer'],
                            final_answer=result['answer'],
                            mode="basic_rag",
                            question_type=classification.question_type,
                            is_answerable_from_local=classification.is_answerable_from_local,
                            hallucination_detected=halluc_result["hallucination_detected"],
                            is_correct=halluc_result["is_correct"],
                            refusal_expected=halluc_result.get("refusal_expected", False),
                            refusal_correct=halluc_result.get("refusal_correct", True),
                            mismatches=halluc_result.get("mismatches", []),
                            self_eval_confidence=self_eval["confidence"],
                            trusted_facts=trusted_facts,
                            sources_used=sources,
                        )

                        st.markdown("---")
                        st.subheader("🔍 Question Classification")
                        col_class1, col_class2 = st.columns(2)
                        with col_class1:
                            st.metric("Question Type", classification.question_type.replace("_", " ").title())
                        with col_class2:
                            st.metric("Answerable from Local", "Yes" if classification.is_answerable_from_local else "No")
                        
                        st.markdown("---")
                        st.subheader("📝 Answer")
                        st.markdown(f"**{result['answer']}**")
                        
                        # Hallucination status
                        if halluc_result["hallucination_detected"]:
                            st.error("🚨 **HALLUCINATION DETECTED!**")
                            if halluc_result.get("mismatches"):
                                for mismatch in halluc_result["mismatches"]:
                                    st.warning(
                                        f"- **{mismatch['field']}**: "
                                        f"Answer said `{mismatch['answer_value']}`, "
                                        f"but ground truth is `{mismatch['true_value']}`"
                                    )
                        else:
                            st.success("✅ **Answer is Correct**")
                        
                        # Self-evaluation
                        st.markdown("---")
                        st.subheader("🤔 Self-Evaluation")
                        st.metric("Confidence", f"{self_eval['confidence']:.2f}")

                        st.markdown("---")
                        st.subheader("📚 Retrieved Documents")
                        st.caption(f"Retrieved {len(result['documents'])} documents from local knowledge base")
                        
                        for idx, (doc, meta) in enumerate(zip(result['documents'], result['metadatas']), 1):
                            with st.expander(f"📄 Document {idx}: {meta.get('planet', 'Unknown Planet')}", expanded=False):
                                st.markdown(f"**Planet:** {meta.get('planet', 'N/A')}")
                                st.markdown(f"**Content:**\n{doc}")
                                st.json(meta)

                    else:  # Plain LLM
                        # Classify question
                        classification = classify_question(question)
                        
                        response = call_groq_chat(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant about planetary science.",
                                },
                                {"role": "user", "content": question},
                            ]
                        )
                        
                        # Self-evaluation
                        self_eval = self_evaluate_answer_confidence(question, response)
                        
                        # Hallucination detection
                        planet_name = classification.detected_planet
                        halluc_result = detect_hallucination_complete(
                            question=question,
                            question_type=classification.question_type,
                            final_answer=response,
                            planet_name=planet_name,
                        )
                        
                        # Log
                        log_interaction(
                            question=question,
                            initial_answer=response,
                            final_answer=response,
                            mode="plain_llm",
                            question_type=classification.question_type,
                            is_answerable_from_local=classification.is_answerable_from_local,
                            hallucination_detected=halluc_result["hallucination_detected"],
                            is_correct=halluc_result["is_correct"],
                            refusal_expected=halluc_result.get("refusal_expected", False),
                            refusal_correct=halluc_result.get("refusal_correct", True),
                            mismatches=halluc_result.get("mismatches", []),
                            self_eval_confidence=self_eval["confidence"],
                            trusted_facts={"local": [], "web": []},
                            sources_used=[],
                        )

                        st.markdown("---")
                        st.subheader("🔍 Question Classification")
                        col_class1, col_class2 = st.columns(2)
                        with col_class1:
                            st.metric("Question Type", classification.question_type.replace("_", " ").title())
                        with col_class2:
                            st.metric("Answerable from Local", "No" if classification.question_type == "nonsense_or_out_of_scope" else "Maybe")
                        
                        st.markdown("---")
                        st.subheader("📝 Answer")
                        st.markdown(f"**{response}**")
                        
                        # Hallucination status
                        if halluc_result["hallucination_detected"]:
                            st.error("🚨 **HALLUCINATION DETECTED!**")
                            if halluc_result.get("mismatches"):
                                for mismatch in halluc_result["mismatches"]:
                                    st.warning(
                                        f"- **{mismatch['field']}**: "
                                        f"Answer said `{mismatch['answer_value']}`, "
                                        f"but ground truth is `{mismatch['true_value']}`"
                                    )
                        else:
                            st.success("✅ **No obvious hallucinations detected**")
                        
                        # Self-evaluation
                        st.markdown("---")
                        st.subheader("🤔 Self-Evaluation")
                        st.metric("Confidence", f"{self_eval['confidence']:.2f}")
                        
                        st.warning(
                            "⚠️ **Warning:** This is a direct LLM response with no verification, retrieval, or hallucination detection."
                        )

                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
                    with st.expander("🔧 Error Details"):
                        st.exception(e)

    # Metrics visualization section
    st.markdown("---")
    st.header("📈 Performance Metrics")
    
    metrics = load_summary_metrics()
    
    # Check if we have the new evaluation format
    if metrics and any(mode in metrics for mode in ["plain_llm", "basic_rag", "agentic_rag"]):
        # New evaluation metrics format
        st.subheader("📊 Evaluation Metrics by Mode")
        
        # Create comparison charts
        modes_data = []
        for mode in ["plain_llm", "basic_rag", "agentic_rag"]:
            if mode in metrics:
                mode_data = metrics[mode]
                if "planet_fact" in mode_data:
                    pf = mode_data["planet_fact"]
                    modes_data.append({
                        "Mode": mode.replace("_", " ").title(),
                        "Accuracy": pf.get("accuracy", 0),
                        "Hallucination Rate": pf.get("hallucination_rate", 0),
                        "Type": "Planet Facts"
                    })
                if "nonsense_or_out_of_scope" in mode_data:
                    ns = mode_data["nonsense_or_out_of_scope"]
                    modes_data.append({
                        "Mode": mode.replace("_", " ").title(),
                        "Refusal Accuracy": ns.get("refusal_accuracy", 0),
                        "Overclaim Rate": ns.get("overclaim_rate", 0),
                        "Type": "Nonsense Questions"
                    })
        
        if modes_data:
            df_metrics = pd.DataFrame(modes_data)
            
            col_met1, col_met2 = st.columns(2)
            
            with col_met1:
                # Accuracy comparison
                if "Accuracy" in df_metrics.columns:
                    df_acc = df_metrics[df_metrics["Type"] == "Planet Facts"]
                    if not df_acc.empty:
                        fig_acc = px.bar(
                            df_acc,
                            x="Mode",
                            y="Accuracy",
                            title="Accuracy by Mode (Planet Facts)",
                            color="Mode",
                            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
                        )
                        fig_acc.update_yaxis(range=[0, 1])
                        st.plotly_chart(fig_acc, use_container_width=True)
                
                # Hallucination rate
                if "Hallucination Rate" in df_metrics.columns:
                    df_hal = df_metrics[df_metrics["Type"] == "Planet Facts"]
                    if not df_hal.empty:
                        fig_hal = px.bar(
                            df_hal,
                            x="Mode",
                            y="Hallucination Rate",
                            title="Hallucination Rate by Mode (Planet Facts)",
                            color="Mode",
                            color_discrete_sequence=["#d62728", "#ff7f0e", "#2ca02c"],
                        )
                        fig_hal.update_yaxis(range=[0, 1])
                        st.plotly_chart(fig_hal, use_container_width=True)
            
            with col_met2:
                # Refusal accuracy
                if "Refusal Accuracy" in df_metrics.columns:
                    df_ref = df_metrics[df_metrics["Type"] == "Nonsense Questions"]
                    if not df_ref.empty:
                        fig_ref = px.bar(
                            df_ref,
                            x="Mode",
                            y="Refusal Accuracy",
                            title="Refusal Accuracy by Mode (Nonsense)",
                            color="Mode",
                            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
                        )
                        fig_ref.update_yaxis(range=[0, 1])
                        st.plotly_chart(fig_ref, use_container_width=True)
                
                # Overclaim rate
                if "Overclaim Rate" in df_metrics.columns:
                    df_over = df_metrics[df_metrics["Type"] == "Nonsense Questions"]
                    if not df_over.empty:
                        fig_over = px.bar(
                            df_over,
                            x="Mode",
                            y="Overclaim Rate",
                            title="Overclaim Rate by Mode (Nonsense)",
                            color="Mode",
                            color_discrete_sequence=["#d62728", "#ff7f0e", "#2ca02c"],
                        )
                        fig_over.update_yaxis(range=[0, 1])
                        st.plotly_chart(fig_over, use_container_width=True)
        
        # Detailed metrics table
        with st.expander("📋 Detailed Metrics Table"):
            metrics_rows = []
            for mode, mode_data in metrics.items():
                if "planet_fact" in mode_data:
                    pf = mode_data["planet_fact"]
                    metrics_rows.append({
                        "Mode": mode,
                        "Question Type": "Planet Facts",
                        "Accuracy": f"{pf.get('accuracy', 0):.2%}",
                        "Hallucination Rate": f"{pf.get('hallucination_rate', 0):.2%}",
                        "Total": pf.get('total', 0),
                    })
                if "nonsense_or_out_of_scope" in mode_data:
                    ns = mode_data["nonsense_or_out_of_scope"]
                    metrics_rows.append({
                        "Mode": mode,
                        "Question Type": "Nonsense/Out-of-Scope",
                        "Refusal Accuracy": f"{ns.get('refusal_accuracy', 0):.2%}",
                        "Overclaim Rate": f"{ns.get('overclaim_rate', 0):.2%}",
                        "Total": ns.get('total', 0),
                    })
            
            if metrics_rows:
                df_table = pd.DataFrame(metrics_rows)
                st.dataframe(df_table, use_container_width=True)
    
    else:
        # Fallback to old metrics format
        col3, col4 = st.columns(2)

        with col3:
            if metrics:
                total = metrics.get("total_questions", 0)
                hallucinations = metrics.get("hallucinations_detected", 0)
                if total > 0:
                    df = pd.DataFrame(
                        {
                            "Type": ["Correct", "Hallucination"],
                            "Count": [total - hallucinations, hallucinations],
                        }
                    )
                    fig = px.pie(
                        df,
                        values="Count",
                        names="Type",
                        title="Hallucination Detection Rate",
                        color_discrete_map={"Correct": "#00cc00", "Hallucination": "#ff0000"},
                    )
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for visualization.")
            else:
                st.info("No metrics available yet. Run evaluation/evaluate_system.py to generate metrics.")

        with col4:
            # Confidence distribution
            interactions = load_recent_interactions(limit=50)
            if interactions:
                confidences = []
                for interaction in interactions:
                    if "self_eval_confidence" in interaction:
                        confidences.append(interaction["self_eval_confidence"])
                    elif "confidence" in interaction:
                        confidences.append(interaction["confidence"])

                if confidences:
                    df_conf = pd.DataFrame({"Confidence": confidences})
                    fig_conf = px.histogram(
                        df_conf,
                        x="Confidence",
                        nbins=10,
                        title="Confidence Score Distribution",
                        labels={"Confidence": "Confidence Score", "count": "Frequency"},
                        color_discrete_sequence=["#1f77b4"],
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                else:
                    st.info("No confidence data available for visualization.")
            else:
                st.info("No interactions available yet.")

    # Debug section
    with st.expander("🔧 Debug: View Raw JSON Log"):
        if INTERACTIONS_LOG.exists():
            with INTERACTIONS_LOG.open("r", encoding="utf-8") as f:
                content = f.read()
            st.code(content, language="json")
        else:
            st.info("No log file found yet.")


if __name__ == "__main__":
    main()
