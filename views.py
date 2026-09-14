import streamlit as st
from streamlit_mic_recorder import speech_to_text


def setup_page(model_name):
    st.set_page_config(
        page_title="MedScan Pro",
        page_icon="🩺",
        layout="wide",
    )
    st.title("🏥 MedScan Pro")
    st.caption("Evidence-grounded multimodal clinical decision support")
    st.markdown(f"**AI Engine:** `{model_name}`  •  **Mode:** Evidence-grounded analysis")


def inject_custom_css():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1250px; padding-top: 2rem; }
        div[data-testid="stHorizontalBlock"]:has(input[placeholder*="Ask me anything"]) {
            background: #202124;
            border: 1px solid #3c4043;
            border-radius: 28px;
            padding: 4px 12px;
            align-items: center;
        }
        div[data-testid="stHorizontalBlock"]:has(input[placeholder*="Ask me anything"]) input {
            border: none !important;
            background: transparent !important;
            color: white !important;
            box-shadow: none !important;
        }
        div[data-testid="stHorizontalBlock"]:has(input[placeholder*="Ask me anything"]) label {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.header("🎯 Analysis Mode")
        view_mode = st.radio(
            "Who is reading the report?",
            ["Doctor (Technical)", "Patient (Simple Terms)"],
        )
        st.divider()
        st.markdown("### Supported uploads")
        st.write("• PDF medical/lab reports\n• JPG / PNG / WEBP medical images")
        st.caption("The model is instructed to use only evidence contained in the uploaded file(s).")
    return view_mode


def render_chat_bar(trigger_analysis_callback, close_popover_callback):
    st.write("---")
    col_attach, col_prompt, col_mic, col_send = st.columns([0.55, 8.1, 0.55, 0.55])

    with col_attach:
        with st.popover("➕"):
            uploaded_image = st.file_uploader(
                "Medical image",
                type=["jpg", "jpeg", "png", "webp"],
                key="medical_image_upload",
            )
            uploaded_pdf = st.file_uploader(
                "Medical / lab PDF",
                type=["pdf"],
                key="medical_pdf_upload",
            )
            st.button("Done", on_click=close_popover_callback, use_container_width=True)

    with col_prompt:
        st.text_input(
            "Message",
            label_visibility="collapsed",
            placeholder="Ask a question or upload a file for automatic analysis...",
            key="user_prompt_widget",
            on_change=trigger_analysis_callback,
        )

    with col_mic:
        mic_text = speech_to_text(
            start_prompt="🎙️",
            stop_prompt="🛑",
            language="en",
            just_once=True,
            key="mic_stt",
        )
        if mic_text and mic_text != st.session_state.user_prompt_widget:
            st.session_state.temp_mic_text = mic_text
            st.rerun()

    with col_send:
        st.button("🚀", on_click=trigger_analysis_callback, use_container_width=True)

    uploaded_image = st.session_state.get("medical_image_upload")
    uploaded_pdf = st.session_state.get("medical_pdf_upload")

    labels = []
    if uploaded_image:
        labels.append("🖼️ Image")
    if uploaded_pdf:
        labels.append("📄 PDF")
    if labels:
        st.caption("📎 Attached: " + " + ".join(labels))

    return uploaded_image, uploaded_pdf


def _show_list(title, values, icon="•"):
    if values:
        st.markdown(f"### {title}")
        for value in values:
            st.markdown(f"{icon} {value}")


def render_response(view_mode, result):
    st.divider()
    if result.get("_intent"):
        st.caption(f"🧭 Workflow: `{result['_intent'].replace('_', ' ').title()}`")
        
    if not isinstance(result, dict):
        st.error("The AI returned an unexpected response format.")
        return

    # General medical questions should be presented as an educational answer,
    # not as a patient/document clinical-analysis report.
    if result.get("response_type") == "general":
        st.subheader(f"📚 Medical Education — {view_mode}")
        general_answer = result.get("general_answer", "").strip()
        if general_answer:
            st.markdown(general_answer)
        else:
            st.info("No educational answer was returned.")
        safety_note = result.get("safety_note", "").strip()
        if safety_note:
            st.info(f"⚠️ {safety_note}")
        if result.get("_model"):
            st.caption(f"Model: `{result['_model']}`")
        if result.get("_request_id"):
            st.caption(f"Request ID: `{result['_request_id']}`")
        return

    st.subheader(f"📋 Clinical Analysis — {view_mode}")
    patient = result.get("patient_info", {})
    st.markdown("### 👤 Patient / Document Information")

    fields = [
        ("Patient name", patient.get("patient_name")),
        ("Age", patient.get("age")),
        ("Sex", patient.get("sex")),
        ("Document type", patient.get("document_type")),
        ("Report date", patient.get("report_date")),
        ("Admission date", patient.get("admission_date")),
        ("Discharge date", patient.get("discharge_date")),
    ]
    visible = [(label, value) for label, value in fields if value]
    if visible:
        cols = st.columns(min(4, len(visible)))
        for index, (label, value) in enumerate(visible):
            cols[index % len(cols)].metric(label, value)
    else:
        st.info("No explicit patient/document metadata was found in the uploaded evidence.")

    summary = result.get("summary", "")
    if summary:
        st.markdown("### 🩺 Summary")
        st.write(summary)

    _show_list("🖼️ Image Findings", result.get("image_findings", []), "- ")
    _show_list("🧪 Laboratory Findings", result.get("lab_findings", []), "- ")
    _show_list("🔗 Clinical Correlations", result.get("correlations", []), "- ")
    _show_list("🔎 Possible Considerations", result.get("possible_considerations", []), "- ")

    uncertainties = result.get("uncertainties", [])
    if uncertainties:
        st.markdown("### ⚠️ Uncertainty / Missing Evidence")
        for item in uncertainties:
            st.warning(item)

    evidence = result.get("evidence", [])
    if evidence:
        st.markdown("### 📚 Evidence from Uploaded File")
        for item in evidence:
            claim = item.get("claim", "")
            source = item.get("source", "")
            page = item.get("page", "")
            with st.expander(claim or "Evidence"):
                st.write(f"**Source:** {source}")
                st.write(f"**Page:** {page}")

    _show_list("📌 Suggested Discussion Points", result.get("suggested_next_steps", []), "- ")

    safety_note = result.get("safety_note", "")
    if safety_note:
        st.info(f"⚠️ {safety_note}")

    if result.get("_model"):
        st.caption(f"Model: `{result['_model']}`")
    if result.get("_request_id"):
        st.caption(f"Request ID: `{result['_request_id']}`")


def render_footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#7f8c8d;font-size:0.85rem;padding:15px 0;'>"
        "⚠️ For clinical decision support only. Treating clinicians remain responsible for clinical decisions."
        "</div>",
        unsafe_allow_html=True,
    )
