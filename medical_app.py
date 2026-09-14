import streamlit as st
from intent_router import IntentRouter
from models import MedicalAgentModel
import views


try:
    model_layer = MedicalAgentModel()
    intent_router = IntentRouter()
except ValueError as exc:
    st.set_page_config(page_title="MedScan Pro", page_icon="🩺")
    st.error(f"🚨 Configuration Error: {exc}")
    st.info("Create a .env file beside medical_app.py and add: GROQ_API_KEY=your_key")
    st.stop()


if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if "user_prompt_widget" not in st.session_state:
    st.session_state.user_prompt_widget = ""

if "temp_mic_text" not in st.session_state:
    st.session_state.temp_mic_text = ""

if "previous_analysis" not in st.session_state:
    st.session_state.previous_analysis = None

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None


def trigger_analysis():
    st.session_state.run_analysis = True


def close_popover():
    pass


views.setup_page(model_layer.model_name)
views.inject_custom_css()
view_mode = views.render_sidebar()

uploaded_image, uploaded_pdf = views.render_chat_bar(
    trigger_analysis,
    close_popover,
)

if st.session_state.run_analysis:
    final_prompt = st.session_state.user_prompt_widget.strip()

    if not final_prompt and (uploaded_image or uploaded_pdf):
        final_prompt = (
            "Analyze the uploaded medical file. Extract the patient information "
            "that is explicitly present, summarize the relevant clinical findings, "
            "lab results and important evidence, and explain the key points "
            "for the selected audience."
        )

    if not final_prompt:
        st.warning("Please type a question or upload a medical file before sending.")
        st.session_state.run_analysis = False

    else:
        try:
            # ---------------------------------
            # INTENT ROUTING
            # ---------------------------------
            route_result = intent_router.route(
                prompt=final_prompt,
                uploaded_image=uploaded_image,
                uploaded_pdf=uploaded_pdf,
                has_previous_context=(
                    st.session_state.previous_analysis is not None
                ),
            )

            st.session_state.last_intent = route_result.intent

            with st.spinner(
                f"🔬 Running {route_result.intent.replace('_', ' ').title()} workflow..."
            ):
                analysis_result = model_layer.analyze_clinical_data(
                    view_mode=view_mode,
                    prompt=final_prompt,
                    uploaded_image=uploaded_image,
                    uploaded_pdf=uploaded_pdf,
                    intent=route_result.intent,
                    response_type=route_result.response_type,
                )

            st.session_state.previous_analysis = analysis_result

            views.render_response(view_mode, analysis_result)

        except Exception as exc:
            st.error("⚠️ The analysis could not be completed.")
            st.exception(exc)
            st.caption(
                "If this is a Groq limit/model error, check your API key "
                "and model settings."
            )

        finally:
            st.session_state.run_analysis = False

views.render_footer()
