from __future__ import annotations

import streamlit as st

from src.app_logic import (
    DEFAULT_MODEL_PATH,
    predict_with_tool,
    resolve_smiles_for_prediction,
)
from src.llm_chat import ChatMessage, PredictionContext, chat_about_prediction


def init_session_state() -> None:
    if "prediction_context" not in st.session_state:
        st.session_state.prediction_context = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def render_prediction(context: PredictionContext) -> None:
    st.metric("Predicted pIC", f"{context['prediction']:.4f}")
    st.caption(f"Prediction SMILES: `{context['prediction_smiles']}`")
    if context["was_fixed"] and context["repair_explanation"] is not None:
        st.info(f"SMILES repair: {context['repair_explanation']}")


def render_chat(context: PredictionContext) -> None:
    st.divider()
    st.subheader("Chat About This Prediction")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about the SMILES or pIC prediction")
    if not user_prompt:
        return

    previous_messages = list(st.session_state.chat_messages)
    user_message: ChatMessage = {"role": "user", "content": user_prompt}
    st.session_state.chat_messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant_response = chat_about_prediction(
                    user_prompt,
                    context,
                    previous_messages,
                )
            st.markdown(assistant_response)
    except Exception as exc:
        st.exception(exc)
        return

    assistant_message: ChatMessage = {
        "role": "assistant",
        "content": assistant_response,
    }
    st.session_state.chat_messages.append(assistant_message)


def main() -> None:
    st.set_page_config(page_title="ChEMBL pIC Predictor")
    init_session_state()

    st.title("ChEMBL pIC Predictor")
    st.write("Enter a SMILES string to validate it and run a pIC prediction.")

    model_path = st.sidebar.text_input(
        "Model checkpoint", value=str(DEFAULT_MODEL_PATH)
    )
    if st.session_state.prediction_context is not None:
        st.session_state.prediction_context.setdefault("model_path", model_path)

    with st.form("smiles-form"):
        smiles = st.text_input(
            "SMILES",
            placeholder="CC(=O)Oc1ccccc1C(=O)O",
        )
        submitted = st.form_submit_button("Predict")

    if not submitted and st.session_state.prediction_context is not None:
        render_prediction(st.session_state.prediction_context)
        render_chat(st.session_state.prediction_context)
        return

    if not submitted:
        return

    st.session_state.prediction_context = None
    st.session_state.chat_messages = []

    resolution = resolve_smiles_for_prediction(smiles)
    if resolution.error is not None or resolution.prediction_smiles is None:
        st.error(resolution.error or "Could not resolve a valid SMILES string.")
        return

    if resolution.was_fixed:
        st.warning("Input SMILES was invalid. Using LLM-repaired SMILES.")
        st.code(resolution.prediction_smiles)
    else:
        st.success("SMILES is valid.")

    try:
        with st.spinner("Running prediction..."):
            prediction = predict_with_tool(
                resolution.prediction_smiles,
                model_path=model_path,
            )
    except Exception as exc:
        st.exception(exc)
        return

    prediction_context: PredictionContext = {
        "input_smiles": resolution.input_smiles,
        "prediction_smiles": resolution.prediction_smiles,
        "prediction": prediction,
        "was_fixed": resolution.was_fixed,
        "llm_fixed_smiles": resolution.llm_fixed_smiles,
        "repair_explanation": resolution.repair_explanation,
        "repair_confidence": resolution.repair_confidence,
        "model_path": model_path,
    }
    st.session_state.prediction_context = prediction_context
    st.session_state.chat_messages = []

    render_prediction(prediction_context)
    render_chat(prediction_context)


if __name__ == "__main__":
    main()
