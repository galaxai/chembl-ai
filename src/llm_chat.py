from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from pydantic import BaseModel, Field
from pydantic_ai import Agent, FunctionToolset, RunContext

CHAT_MODEL = "openai-responses:gpt-5.4-nano"


class ChatMessage(TypedDict):
    role: str
    content: str


class PredictionContext(TypedDict):
    input_smiles: str
    prediction_smiles: str
    prediction: float
    was_fixed: bool
    llm_fixed_smiles: str | None
    repair_explanation: str | None
    repair_confidence: float | None
    model_path: str


class SmilesRepairOutput(BaseModel):
    fixed_smiles: str | None = Field(
        description="A corrected SMILES string, or null if no safe correction exists."
    )
    explanation: str = Field(
        description="Brief explanation of what was changed or why no fix is possible."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence that fixed_smiles is the intended molecule.",
    )


class SmilesValidationOutput(BaseModel):
    is_valid: bool
    canonical_smiles: str | None = None
    issue: str | None = None


class PredictionToolOutput(BaseModel):
    input_smiles: str
    prediction_smiles: str | None = None
    prediction: float | None = None
    was_fixed: bool = False
    repair_explanation: str | None = None
    error: str | None = None


SYSTEM_PROMPT = """
You are a concise medicinal chemistry assistant for a ChEMBL pIC prediction app.
Use the provided SMILES and model output as context. Be clear when something is
an interpretation of a model prediction rather than an experimentally measured
value. Do not invent assay metadata that was not provided.

The model for predictions was trained on human cells.
Use the calculate_pic_tool when the user asks you to predict pIC for another
SMILES or compare the current molecule against another molecule.
""".strip()


SMILES_REPAIR_PROMPT = """
You repair malformed SMILES strings for a molecular property prediction app.
Return only a corrected SMILES when the intended molecule is reasonably clear.
If there are multiple plausible corrections or the input is not molecule-like,
return null for fixed_smiles and explain why. Do not invent large missing
substructures.

Always use validate_smiles_tool to test a proposed fix before returning it. If
RDKit reports a kekulization error, inspect the fused aromatic ring notation:
one common issue is an impossible aromatic ring atom count around ring closures.
""".strip()


prediction_toolset = FunctionToolset()
repair_toolset = FunctionToolset()


@repair_toolset.tool_plain
def validate_smiles_tool(smiles: str) -> SmilesValidationOutput:
    """Validate a SMILES string with RDKit and return the canonical SMILES or issue."""
    from src.app_logic import canonicalize_smiles, describe_smiles_issue

    canonical_smiles = canonicalize_smiles(smiles)
    if canonical_smiles is not None:
        return SmilesValidationOutput(
            is_valid=True,
            canonical_smiles=canonical_smiles,
        )

    return SmilesValidationOutput(
        is_valid=False,
        issue=describe_smiles_issue(smiles),
    )


@prediction_toolset.tool
def calculate_pic_tool(
    ctx: RunContext[PredictionContext], smiles: str
) -> PredictionToolOutput:
    """Predict pIC on human cells from SMILES, repairing invalid SMILES if possible."""
    from src.app_logic import canonicalize_smiles, predict_with_tool

    input_smiles = smiles.strip()
    if not input_smiles:
        return PredictionToolOutput(
            input_smiles=input_smiles,
            error="No SMILES string was provided.",
        )

    was_fixed = False
    repair_explanation = None
    canonical_smiles = canonicalize_smiles(input_smiles)
    if canonical_smiles is None:
        repair_result = repair_smiles_for_tool(input_smiles)
        if repair_result.error is not None or repair_result.prediction_smiles is None:
            return repair_result

        canonical_smiles = repair_result.prediction_smiles
        repair_explanation = repair_result.repair_explanation
        was_fixed = True

    try:
        prediction = predict_with_tool(
            canonical_smiles,
            model_path=ctx.deps["model_path"],
        )
    except Exception as exc:
        if not was_fixed:
            repair_result = repair_smiles_for_tool(input_smiles)
            if (
                repair_result.error is None
                and repair_result.prediction_smiles is not None
            ):
                try:
                    prediction = predict_with_tool(
                        repair_result.prediction_smiles,
                        model_path=ctx.deps["model_path"],
                    )
                except Exception as retry_exc:
                    repair_result.error = (
                        f"Prediction failed after SMILES repair: {retry_exc}"
                    )
                    return repair_result

                repair_result.prediction = prediction
                return repair_result

        return PredictionToolOutput(
            input_smiles=input_smiles,
            prediction_smiles=canonical_smiles,
            was_fixed=was_fixed,
            repair_explanation=repair_explanation,
            error=f"Prediction failed after SMILES validation: {exc}",
        )

    return PredictionToolOutput(
        input_smiles=input_smiles,
        prediction_smiles=canonical_smiles,
        prediction=prediction,
        was_fixed=was_fixed,
        repair_explanation=repair_explanation,
    )


def repair_smiles_for_tool(smiles: str) -> PredictionToolOutput:
    from src.app_logic import canonicalize_smiles

    try:
        repair_result = repair_smiles_with_llm(smiles)
    except Exception as exc:
        return PredictionToolOutput(
            input_smiles=smiles,
            error=f"Invalid SMILES and repair failed: {exc}",
        )

    if repair_result.fixed_smiles is None:
        return PredictionToolOutput(
            input_smiles=smiles,
            repair_explanation=repair_result.explanation,
            error="Invalid SMILES and the repair agent could not infer a safe fix.",
        )

    canonical_smiles = canonicalize_smiles(repair_result.fixed_smiles)
    if canonical_smiles is None:
        return PredictionToolOutput(
            input_smiles=smiles,
            repair_explanation=repair_result.explanation,
            error="The repair agent returned a SMILES string that RDKit still rejects.",
        )

    return PredictionToolOutput(
        input_smiles=smiles,
        prediction_smiles=canonical_smiles,
        was_fixed=True,
        repair_explanation=repair_result.explanation,
    )


@lru_cache(maxsize=1)
def prediction_chat_agent() -> Agent[PredictionContext, str]:
    return Agent(
        CHAT_MODEL,
        deps_type=PredictionContext,
        system_prompt=SYSTEM_PROMPT,
        toolsets=[prediction_toolset],
    )


@lru_cache(maxsize=1)
def smiles_repair_agent() -> Agent[None, SmilesRepairOutput]:
    return Agent(
        CHAT_MODEL,
        output_type=SmilesRepairOutput,
        system_prompt=SMILES_REPAIR_PROMPT,
        toolsets=[repair_toolset],
    )


def repair_smiles_with_llm(smiles: str) -> SmilesRepairOutput:
    from src.app_logic import describe_smiles_issue

    issue = describe_smiles_issue(smiles)
    prompt = f"""
Input SMILES:
{smiles}

RDKit issue:
{issue}

Fix this SMILES if there is one clear, chemically plausible correction.
""".strip()

    result = smiles_repair_agent().run_sync(prompt)
    return result.output


def chat_about_prediction(
    user_prompt: str,
    prediction_context: PredictionContext,
    chat_history: list[ChatMessage],
) -> str:
    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in chat_history[-8:]
    )
    fixed_note = "yes" if prediction_context["was_fixed"] else "no"
    repair_note = prediction_context["repair_explanation"] or "No LLM repair was used."
    proposed_fix = prediction_context["llm_fixed_smiles"] or "None"
    repair_confidence = prediction_context["repair_confidence"]
    confidence_note = (
        f"{repair_confidence:.2f}" if repair_confidence is not None else "None"
    )
    prompt = f"""
Prediction context:
- Original input SMILES: {prediction_context["input_smiles"]}
- LLM proposed fixed SMILES: {proposed_fix}
- SMILES used for prediction: {prediction_context["prediction_smiles"]}
- Input was LLM-repaired before prediction: {fixed_note}
- SMILES repair explanation: {repair_note}
- SMILES repair confidence: {confidence_note}
- Predicted pIC: {prediction_context["prediction"]:.4f}
- Prediction model checkpoint: {prediction_context["model_path"]}

Recent chat history:
{transcript or "No previous chat messages."}

User question:
{user_prompt}
""".strip()

    result = prediction_chat_agent().run_sync(prompt, deps=prediction_context)
    return str(result.output)
