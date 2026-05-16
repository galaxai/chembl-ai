from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from rdkit import Chem

from src.config import DEFAULT_MODEL_PATH


@dataclass(frozen=True)
class SmilesResolution:
    input_smiles: str
    prediction_smiles: str | None
    was_fixed: bool
    llm_fixed_smiles: str | None = None
    repair_explanation: str | None = None
    repair_confidence: float | None = None
    error: str | None = None


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES if RDKit can parse it, otherwise None."""
    cleaned = smiles.strip()
    if not cleaned:
        return None

    mol = Chem.MolFromSmiles(cleaned)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def describe_smiles_issue(smiles: str) -> str:
    cleaned = smiles.strip()
    if not cleaned:
        return "SMILES is empty."

    mol = Chem.MolFromSmiles(cleaned, sanitize=False)
    if mol is None:
        return "RDKit could not parse the SMILES syntax."

    problems = Chem.DetectChemistryProblems(mol)
    if problems:
        return "; ".join(problem.Message() for problem in problems)

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return str(exc)

    return "RDKit parsed the SMILES without a reported chemistry problem."


def is_valid_smiles(smiles: str) -> bool:
    return canonicalize_smiles(smiles) is not None


def fix_smiles_with_llm(smiles: str):
    """Repair invalid SMILES with the PydanticAI LLM hook."""
    from src.llm_chat import repair_smiles_with_llm

    return repair_smiles_with_llm(smiles)


def resolve_smiles_for_prediction(smiles: str) -> SmilesResolution:
    input_smiles = smiles.strip()
    if not input_smiles:
        return SmilesResolution(
            input_smiles=input_smiles,
            prediction_smiles=None,
            was_fixed=False,
            error="Enter a SMILES string.",
        )

    canonical_smiles = canonicalize_smiles(input_smiles)
    if canonical_smiles is not None:
        return SmilesResolution(
            input_smiles=input_smiles,
            prediction_smiles=canonical_smiles,
            was_fixed=False,
        )

    try:
        fixed_result = fix_smiles_with_llm(input_smiles)
    except Exception as exc:
        return SmilesResolution(
            input_smiles=input_smiles,
            prediction_smiles=None,
            was_fixed=False,
            error=f"Invalid SMILES. The LLM repair agent failed: {exc}",
        )

    fixed_smiles = fixed_result.fixed_smiles
    if fixed_smiles is None:
        return SmilesResolution(
            input_smiles=input_smiles,
            prediction_smiles=None,
            was_fixed=False,
            repair_explanation=fixed_result.explanation,
            repair_confidence=fixed_result.confidence,
            error="Invalid SMILES. The LLM repair agent did not return a fix.",
        )

    canonical_fixed_smiles = canonicalize_smiles(fixed_smiles)
    if canonical_fixed_smiles is None:
        return SmilesResolution(
            input_smiles=input_smiles,
            prediction_smiles=None,
            was_fixed=False,
            llm_fixed_smiles=fixed_smiles,
            repair_explanation=fixed_result.explanation,
            repair_confidence=fixed_result.confidence,
            error="The LLM repair agent returned an invalid SMILES string.",
        )

    return SmilesResolution(
        input_smiles=input_smiles,
        prediction_smiles=canonical_fixed_smiles,
        was_fixed=True,
        llm_fixed_smiles=fixed_smiles,
        repair_explanation=fixed_result.explanation,
        repair_confidence=fixed_result.confidence,
    )


@lru_cache(maxsize=2)
def load_prediction_model(
    model_path: str = str(DEFAULT_MODEL_PATH),
    device: str | None = None,
) -> Any:
    from src.inference import load_model

    return load_model(model_path=Path(model_path), device=device)


def predict_with_tool(
    smiles: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str | None = None,
) -> float:
    """Hook for the SMILES -> pipeline -> model prediction tool call.

    Replace this body with your tool call when it is available. For now it uses
    the local model pipeline so the Streamlit app can run end to end.
    """
    from src.inference import predict_pic

    model = load_prediction_model(str(model_path), device=device)
    return predict_pic(smiles, model=model, device=device)
