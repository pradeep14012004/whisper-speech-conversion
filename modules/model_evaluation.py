import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from jiwer import cer as jiwer_cer
    from jiwer import wer as jiwer_wer
except Exception:  # pragma: no cover
    jiwer_cer = None
    jiwer_wer = None


def normalize_zh_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", "", str(text))


def token_metrics(ref: str, hyp: str):
    ref_tokens = normalize_zh_text(ref)
    hyp_tokens = normalize_zh_text(hyp)
    if not ref_tokens and not hyp_tokens:
        return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    common = sum(1 for ch in set(ref_tokens) if ch in hyp_tokens)
    precision = common / max(len(set(hyp_tokens)), 1)
    recall = common / max(len(set(ref_tokens)), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = 1.0 - min(1.0, abs(len(hyp_tokens) - len(ref_tokens)) / max(len(ref_tokens), 1))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def render_model_evaluation(result):
    if not result:
        st.info("Evaluation metrics will appear after the model run completes.")
        return

    gt_value = st.session_state.get("ground_truth", "")
    ground_truth = st.text_area("Ground-truth Chinese transcript (optional)", value=gt_value, height=90)
    st.session_state["ground_truth"] = ground_truth

    if not ground_truth.strip():
        st.info("Provide a ground-truth transcript to enable WER/CER evaluation. Other metrics remain N/A until then.")
        return

    rows = []
    for label, text in [
        ("Original AV", result.get("original_av_prediction") or ""),
        ("New 2K LoRA", result.get("new_lora_prediction") or ""),
        ("Experimental Fusion", result.get("fused_prediction") or ""),
    ]:
        if jiwer_wer is not None and jiwer_cer is not None:
            wer_value = float(jiwer_wer(ground_truth, text))
            cer_value = float(jiwer_cer(ground_truth, text))
        else:
            wer_value = np.nan
            cer_value = np.nan
        metrics = token_metrics(ground_truth, text)
        rows.append({
            "Model": label,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "WER": round(wer_value, 4) if not np.isnan(wer_value) else np.nan,
            "CER": round(cer_value, 4) if not np.isnan(cer_value) else np.nan,
            "Latency (s)": 0.0,
        })

    metric_df = pd.DataFrame(rows)
    plot_df = metric_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1", "WER", "CER"], var_name="Metric", value_name="Score")
    fig = px.bar(plot_df, x="Metric", y="Score", color="Model", barmode="group", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metric_df, use_container_width=True)
