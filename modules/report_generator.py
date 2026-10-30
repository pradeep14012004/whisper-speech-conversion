from pathlib import Path
import streamlit as st

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except Exception:  # pragma: no cover
    canvas = None
    letter = None


def generate_experiment_pdf(run_result, output_path):
    if canvas is None:
        raise RuntimeError("ReportLab is not installed. Install it with `pip install reportlab`.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    pdf.setTitle("Whisper AV Studio Research Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, 760, "Whisper AV Studio")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 730, f"Experiment ID: {run_result.get('run_id', 'N/A')}")
    pdf.drawString(72, 710, f"Date and time: {run_result.get('timestamp', 'N/A')}")
    pdf.drawString(72, 690, f"Selected transcript: {run_result.get('selected_chinese', 'N/A')[:180]}")
    pdf.drawString(72, 670, f"English translation: {run_result.get('english_translation', 'N/A')[:180]}")
    pdf.drawString(72, 650, f"Performance: {run_result.get('processing_time', 'N/A')}")
    pdf.save()
    return output_path


def render_report_generator(run_result):
    if not run_result:
        st.info("Generate a report after running an experiment.")
        return
    report_path = Path("outputs") / f"report_{run_result.get('run_id', 'current')}.pdf"
    if st.button("Generate downloadable PDF report"):
        try:
            pdf_path = generate_experiment_pdf(run_result, report_path)
            st.success(f"PDF report created at {pdf_path}")
            st.download_button("Download PDF report", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
        except Exception as exc:
            st.error(f"PDF generation failed: {type(exc).__name__}: {exc}")
