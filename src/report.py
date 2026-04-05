import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(ticker, fundamentals, forecast, advice):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{ticker} Stock Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    data = [
        ["Metric", "Value"],
        ["Current Price", f"${fundamentals['current_price']:.2f}"],
        ["1W Forecast", f"${forecast:.2f}"],
        ["P/E Ratio", str(fundamentals["pe_ratio"])],
        ["Dividend Yield", f"{fundamentals['dividend_yield']:.1f}%"],
    ]
    story.append(Table(data))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>AI Recommendation:</b>", styles["Heading2"]))
    story.append(Paragraph(advice, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()