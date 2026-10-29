"""Render the editable Markdown report with ReportLab. Requires reportlab."""
from pathlib import Path
import re
from html import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Preformatted, Table, TableStyle, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#182d3c")
TEAL = colors.HexColor("#087e8b")


def inline(value):
    value = escape(value)
    value = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`(.*?)`", r'<font name="Courier">\1</font>',value)
    return value


def main():
    out = ROOT/"output/pdf/SW_CF_Project_Report.pdf"
    out.parent.mkdir(parents=True,exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySW",fontName="Helvetica",fontSize=9.6,
        leading=14,spaceAfter=8,textColor=INK,splitLongWords=True))
    styles.add(ParagraphStyle(name="HeadingSW",fontName="Helvetica-Bold",fontSize=14,
        leading=19,spaceBefore=15,spaceAfter=9,textColor=TEAL,keepWithNext=True))
    styles.add(ParagraphStyle(name="CodeSW",fontName="Courier",fontSize=8,
        leading=11,backColor=colors.HexColor("#f1f5f7"),borderPadding=8,spaceBefore=6,spaceAfter=12))
    styles.add(ParagraphStyle(name="CaptionSW",fontSize=8,leading=11,
        textColor=colors.HexColor("#516571"),spaceAfter=14))
    styles.add(ParagraphStyle(name="TitleSW",fontName="Helvetica-Bold",fontSize=33,
        leading=40,textColor=INK,spaceAfter=20))
    body, heading, code, caption = [styles[n] for n in ("BodySW","HeadingSW","CodeSW","CaptionSW")]
    story = [Spacer(1,55),Paragraph("SIGNALS AND SYSTEMS",caption),
        Paragraph("Spatially-Weighted<br/>Chrominance Fusion",styles["TitleSW"]),
        Paragraph("SW-CF  /  Contactless pulse estimation from RGB video",heading),
        Spacer(1,18),Paragraph("An implemented Python research prototype with five-region quality weighting, POS projection, digital filtering, and spectral pulse-rate estimation.",body),
        Spacer(1,18)]
    box=Table([[Paragraph("<b>Evidence status</b><br/>Software tests and synthetic validation are included. Real webcam-reference and UBFC accuracy have not been measured. Pulse-interval variability is experimental; no validated stress score is reported.",body)]],colWidths=[460])
    box.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#e8f4f4")),("BOX",(0,0),(-1,-1),.6,TEAL),
        ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),("TOPPADDING",(0,0),(-1,-1),13),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [box,Spacer(1,30),Paragraph("Team, roll numbers, institution, and supervisor:<br/><b>Add your own details in docs/REPORT.md before submission.</b>",body),
        Spacer(1,20),Paragraph("Submission edition: 27 October 2026",body),PageBreak()]
    lines=(ROOT/"docs/REPORT.md").read_text(encoding="utf-8").splitlines()
    # Cover carries title and metadata. Start the body at the abstract.
    i=next(i for i,line in enumerate(lines) if line=="## Abstract")
    while i<len(lines):
        line=lines[i]
        if not line.strip():
            i+=1; continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]),heading)); i+=1; continue
        if line.startswith("```"):
            block=[]; i+=1
            while i<len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i+=1
            story.append(Preformatted("\n".join(block),code)); i+=1; continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].startswith("|"):
                row=[x.strip() for x in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+",cell) for cell in row):
                    rows.append([Paragraph(inline(cell),body) for cell in row])
                i+=1
            table=Table(rows,colWidths=[460/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8f4f4")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.7,TEAL),
                ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#d7e1e6")),
                ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
                ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
            story.append(KeepTogether([table,Spacer(1,12)])); continue
        block=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("## ","```","|")):
            block.append(lines[i]);i+=1
        story.append(Paragraph(inline(" ".join(block)),body))
    story += [PageBreak(),Paragraph("Appendix A. Synthetic signal figures",heading)]
    for filename, desc in [
        ("signal_pipeline.png","Figure A1. Artificial five-region RGB input, supplied-quality fusion weights, pulse waveform, and Welch PSD. The target is 72 BPM; this is not a human recording."),
        ("filter_response.png","Figure A2. Combined forward/backward Butterworth magnitude response. Shading marks the chosen 0.75-2.5 Hz passband.")]:
        image=Image(str(ROOT/"results"/filename));image.drawHeight *= 460/image.drawWidth;image.drawWidth=460
        story += [image,Paragraph(desc,caption),Spacer(1,8)]
    story += [PageBreak(),Paragraph("Appendix B. Demonstration dashboard",heading)]
    image=Image(str(ROOT/"results/dashboard_demo.png"));image.drawHeight *= 460/image.drawWidth;image.drawWidth=460
    story += [image,Spacer(1,12),Paragraph("Figure B1. Saved synthetic dashboard after a 70-second generated signal. The target is 72 BPM. Two corrupted patches receive low supplied quality. All figures on this page are artificial; the experimental interval SD is processing/noise variation, not physiological variability.",caption),
        Paragraph("Reproduce the demonstration",heading),Preformatted("python main.py demo --headless --duration 70 --output outputs/demo\npython scripts/validate.py\npython -m pytest -q",code),
        Paragraph("Use the project's Python virtual environment. See README.md for setup and Windows commands. CSV evidence, scripts, test output, and the editable report source accompany this PDF.",body)]

    def decorate(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#d7e1e6"));canvas.line(48,43,A4[0]-48,43)
        canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#516571"))
        canvas.drawString(48,29,"SW-CF | Signals and Systems | Educational research prototype")
        canvas.drawRightString(A4[0]-48,29,str(doc.page))
        if doc.page>1:
            canvas.setFont("Helvetica",8);canvas.drawString(48,A4[1]-32,"SPATIALLY-WEIGHTED CHROMINANCE FUSION")
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=60,leftMargin=60,
        topMargin=53,bottomMargin=57,title="SW-CF Project Report",author="SW-CF Project Team")
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(out)


if __name__=="__main__":
    main()
