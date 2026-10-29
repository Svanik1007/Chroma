"""Build the complete four-person teaching transcript as a navigable PDF."""
from pathlib import Path
from html import escape
import json
import re
import textwrap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    Preformatted, LongTable, TableStyle, KeepTogether, Image,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/SW_CF_Four_Person_Code_and_Viva_Guide.pdf"
WIDTH, HEIGHT = A4
LEFT, RIGHT, TOP, BOTTOM = 49, 49, 57, 48
CONTENT_WIDTH = WIDTH - LEFT - RIGHT
INK = colors.HexColor("#183243")
TEAL = colors.HexColor("#087f87")
PALE = colors.HexColor("#edf5f6")
MUTED = colors.HexColor("#58707d")


def normalize(s):
    return (s.replace("\u2014", " - ").replace("\u2013", "-")
            .replace("\u2011", "-").replace("\u2010", "-")
            .replace("\u00a0", " ").replace("\u2018", "'")
            .replace("\u2019", "'").replace("\u201c", '"')
            .replace("\u201d", '"').replace("\u2026", "..."))


def inline(s):
    s = escape(normalize(s))
    # Keep local source labels legible. Paths stay in PDF link annotations;
    # the underlying Markdown retains the exact original targets.
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: '<link href="'+m.group(2)+'" color="#087f87">'+m.group(1)+'</link>', s)
    s = re.sub(r"`([^`]+)`", r'<font name="Code" size="9">\1</font>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s


def register_fonts():
    folder = Path("C:/Windows/Fonts")
    for name, filename in [("Body", "segoeui.ttf"), ("Body-Bold", "segoeuib.ttf"),
                           ("Body-Italic", "segoeuii.ttf"), ("Code", "consola.ttf"),
                           ("Code-Bold", "consolab.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                 italic="Body-Italic", boldItalic="Body-Bold")
    pdfmetrics.registerFontFamily("Code", normal="Code", bold="Code-Bold")


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=LEFT,
            rightMargin=RIGHT, topMargin=TOP, bottomMargin=BOTTOM,
            title="SW-CF - Complete Four-Person Code Explanation and Viva Guide",
            author="SW-CF Project Team", pageCompression=1)
        frame = Frame(LEFT, BOTTOM, CONTENT_WIDTH, HEIGHT-TOP-BOTTOM,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="guide", frames=frame, onPage=self.decorate))
        self.current_chapter = ""

    def beforeDocument(self):
        self.current_chapter = ""

    def afterFlowable(self, flowable):
        if hasattr(flowable, "chapter_title"):
            title, key = flowable.chapter_title, flowable.chapter_key
            self.current_chapter = title
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, 0, False)
            self.notify("TOCEntry", (0, title, self.page, key))

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Body", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(LEFT, HEIGHT-29, "SW-CF | FOUR-PERSON CODE AND VIVA GUIDE")
        canvas.setStrokeColor(colors.HexColor("#cddde2"))
        canvas.line(LEFT, 34, WIDTH-RIGHT, 34)
        canvas.setFont("Body", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(LEFT, 21, "Signals and Systems | Complete study compilation | 28 October 2026")
        canvas.drawRightString(WIDTH-RIGHT, 21, str(doc.page))
        canvas.restoreState()


def make_styles():
    common = dict(fontName="Body", textColor=INK, allowWidows=0, allowOrphans=0)
    return {
        "body": ParagraphStyle("body", **common, fontSize=10, leading=14.5, spaceAfter=7),
        "small": ParagraphStyle("small", **common, fontSize=8.5, leading=12, spaceAfter=6),
        "chapter": ParagraphStyle("chapter", fontName="Body-Bold", textColor=TEAL,
            fontSize=23, leading=29, spaceAfter=19, keepWithNext=True),
        "heading": ParagraphStyle("heading", fontName="Body-Bold", textColor=TEAL,
            fontSize=13, leading=18, spaceBefore=13, spaceAfter=8, keepWithNext=True),
        "sub": ParagraphStyle("sub", fontName="Body-Bold", textColor=INK,
            fontSize=10.5, leading=15, spaceBefore=8, spaceAfter=6, keepWithNext=True),
        "quote": ParagraphStyle("quote", **common, fontSize=10, leading=15,
            leftIndent=13, rightIndent=9, borderColor=TEAL, borderWidth=.5,
            borderPadding=9, backColor=PALE, spaceBefore=5, spaceAfter=11),
        "bullet": ParagraphStyle("bullet", **common, fontSize=10, leading=14.5,
            leftIndent=14, firstLineIndent=-10, spaceAfter=4),
        "code": ParagraphStyle("code", fontName="Code", textColor=INK,
            fontSize=8.5, leading=11.5, leftIndent=9, rightIndent=9,
            borderPadding=8, backColor=colors.HexColor("#f1f5f7"),
            spaceBefore=5, spaceAfter=10),
        "cell": ParagraphStyle("cell", **common, fontSize=9, leading=12.8, spaceAfter=0),
    }


def code_block(lines, style):
    # Wrap only when physically necessary; original lines remain in Markdown.
    max_width = CONTENT_WIDTH - 34
    out = []
    for line in lines:
        line = normalize(line).expandtabs(4)
        if pdfmetrics.stringWidth(line, "Code", 8.5) <= max_width:
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        continuation = " " * min(indent + 4, 20)
        while pdfmetrics.stringWidth(line, "Code", 8.5) > max_width:
            limit = min(len(line), 100)
            while pdfmetrics.stringWidth(line[:limit], "Code", 8.5) > max_width:
                limit -= 1
            cut = line.rfind(" ", 0, limit)
            if cut < max(20, indent+5):
                cut = limit
            out.append(line[:cut])
            line = continuation + line[cut:].lstrip()
        out.append(line)
    block = Preformatted("\n".join(out), style)
    return KeepTogether([block]) if len(out) <= 32 else block


def render_markdown(text, styles):
    lines = text.splitlines()
    result, i, blocks, rows_total = [], 0, 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("```"):
            content = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                content.append(lines[i]); i += 1
            result.append(code_block(content, styles["code"]))
            blocks += 1
            i += 1
            continue
        if line.startswith("|"):
            data = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in row):
                    data.append(row)
                i += 1
            count = len(data[0])
            widths = ([CONTENT_WIDTH*.38, CONTENT_WIDTH*.62] if count == 2 else
                      [CONTENT_WIDTH/count]*count)
            table = LongTable([[Paragraph(inline(c), styles["cell"]) for c in r] for r in data],
                colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dceef0")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f8fa")]),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LINEBELOW", (0,0), (-1,0), .7, TEAL),
                ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#dce5e8")),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING", (0,0), (-1,-1), 7),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ]))
            # Long viva tables are grouped so a full-page table does not force
            # a nearly empty page after the preceding presentation script.
            if len(data) > 11 and count == 2:
                for start in range(1, len(data), 6):
                    group = [data[0], *data[start:start+6]]
                    piece = LongTable([[Paragraph(inline(c), styles["cell"]) for c in r] for r in group],
                        colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
                    piece.setStyle(table._cellStyles and TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dceef0")),
                        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f8fa")]),
                        ("VALIGN", (0,0), (-1,-1), "TOP"),
                        ("LINEBELOW", (0,0), (-1,0), .7, TEAL),
                        ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#dce5e8")),
                        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
                        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                    ]))
                    result += [piece, Spacer(1,8)]
            else:
                result += [table, Spacer(1,10)]
            rows_total += len(data)
            continue
        if line.startswith("#"):
            result.append(Paragraph(inline(line.lstrip("# ")), styles["heading"]))
            i += 1
            continue
        if re.fullmatch(r"\*\*.+\*\*", line) and len(line) < 150:
            label = line[2:-2]
            style = styles["heading"] if re.match(r"\d+\.|PART |PERSON |A one-minute|Viva questions", label) else styles["sub"]
            result.append(Paragraph(inline(label), style))
            i += 1
            continue
        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip()); i += 1
            result.append(Paragraph(inline(" ".join(quote)), styles["quote"]))
            continue
        if re.match(r"^[-*] ", line) or re.match(r"^\d+\. ", line):
            if line.startswith(("- ", "* ")):
                label = "• " + line[2:]
            else:
                label = line
            result.append(Paragraph(inline(label), styles["bullet"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            next_line = lines[i].strip()
            if next_line.startswith(("```", "|", ">", "#", "- ", "* ")) or re.match(r"^\d+\. ",next_line):
                break
            if re.fullmatch(r"\*\*.+\*\*", next_line) and len(next_line) < 150:
                break
            paragraph.append(next_line)
            i += 1
        result.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
    return result, blocks, rows_total


def main():
    register_fonts()
    styles = make_styles()
    source = json.loads((ROOT/"docs/four_person_guide_content.json").read_text(encoding="utf-8"))
    story = [Spacer(1,45), Paragraph("SIGNALS AND SYSTEMS", styles["small"]),
        Paragraph("SW-CF", ParagraphStyle("cover",fontName="Body-Bold",fontSize=46,leading=55,textColor=TEAL,spaceAfter=15)),
        Paragraph("Four-Person Code Explanation<br/>and Viva Guide", ParagraphStyle("cover_sub",fontName="Body-Bold",fontSize=28,leading=36,textColor=INK,spaceAfter=23)),
        Paragraph("Spatially-Weighted Chrominance Fusion",styles["heading"]),
        Paragraph("Complete explanations from the project discussion, compiled for all four team members. Includes every code block, worked example, evaluation script, and viva question from the four teaching responses.",styles["body"]),
        Spacer(1,15),Paragraph("Also included: the pretrained-model clarification, step-by-step running instructions, and cyan/orange quality-color explanation.",styles["body"]),
        Spacer(1,20),Paragraph("Study edition: 28 October 2026",styles["body"]),
        Paragraph("Evidence boundary: software checks and synthetic verification are complete. Real webcam-reference and real UBFC accuracy remain unmeasured. Pulse-interval variability is experimental, with no validated stress score.",styles["quote"]),
        Paragraph("Code fragments retain their teaching context. Some use ellipses or simplified notation; use the project source files when running or editing the application. Sample dates and output-folder names are illustrative.",styles["small"]),
        PageBreak(),Paragraph("Contents",styles["chapter"])]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc",fontName="Body",fontSize=11,leading=17,
        textColor=INK,spaceBefore=13,leftIndent=0,firstLineIndent=0,rightIndent=22)]
    story += [toc,Spacer(1,28),Paragraph("How to use this guide",styles["heading"]),
        Paragraph("Each member should understand the overall pipeline, study their assigned part, rehearse its one-minute explanation, and practise its viva questions. The original teaching responses are preserved rather than condensed. Shared files are divided by responsibility and function.",styles["body"]),
        Paragraph("The model clarification is useful for everyone: MediaPipe locates facial landmarks; the pulse estimate comes from mathematical signal processing. No heart-rate model was trained.",styles["body"])]
    manifest = []
    for number, chapter in enumerate(source["chapters"],1):
        story.append(Spacer(1,18) if chapter["title"].startswith("Appendix B") else PageBreak())
        title = normalize(chapter["title"])
        heading = Paragraph(escape(title),styles["chapter"])
        heading.chapter_title, heading.chapter_key = title, f"chapter_{number}"
        story.append(heading)
        flows, blocks, rows = render_markdown(chapter["text"],styles)
        expected = len(re.findall(r"^```",chapter["text"],re.M))//2
        assert blocks == expected, (title,blocks,expected)
        story += flows
        manifest.append({"chapter":title,"original_characters":len(chapter["text"]),
                         "code_blocks_preserved":blocks,"table_rows":rows})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc = GuideDoc(OUT)
    doc.multiBuild(story)
    from pypdf import PdfReader
    pdf = PdfReader(OUT)
    manifest_data = {"pdf":str(OUT),"pages":len(pdf.pages),"chapters":manifest,
                     "total_code_blocks":sum(c["code_blocks_preserved"] for c in manifest)}
    checks = ROOT/"tmp/pdfs"
    checks.mkdir(parents=True, exist_ok=True)
    (checks/"study_guide_manifest.json").write_text(json.dumps(manifest_data,indent=2),encoding="utf-8")
    print(json.dumps(manifest_data,indent=2))


if __name__ == "__main__":
    main()
