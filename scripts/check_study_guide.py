"""Content and page-bound checks for the compiled teaching guide."""
from pathlib import Path
from collections import Counter
import json
import re
import sys
from pypdf import PdfReader
import pdfplumber
from build_study_guide import normalize

ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/"output/pdf/SW_CF_Four_Person_Code_and_Viva_Guide.pdf"


def compact(s):
    return re.sub(r"\s+", "", normalize(s))


def main():
    (ROOT/"tmp/pdfs").mkdir(parents=True, exist_ok=True)
    data=json.loads((ROOT/"docs/four_person_guide_content.json").read_text(encoding="utf-8"))
    reader=PdfReader(PDF)
    pages=[p.extract_text() for p in reader.pages]
    text="\n".join(pages)
    (ROOT/"tmp/pdfs/study_extracted.txt").write_text(text,encoding="utf-8")
    all_compact=compact(text)
    missing=[]
    snippets=[]
    for chapter in data["chapters"]:
        active=False
        block=[]
        for line in chapter["text"].splitlines():
            if line.strip().startswith("```"):
                if active:
                    code="\n".join(block)
                    snippets.append(code)
                    if compact(code) not in all_compact:
                        missing.append({"chapter":chapter["title"],"code":code})
                    block=[]
                active=not active
            elif active:
                block.append(line)
        assert not active
    # Check prose word occurrence counts, discounting Markdown links/notation.
    source="\n".join(c["text"] for c in data["chapters"])
    source=re.sub(r"^```[^\n]*$", "", source, flags=re.M)
    source=re.sub(r"\[([^\]]+)\]\([^)]+\)",r"\1",source)
    expected=Counter(re.findall(r"[A-Za-z][A-Za-z_]{3,}",normalize(source)))
    actual=Counter(re.findall(r"[A-Za-z][A-Za-z_]{3,}",text))
    deficits={w:n-actual[w] for w,n in expected.items() if n>actual[w]}
    outside=[]
    with pdfplumber.open(PDF) as document:
        for i,page in enumerate(document.pages,1):
            for char in page.chars:
                if char["text"].strip() and (char["x0"]<40 or char["x1"]>page.width-39 or char["top"]<15 or char["bottom"]>page.height-14):
                    outside.append({"page":i,"text":char["text"],"x0":char["x0"],"x1":char["x1"],"top":char["top"]})
    report={"pages":len(pages),"bookmarks":len(reader.outline),"source_code_blocks":len(snippets),
            "code_blocks_found":len(snippets)-len(missing),"missing_code_blocks":missing,
            "word_count_deficits":deficits,"out_of_bounds_characters":outside,
            "replacement_characters":text.count(chr(65533)),"pdf_bytes":PDF.stat().st_size}
    (ROOT/"tmp/pdfs/study_content_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=True))
    assert not missing and not outside and not deficits


if __name__=="__main__":
    main()
