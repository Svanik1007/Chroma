"""Create a clean source submission ZIP; no human data, models or environment."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("swcf", "tests", "scripts", "docs", "results", "output/pdf")


def main():
    out = ROOT/"dist"
    out.mkdir(exist_ok=True)
    target = out/"SW_CF_Submission.zip"
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".md",".py",".toml",".txt",".bat")]
    files.append(ROOT/".gitignore")
    files.append(ROOT/"models/README.md")
    for directory in DIRECTORIES:
        files.extend(p for p in (ROOT/directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix not in (".pyc",".download"))
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p,Path("SW_CF")/p.relative_to(ROOT))
    with zipfile.ZipFile(target,"r") as z:
        assert z.testzip() is None
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    (out/"SHA256.txt").write_text(f"{digest}  {target.name}\n",encoding="utf-8")
    print(f"{target}\n{len(set(files))} files, {target.stat().st_size:,} bytes\nSHA256: {digest}")


if __name__=="__main__":
    main()
