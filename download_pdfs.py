"""
Download ONLY PDF files from a public Google Drive folder (no API key).
Requires: .venv with gdown (pip install gdown)

Usage:
    # folder mode
    .\\.venv\\Scripts\\python.exe download_pdfs.py --folder-url "https://drive.google.com/drive/folders/XXX"
    # file mode (fallback if folder 404s - paste the 2 PDF share links)
    .\\.venv\\Scripts\\python.exe download_pdfs.py --files "https://drive.google.com/file/d/XXX/view" "https://drive.google.com/file/d/YYY/view"
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/18CtJ8kydaOVkOVIb5loGkWPNYBnXEYK_"
DEFAULT_OUT = Path(__file__).parent / "data" / "pdfs"
RAW_TMP = Path(__file__).parent / "data" / "_raw_drive"


def extract_file_id(url: str) -> str | None:
    # supports /file/d/ID/, ?id=ID, /uc?id=ID
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def download_single_files(file_urls: list[str], out_dir: Path) -> int:
    import gdown
    count = 0
    for url in file_urls:
        url = url.strip().strip('"').strip("'")
        if not url:
            continue
        fid = extract_file_id(url)
        if not fid:
            print(f"  SKIP: could not parse file ID from {url}")
            continue
        # friendly name: try to keep original, gdown will resolve with fuzzy
        print(f"  Downloading file ID {fid} ...")
        # use uc link + fuzzy to get correct filename
        dl_url = f"https://drive.google.com/uc?id={fid}"
        # download to temp then validate pdf
        tmp_out = RAW_TMP / f"{fid}.pdf"
        RAW_TMP.mkdir(parents=True, exist_ok=True)
        ok = gdown.download(url=dl_url, output=str(tmp_out), quiet=False, fuzzy=True)
        if ok and tmp_out.exists() and is_pdf(tmp_out):
            dest = out_dir / f"{fid}.pdf"
            # try to rename to real name if available later; keep ID for now
            shutil.copy2(tmp_out, dest)
            print(f"  KEPT: {dest.name} ({dest.stat().st_size/1024:.1f} KB)")
            count += 1
        else:
            print(f"  FAIL: {fid} did not download as PDF. Is sharing='Anyone with link'?")
    return count


def is_pdf(path: Path) -> bool:
    # 1. extension check
    if path.suffix.lower() == ".pdf":
        return True
    # 2. magic-byte check (Drive files like "Final Document" may lack extension)
    try:
        with open(path, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder-url", default=FOLDER_URL)
    ap.add_argument("--files", nargs="*", default=[], help="individual PDF share links (fallback)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--keep-raw", action="store_true", help="keep non-PDF files for debug")
    args = ap.parse_args()

    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Run: .\\.venv\\Scripts\\python.exe -m pip install gdown")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_TMP.mkdir(parents=True, exist_ok=True)

    # --- FILE MODE (most reliable, no folder listing needed) ---
    if args.files:
        print(f"[file mode] Downloading {len(args.files)} PDFs -> {out_dir}")
        n = download_single_files(args.files, out_dir)
        print(f"\nDone. PDFs kept: {n}")
        for f in sorted(out_dir.glob("*.pdf")):
            print(f"  - {f.name} ({f.stat().st_size/1024:.1f} KB)")
        if not args.keep_raw:
            shutil.rmtree(RAW_TMP, ignore_errors=True)
        return

    print(f"[1/3] Downloading public folder:\n  {args.folder_url}\n  -> temp: {RAW_TMP}")
    # download_folder fetches everything; we filter to PDFs afterwards
    gdown.download_folder(
        url=args.folder_url,
        output=str(RAW_TMP),
        quiet=False,
        use_cookies=False,
    )

    print(f"\n[2/3] Filtering PDFs -> {out_dir}")
    pdf_count = 0
    for p in RAW_TMP.rglob("*"):
        if not p.is_file():
            continue
        if is_pdf(p):
            # ensure .pdf extension
            dest_name = p.stem + ".pdf" if p.suffix.lower() != ".pdf" else p.name
            # avoid collisions
            dest = out_dir / dest_name
            i = 1
            while dest.exists():
                dest = out_dir / f"{p.stem}_{i}.pdf"
                i += 1
            shutil.copy2(p, dest)
            size_kb = dest.stat().st_size / 1024
            print(f"  KEPT: {p.name} -> {dest.name} ({size_kb:.1f} KB)")
            pdf_count += 1
        else:
            print(f"  SKIP (not pdf): {p.name}")

    print(f"\n[3/3] Done. PDFs kept: {pdf_count} in {out_dir}")
    for f in sorted(out_dir.glob("*.pdf")):
        print(f"  - {f.name} ({f.stat().st_size/1024:.1f} KB)")

    if not args.keep_raw:
        shutil.rmtree(RAW_TMP, ignore_errors=True)
        print("Cleaned temp raw folder.")
    else:
        print(f"Raw files kept at {RAW_TMP}")

    if pdf_count == 0:
        print("\nWARNING: 0 PDFs found. Check:\n"
              " 1. Folder sharing = 'Anyone with link - Viewer'\n"
              " 2. Files are real PDFs (not native Google Docs needing export)")
        sys.exit(2)


if __name__ == "__main__":
    main()
