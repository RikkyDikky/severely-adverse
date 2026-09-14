"""
One-off diagnostic: SR 15-18's footnote text is coming out scrambled with
the default pdfplumber extraction (probably because this PDF went through
an OCR/reflow tool -- its Producer metadata says ABBYY FineReader, unlike
the other four Fed-native PDFs). This compares the default extraction
against pdfplumber's layout=True mode on the same page, to see whether
that avoids the interleaving.
"""

import pdfplumber

PATH = "../data/sr_15-18.pdf"
PAGE_NUM = 0  # 0-indexed; this is metadata "page 1" where the scrambled text showed up

with pdfplumber.open(PATH) as pdf:
    page = pdf.pages[PAGE_NUM]

    print("=== DEFAULT extract_text() ===")
    print(page.extract_text()[:1500])

    print("\n\n=== extract_text(layout=True) ===")
    print(page.extract_text(layout=True)[:1500])

with pdfplumber.open(PATH) as pdf:
    page = pdf.pages[PAGE_NUM]
    print("\n\n=== extract_text(use_text_flow=True) ===")
    print(page.extract_text(use_text_flow=True)[:1200])
