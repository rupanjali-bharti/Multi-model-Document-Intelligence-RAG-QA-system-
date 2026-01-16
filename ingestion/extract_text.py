import fitz

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            chunks.append({
                "page": i + 1,
                "content": text,
                "modality": "text"
            })
    return chunks
