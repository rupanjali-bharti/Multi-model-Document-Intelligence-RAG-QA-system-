import fitz
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_images_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []

    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base["image"]))
            text = pytesseract.image_to_string(image)

            if text.strip():
                chunks.append({
                    "page": i + 1,
                    "content": text,
                    "modality": "image"
                })
    return chunks
