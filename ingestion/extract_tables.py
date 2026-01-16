import pdfplumber

def extract_tables(pdf_path):
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                cleaned_rows = []
                for row in table:
                    if row:
                        # Replace None with empty string
                        cleaned_row = [
                            cell if cell is not None else "" for cell in row
                        ]
                        cleaned_rows.append(" | ".join(cleaned_row))

                table_text = "\n".join(cleaned_rows)

                if table_text.strip():
                    chunks.append({
                        "page": i + 1,
                        "content": table_text,
                        "modality": "table"
                    })

    return chunks
