def chunk_text(text, page, modality, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append({
            "page": page,
            "content": " ".join(words[i:i + chunk_size]),
            "modality": modality
        })

    return chunks
