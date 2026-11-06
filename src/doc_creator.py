from pypdf import PdfReader
from langchain_core.documents import Document


def read_pdfs_and_convert_to_documents(pdf_files: list) -> list:
    """Reads PDF files and converts them to Document objects.

    Args:
        pdf_files (list): A list of paths to PDF files.

    Raises:
        FileNotFoundError: If no PDF files are found.

    Returns:
        list: A list of Document objects.
    """
    documents = []

    if not pdf_files:
        raise FileNotFoundError("Nenhum arquivo PDF encontrado na pasta 'data/documents'.")

    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)

        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            document = Document(
                page_content=text,
                metadata={
                    "page": index,
                    "source": pdf_path.name,
                }
            )

            documents.append(document)

    return documents