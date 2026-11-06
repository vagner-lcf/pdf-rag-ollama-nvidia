import hashlib, json
from pathlib import Path

def get_pdf_hashes(pdf_dir: str) -> dict:
    """Calculates SHA256 hashes for all PDF files in the specified directory.

    Args:
        pdf_dir (str): Path to the directory containing PDF files.

    Returns:
        dict: A dictionary mapping PDF filenames to their SHA256 hashes.
    """
    return {
        pdf.name: hashlib.sha256(pdf.read_bytes()).hexdigest()
        for pdf in sorted(Path(pdf_dir).glob("*.pdf"))
    }

def needs_reindex(data_dir:str, persist_directory:str) -> bool:
    """Checks if the hash manifest needs to be updated.

    Args:
        data_dir (str): Path to the directory containing PDF files.
        persist_directory (str): Path to the directory where the hash manifest is stored.

    Raises:
        ValueError: If no PDF files are found in the data directory.

    Returns:
        bool: True if the hash manifest needs to be updated, False otherwise.
    """
    manifest_path = Path(persist_directory) / "hash_manifest.json"
    current_hashes = get_pdf_hashes(data_dir)

    if not current_hashes:
        raise ValueError(f"Nenhum pdf encontrado para indexação no diretório {data_dir}.")

    if not manifest_path.exists():
        return True

    try:
        saved_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))

    except (json.JSONDecodeError, OSError):
        return True

    return current_hashes != saved_hashes

def save_manifest(data_dir:str, persist_directory:str) -> None:
    """Saves the hash manifest to the specified directory.

    Args:
        data_dir (str): Path to the directory containing PDF files.
        persist_directory (str): Path to the directory where the hash manifest is stored.
    """
    manifest_path = Path(persist_directory) / "hash_manifest.json"

    manifest_path.write_text(
        json.dumps(get_pdf_hashes(data_dir), indent=2),
        encoding="utf-8",
    )
