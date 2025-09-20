import sys
from pathlib import Path
import pytest

# Agregamos src al sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr.ocr_pipeline import pdf_to_text

FIXTURES = Path("tests/fixtures")

PDF_FILES = list(FIXTURES.glob("*.pdf"))

@pytest.mark.parametrize("pdf_path", PDF_FILES)
def test_pdf_extraction_and_print(pdf_path):
    """Verifica y muestra el contenido de cada PDF en fixtures."""
    assert pdf_path.exists(), f"El archivo {pdf_path} no existe"

    # extrae texto
    full_text = pdf_to_text(str(pdf_path))

    print(f"\n\n======= OCR RESULTADO: {pdf_path.name} =======\n")
    print(full_text[:5000])
    print(f"\n======= FIN {pdf_path.name} =======\n")

    assert isinstance(full_text, str)
    assert len(full_text.strip()) > 0, f"{pdf_path.name} no devolvió texto"
