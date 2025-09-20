import sys
from pathlib import Path
import pytest

# añadir src al sys.path PRIMERO
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr.ocr_pipeline_anexo1 import pdf_to_structured



FIXTURES = Path("tests/fixtures")

def test_pdf_anexo1():
    sample_pdf = FIXTURES / "anexo1.pdf"
    assert sample_pdf.exists(), f"{sample_pdf} no existe"

    res = pdf_to_structured(str(sample_pdf), lang="spa+eng")

    # imprimir para inspección manual
    print("\n======= OCR ANEXO 1 RESULTADO =======")
    for page in res["pages"]:
        print(f"\n--- Página {page['page']} ---")
        print("Campos:", page["fields"])
        print("Insumos:", page["insumos"])
        print("Trazabilidad:", page["trazabilidad"])
        print("Firma presente:", page["firma_presente"])
    print("======= FIN =======")

    assert isinstance(res, dict)
    assert "pages" in res
    assert len(res["pages"]) > 0
