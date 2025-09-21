"""
Test simple para ocr_extractor_anexo2.py - lee PDF anexo2 y produce salida
"""

import unittest
import json
import sys
from pathlib import Path

# Agregar src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr.ocr_extractor_anexo2 import process_pdf_anexo2 as process_pdf

class TestOCRExtractorAnexo2(unittest.TestCase):
    """Test simple para extractor OCR del anexo2.pdf"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial"""
        cls.output_dir = Path(__file__).parent.parent / "output"
        cls.output_dir.mkdir(exist_ok=True)
        
        cls.fixtures_dir = Path(__file__).parent / "fixtures"
        cls.anexo2_pdf = cls.fixtures_dir / "anexo2.pdf"
        
        # Verificar que el PDF existe
        if not cls.anexo2_pdf.exists():
            raise FileNotFoundError(f"PDF anexo2 no encontrado: {cls.anexo2_pdf}")
        
        print(f"📄 Procesando anexo2.pdf: {cls.anexo2_pdf}")
    
    def test_process_anexo2_pdf(self):
        """Test: Procesar anexo2.pdf y producir salida"""
        print("\n🧪 Procesando anexo2.pdf...")
        
        # Procesar PDF
        result = process_pdf(self.anexo2_pdf)
        
        # Guardar resultado completo
        output_file = self.output_dir / "output_anexo2.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Resultado guardado en: {output_file}")
        print(f"✅ Campos extraídos: {len(result.get('extracted_fields', {}))}")
        print(f"✅ Items de tabla: {len(result.get('table_data', []))}")
        print(f"✅ Tokens detectados: {len(result.get('raw_tokens', []))}")
        
        # Validación mínima - solo verificar que existe estructura
        self.assertIsInstance(result, dict)
        self.assertIn('extracted_fields', result)
        self.assertIn('table_data', result)
        self.assertIn('raw_tokens', result)


def run_tests():
    """Ejecuta el test simple para anexo2"""
    print("🚀 TEST SIMPLE - OCR_EXTRACTOR_ANEXO2")
    print("=" * 50)
    
    # Ejecutar test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOCRExtractorAnexo2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print("📊 RESULTADO")
    print("=" * 50)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Estado: {'EXITOSO' if result.wasSuccessful() else 'FALLO'}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)