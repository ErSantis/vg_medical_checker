"""
Test simple para ocr_extractor_anexo1.py - lee PDF anexo1 y produce salida
"""

import unittest
import json
import sys
from pathlib import Path

# Agregar src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr.ocr_extractor_anexo1 import process_pdf

class TestOCRExtractorAnexo1(unittest.TestCase):
    """Test simple para extractor OCR del anexo1.pdf"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial"""
        cls.output_dir = Path(__file__).parent.parent / "output"
        cls.output_dir.mkdir(exist_ok=True)
        
        cls.fixtures_dir = Path(__file__).parent / "fixtures"
        cls.anexo1_pdf = cls.fixtures_dir / "anexo1.pdf"
        
        # Verificar que el PDF existe
        if not cls.anexo1_pdf.exists():
            raise FileNotFoundError(f"PDF anexo1 no encontrado: {cls.anexo1_pdf}")
        
        print(f"📄 Procesando anexo1.pdf: {cls.anexo1_pdf}")
    
    def test_process_anexo1_pdf(self):
        """Test: Procesar anexo1.pdf y producir salida"""
        print("\n🧪 Procesando anexo1.pdf...")
        
        # Procesar PDF
        result = process_pdf(self.anexo1_pdf)
        
        # Guardar resultado completo
        output_file = self.output_dir / "output_anexo1.json"
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
    """Ejecuta el test simple para anexo1"""
    print("🚀 TEST SIMPLE - OCR_EXTRACTOR_ANEXO1")
    print("=" * 50)
    
    # Ejecutar test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOCRExtractorAnexo1)
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