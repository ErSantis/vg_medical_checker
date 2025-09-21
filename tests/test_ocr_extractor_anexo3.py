"""
Test simple para OCR extractor del anexo 3
Procesa el PDF real y genera salida JSON
"""

import unittest
import json
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr.ocr_extractor_anexo3 import process_pdf_anexo3

class TestOCRExtractorAnexo3(unittest.TestCase):
    
    def test_process_anexo3_pdf(self):
        """Test procesando el PDF real del anexo 3"""
        
        # Buscar el archivo anexo3.pdf
        test_dir = Path(__file__).parent
        pdf_path = test_dir / "fixtures" / "anexo3.pdf"
        
        if not pdf_path.exists():
            self.skipTest(f"Archivo de prueba no encontrado: {pdf_path}")
        
        # Procesar el PDF
        result = process_pdf_anexo3(pdf_path)
        
        # Verificaciones básicas
        self.assertIsInstance(result, dict)
        self.assertIn('extracted_fields', result)
        self.assertIn('tabla_insumos', result)
        self.assertIn('raw_tokens', result)
        
        # Verificar campos extraídos
        fields = result['extracted_fields']
        self.assertIn('fecha_impresion', fields)
        self.assertIn('documento', fields)
        self.assertIn('nombre', fields)
        self.assertIn('medico_tratante', fields)
        self.assertIn('lugar', fields)
        
        # Mostrar resultados
        print("\n=== ANEXO 3 EXTRACTION RESULTS ===")
        print(f"Fecha de impresión: {fields['fecha_impresion']}")
        print(f"Documento: {fields['documento']}")
        print(f"Nombre: {fields['nombre']}")
        print(f"Médico tratante: {fields['medico_tratante']}")
        print(f"Lugar: {fields['lugar']}")
        print(f"Insumos en tabla: {len(result['tabla_insumos'])}")
        
        for i, insumo in enumerate(result['tabla_insumos'], 1):
            print(f"  {i}. {insumo['descripcion']} x{insumo['cantidad']}")
        
        print(f"Total tokens: {len(result['raw_tokens'])}")
        
        # Crear directorio de salida si no existe
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Guardar resultado como JSON
        output_file = output_dir / "output_anexo3.json"
        
        # Preparar datos para JSON (remover raw_tokens para que sea más limpio)
        json_data = {
            'extracted_fields': result['extracted_fields'],
            'tabla_insumos': result['tabla_insumos'],
            'summary': {
                'total_campos': len([v for v in result['extracted_fields'].values() if v != "[CAMPO_NO_DETECTADO_POR_OCR]"]),
                'total_insumos': len(result['tabla_insumos']),
                'total_tokens': len(result['raw_tokens'])
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Resultado guardado en: {output_file}")

if __name__ == '__main__':
    unittest.main()