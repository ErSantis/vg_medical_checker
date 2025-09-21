"""
Test simple para document_matcher.py - compara anexo1 vs anexo2 reales
"""

import unittest
import sys
import json
from pathlib import Path
from dataclasses import asdict

# Agregar src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matcher.document_matcher import DocumentMatcher
from ocr.ocr_extractor_anexo1 import process_pdf as process_pdf_anexo1
from ocr.ocr_extractor_anexo2 import process_pdf_anexo2
from ocr.ocr_extractor_anexo3 import process_pdf_anexo3

class TestDocumentMatcher(unittest.TestCase):
    """Test simple para document matcher con PDFs reales"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial"""
        cls.output_dir = Path(__file__).parent.parent / "output"
        cls.output_dir.mkdir(exist_ok=True)
        
        cls.fixtures_dir = Path(__file__).parent / "fixtures"
        cls.anexo1_pdf = cls.fixtures_dir / "anexo1.pdf"
        cls.anexo2_pdf = cls.fixtures_dir / "anexo2.pdf"
        cls.anexo3_pdf = cls.fixtures_dir / "anexo3.pdf"
        
        # Verificar que los PDFs existen
        if not cls.anexo1_pdf.exists():
            raise FileNotFoundError(f"PDF anexo1 no encontrado: {cls.anexo1_pdf}")
        if not cls.anexo2_pdf.exists():
            raise FileNotFoundError(f"PDF anexo2 no encontrado: {cls.anexo2_pdf}")
        if not cls.anexo3_pdf.exists():
            raise FileNotFoundError(f"PDF anexo3 no encontrado: {cls.anexo3_pdf}")
        
        print(f"📄 PDFs encontrados: {cls.anexo1_pdf}, {cls.anexo2_pdf}, {cls.anexo3_pdf}")
    
    def test_compare_three_documents(self):
        """Test: Comparar anexo1 vs anexo2 vs anexo3 reales usando comparación de 3 documentos"""
        print("\n🧪 Comparando 3 documentos reales...")
        
        # Procesar los 3 PDFs
        print("📄 Procesando anexo1.pdf...")
        anexo1_result = process_pdf_anexo1(self.anexo1_pdf)
        
        print("📄 Procesando anexo2.pdf...")
        anexo2_result = process_pdf_anexo2(self.anexo2_pdf)
        
        print("📄 Procesando anexo3.pdf...")
        anexo3_result = process_pdf_anexo3(self.anexo3_pdf)
        
        # Verificar que se procesaron correctamente
        self.assertIn('extracted_fields', anexo1_result)
        self.assertIn('table_data', anexo1_result)
        self.assertIn('extracted_fields', anexo2_result)
        self.assertIn('table_data', anexo2_result)
        self.assertIn('extracted_fields', anexo3_result)
        self.assertIn('tabla_insumos', anexo3_result)  # anexo3 usa tabla_insumos
        
        print(f"✅ Anexo 1 - Campos: {len(anexo1_result['extracted_fields'])}, Items tabla: {len(anexo1_result['table_data'])}")
        print(f"✅ Anexo 2 - Campos: {len(anexo2_result['extracted_fields'])}, Items tabla: {len(anexo2_result['table_data'])}")
        print(f"✅ Anexo 3 - Campos: {len(anexo3_result['extracted_fields'])}, Items tabla: {len(anexo3_result['tabla_insumos'])}")
        
        # Crear matcher y comparar 3 documentos
        print("🔍 Comparando 3 documentos...")
        matcher = DocumentMatcher()
        three_way_comparison = matcher.compare_three_documents(anexo1_result, anexo2_result, anexo3_result)
        
        # Generar reporte de 3 documentos
        report = matcher.format_three_way_report(three_way_comparison)
        
        # Guardar resultado de comparación de 3 documentos
        comparison_file = self.output_dir / "three_way_comparison_result.json"
        report_file = self.output_dir / "three_way_report.json"
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(three_way_comparison), f, ensure_ascii=False, indent=2)
            
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Comparación de 3 documentos guardada en: {comparison_file}")
        print(f"✅ Reporte de 3 documentos guardado en: {report_file}")
        
        # Mostrar resumen de la comparación de 3 documentos
        print(f"\n📊 RESUMEN DE COMPARACIÓN DE 3 DOCUMENTOS:")
        print(f"Status general: {report['overall_status']}")
        
        # Mostrar información de tablas con fuzzy matching
        table_info = report['table_comparison']
        print(f"\n📋 COMPARACIÓN DE INSUMOS (CON FUZZY MATCHING):")
        print(f"Total insumos únicos: {table_info['summary']['total_unique_items']}")
        print(f"Anexo 1: {table_info['summary']['total_items_anexo1']} items")
        print(f"Anexo 2: {table_info['summary']['total_items_anexo2']} items")
        print(f"Anexo 3: {table_info['summary']['total_items_anexo3']} items")
        print(f"En los 3 anexos: {table_info['summary']['items_in_all_three']} items")
        print(f"Solo en anexo 1: {table_info['summary']['items_only_anexo1']} items")
        print(f"Solo en anexo 2: {table_info['summary']['items_only_anexo2']} items")
        print(f"Solo en anexo 3: {table_info['summary']['items_only_anexo3']} items")
        
        # Mostrar algunos ejemplos de comparaciones de insumos
        print(f"\n🔍 EJEMPLOS DE COMPARACIONES DE INSUMOS:")
        item_count = 0
        for item_key, insumo_data in table_info['item_comparisons'].items():
            if item_count < 5:  # Mostrar los primeros 5 ejemplos
                print(f"📦 {insumo_data['insumo'][:50]}...")
                print(f"    Anexo 1: {insumo_data['anexo1_value']}")
                print(f"    Anexo 2: {insumo_data['anexo2_value']}")
                print(f"    Anexo 3: {insumo_data['anexo3_value']}")
                print(f"    Análisis: {insumo_data['discrepancy_analysis']}")
                print(f"    Presente en: {', '.join(insumo_data['present_in_anexos'])}")
                
                # Mostrar scores de similitud
                if insumo_data['pair_1_2']['similarity_score'] > 0:
                    print(f"    Similitud A1-A2: {insumo_data['pair_1_2']['similarity_score']:.1%}")
                if insumo_data['pair_2_3']['similarity_score'] > 0:
                    print(f"    Similitud A2-A3: {insumo_data['pair_2_3']['similarity_score']:.1%}")
                if insumo_data['pair_1_3']['similarity_score'] > 0:
                    print(f"    Similitud A1-A3: {insumo_data['pair_1_3']['similarity_score']:.1%}")
                print()
                item_count += 1
        
        if len(table_info['item_comparisons']) > 5:
            print(f"... y {len(table_info['item_comparisons']) - 5} insumos más")
        
        # Mostrar algunos ejemplos de comparaciones de insumos
        print(f"\n� EJEMPLOS DE COMPARACIONES DE INSUMOS:")
        item_count = 0
        for insumo_name, insumo_data in table_info['item_comparisons'].items():
            if item_count < 3:  # Mostrar solo los primeros 3 ejemplos
                print(f"📦 {insumo_name}:")
                print(f"    Anexo 1: {insumo_data['anexo1_value']}")
                print(f"    Anexo 2: {insumo_data['anexo2_value']}")
                print(f"    Anexo 3: {insumo_data['anexo3_value']}")
                print(f"    Análisis: {insumo_data['discrepancy_analysis']}")
                print(f"    Presente en: {', '.join(insumo_data['present_in_anexos'])}")
                item_count += 1
        
        if len(table_info['item_comparisons']) > 3:
            print(f"... y {len(table_info['item_comparisons']) - 3} insumos más")
        
        # Mostrar campos comparados
        print(f"\n🔍 CAMPOS COMPARADOS:")
        for field, data in report['field_comparisons'].items():
            status_emoji = "✅" if data['discrepancy_analysis'] == 'Sin discrepancias' else "⚠️"
            print(f"{status_emoji} {field}:")
            print(f"    Anexo 1: {data['anexo1_value']}")
            print(f"    Anexo 2: {data['anexo2_value']}")
            print(f"    Anexo 3: {data['anexo3_value']}")
            print(f"    Análisis: {data['discrepancy_analysis']}")
            if data['recommendation']:
                print(f"    Recomendación: {data['recommendation']}")
        
        # Validaciones para comparación de 3 documentos
        field_count = len(report['field_comparisons'])
        
        print(f"\n✅ Validación - Campos comparados: {field_count}")
        print(f"✅ Validación - Problemas anexo 1: {report['summary']['anexo1_issues']}")
        print(f"✅ Validación - Problemas anexo 2: {report['summary']['anexo2_issues']}")
        print(f"✅ Validación - Problemas anexo 3: {report['summary']['anexo3_issues']}")
        print(f"✅ Validación - Discrepancias múltiples: {report['summary']['multiple_discrepancies']}")
        
        # Validación mínima - verificar estructura de 3 documentos
        self.assertIsInstance(report, dict)
        self.assertIn('field_comparisons', report)
        self.assertIn('table_comparison', report)
        self.assertIn('overall_status', report)
        self.assertIn('summary', report)
        
        # Verificar que hay campos comparados
        self.assertGreater(field_count, 0, "Debe haber al menos un campo comparado")
        
        # Verificar estructura del summary para 3 documentos
        summary = report['summary']
        self.assertIn('anexo1_issues', summary)
        self.assertIn('anexo2_issues', summary)
        self.assertIn('anexo3_issues', summary)
        self.assertIn('multiple_discrepancies', summary)
        
        # Verificar que cada campo tiene la estructura correcta para 3 documentos
        for field_name, field_data in report['field_comparisons'].items():
            self.assertIn('anexo1_value', field_data)
            self.assertIn('anexo2_value', field_data)
            self.assertIn('anexo3_value', field_data)
            self.assertIn('discrepancy_analysis', field_data)


def run_tests():
    """Ejecuta el test simple para document matcher"""
    print("🚀 TEST SIMPLE - DOCUMENT_MATCHER")
    print("=" * 50)
    
    # Ejecutar test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDocumentMatcher)
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