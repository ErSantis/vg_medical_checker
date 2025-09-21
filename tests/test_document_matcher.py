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
        
        # Mostrar información de tablas con nueva lógica (solo anexo 1 vs anexo 2)
        table_info = report['table_comparison']
        print(f"\n📋 COMPARACIÓN DE INSUMOS (SOLO ANEXO 1 vs ANEXO 2):")
        print(f"Nota: {table_info.get('note', 'Comparación limitada a anexo 1 y anexo 2')}")
        print(f"Status: {table_info['status']}")
        print(f"Similitud total: {table_info['total_similarity']:.1%}")
        print(f"Items coincidentes: {table_info['summary']['matches']}")
        print(f"Items solo en anexo 1: {table_info['summary']['anexo1_only']}")
        print(f"Items solo en anexo 2: {table_info['summary']['anexo2_only']}")
        
        # Mostrar algunos ejemplos de items coincidentes
        if table_info['matched_items']:
            print(f"\n🔍 EJEMPLOS DE ITEMS COINCIDENTES:")
            for i, match in enumerate(table_info['matched_items'][:3], 1):  # Solo los primeros 3
                print(f"  Match {i}:")
                print(f"    Anexo1: Cant={match['anexo1_item'].get('cantidad', 'N/A')} - {match['anexo1_item'].get('descripcion', 'N/A')}")
                print(f"    Anexo2: Cant={match['anexo2_item'].get('cantidad', 'N/A')} - {match['anexo2_item'].get('descripcion', 'N/A')}")
                print(f"    Similitud descripción: {match['descripcion_similarity']:.1%}")
                print(f"    Status: {match['status']}")
            
            if len(table_info['matched_items']) > 3:
                print(f"    ... y {len(table_info['matched_items']) - 3} matches más")
        
        # Mostrar items solo en anexo1
        if table_info['anexo1_only']:
            print(f"\n📦 ITEMS SOLO EN ANEXO 1:")
            for item in table_info['anexo1_only'][:3]:  # Solo los primeros 3
                print(f"    - Cant={item.get('cantidad', 'N/A')} - {item.get('descripcion', 'N/A')}")
            if len(table_info['anexo1_only']) > 3:
                print(f"    ... y {len(table_info['anexo1_only']) - 3} items más")
        
        # Mostrar items solo en anexo2
        if table_info['anexo2_only']:
            print(f"\n📦 ITEMS SOLO EN ANEXO 2:")
            for item in table_info['anexo2_only'][:3]:  # Solo los primeros 3
                print(f"    - Cant={item.get('cantidad', 'N/A')} - {item.get('descripcion', 'N/A')}")
            if len(table_info['anexo2_only']) > 3:
                print(f"    ... y {len(table_info['anexo2_only']) - 3} items más")
        
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
        
        # Validaciones para comparación de 3 documentos con nueva lógica
        field_count = len(report['field_comparisons'])
        
        print(f"\n✅ Validación - Campos comparados: {field_count}")
        print(f"✅ Validación - Problemas anexo 1: {report['summary']['anexo1_issues']}")
        print(f"✅ Validación - Problemas anexo 2: {report['summary']['anexo2_issues']}")
        print(f"✅ Validación - Problemas anexo 3: {report['summary']['anexo3_issues']}")
        print(f"✅ Validación - Discrepancias múltiples: {report['summary']['multiple_discrepancies']}")
        print(f"✅ Validación - Tabla comparada solo entre anexo 1 y 2: {table_info.get('comparison_scope', 'N/A')}")
        
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
        
        # Verificar que la tabla tiene el nuevo scope
        self.assertEqual(table_info.get('comparison_scope'), 'anexo1_vs_anexo2_only', 
                        "La tabla debe compararse solo entre anexo 1 y anexo 2")
        
        # Verificar que cada campo tiene la estructura correcta para 3 documentos
        for field_name, field_data in report['field_comparisons'].items():
            self.assertIn('anexo1_value', field_data)
            self.assertIn('anexo2_value', field_data)
            self.assertIn('anexo3_value', field_data)
            self.assertIn('discrepancy_analysis', field_data)
        
        # Verificar estructura de tabla para comparación de 2 documentos
        self.assertIn('status', table_info)
        self.assertIn('total_similarity', table_info)
        self.assertIn('matched_items', table_info)
        self.assertIn('anexo1_only', table_info)
        self.assertIn('anexo2_only', table_info)
        self.assertIn('summary', table_info)
        
        # Verificar que el summary de tabla tiene la estructura correcta para 2 documentos
        table_summary = table_info['summary']
        self.assertIn('matches', table_summary)
        self.assertIn('anexo1_only', table_summary)
        self.assertIn('anexo2_only', table_summary)


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