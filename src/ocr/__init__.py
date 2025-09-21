"""
OCR Module - Extractores de texto para documentos médicos

Este módulo contiene extractores especializados para diferentes tipos de documentos:
- ocr_extractor_anexo1: Para "Reporte de Gasto Quirúrgico" 
- ocr_extractor_anexo2: Para "Gastos de Materiales de Osteosíntesis"
- ocr_extractor_anexo3: Para documentos impresos con información médica

Todos usan PaddleOCR con configuración optimizada para texto médico.
"""

from .ocr_extractor_anexo1 import (
    process_image,
    process_pdf,
    extract_table_data,
    get_paddle_ocr
)

from .ocr_extractor_anexo2 import (
    process_image_anexo2,
    process_pdf_anexo2,
    extract_table_anexo2,
    extract_kv_from_anexo2
)

from .ocr_extractor_anexo3 import (
    process_image_anexo3,
    process_pdf_anexo3,
    extract_tabla_insumos,
    extract_paciente_info
)

__all__ = [
    'process_image',
    'process_pdf', 
    'extract_table_data',
    'process_image_anexo2',
    'process_pdf_anexo2',
    'extract_table_anexo2',
    'extract_kv_from_anexo2',
    'process_image_anexo3',
    'process_pdf_anexo3',
    'extract_tabla_insumos',
    'extract_paciente_info',
    'get_paddle_ocr'
]