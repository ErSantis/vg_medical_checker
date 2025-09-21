"""
Test de debug para ver los tokens detectados en anexo 3
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ocr.ocr_extractor_anexo3 import process_pdf_anexo3

def debug_anexo3():
    """Debug para ver tokens detectados"""
    
    # Buscar el archivo anexo3.pdf
    test_dir = Path(__file__).parent
    pdf_path = test_dir / "tests" / "fixtures" / "anexo3.pdf"
    
    if not pdf_path.exists():
        print(f"Archivo no encontrado: {pdf_path}")
        return
    
    # Procesar el PDF
    result = process_pdf_anexo3(pdf_path)
    
    # Mostrar todos los tokens detectados
    print("=== TODOS LOS TOKENS DETECTADOS ===")
    for i, token in enumerate(result['raw_tokens']):
        print(f"{i:3d}: '{token['text']}' (cx:{token['cx']:.1f}, cy:{token['cy']:.1f}, score:{token['score']:.2f})")
    
    print(f"\nTotal tokens: {len(result['raw_tokens'])}")
    
    # Buscar tokens específicos
    print("\n=== BÚSQUEDA DE ETIQUETAS ===")
    
    # Buscar "lugar"
    lugar_tokens = [t for t in result['raw_tokens'] if 'lugar' in t['text'].lower()]
    print(f"Tokens con 'lugar': {len(lugar_tokens)}")
    for token in lugar_tokens:
        print(f"  '{token['text']}' (cx:{token['cx']:.1f}, cy:{token['cy']:.1f})")
    
    # Buscar "médico"
    medico_tokens = [t for t in result['raw_tokens'] if 'médico' in t['text'].lower() or 'medico' in t['text'].lower()]
    print(f"Tokens con 'médico': {len(medico_tokens)}")
    for token in medico_tokens:
        print(f"  '{token['text']}' (cx:{token['cx']:.1f}, cy:{token['cy']:.1f})")
    
    # Buscar "gasto"
    gasto_tokens = [t for t in result['raw_tokens'] if 'gasto' in t['text'].lower()]
    print(f"Tokens con 'gasto': {len(gasto_tokens)}")
    for token in gasto_tokens:
        print(f"  '{token['text']}' (cx:{token['cx']:.1f}, cy:{token['cy']:.1f})")
    
    # Buscar "quirúrgico"
    quirurgico_tokens = [t for t in result['raw_tokens'] if 'quirúrgico' in t['text'].lower() or 'quirurgico' in t['text'].lower()]
    print(f"Tokens con 'quirúrgico': {len(quirurgico_tokens)}")
    for token in quirurgico_tokens:
        print(f"  '{token['text']}' (cx:{token['cx']:.1f}, cy:{token['cy']:.1f})")
    
    # Buscar tornillos y placas
    insumos_tokens = [t for t in result['raw_tokens'] if any(keyword in t['text'].lower() for keyword in ['tornillo', 'placa', 'encefálico', 'curvavelos'])]
    print(f"Tokens con insumos: {len(insumos_tokens)}")
    for token in insumos_tokens:
        print(f"  '{token['text']}' (cx:{token['cx']:.1f}, cy:{token['cy']:.1f})")

if __name__ == '__main__':
    debug_anexo3()