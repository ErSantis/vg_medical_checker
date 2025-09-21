"""
OCR Extractor para Anexo 3 - Documento impreso con información médica
Extrae: fecha de impresión, paciente (CC + nombre), médico tratante, lugar, tabla de insumos
"""

import re
import cv2
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from paddleocr import PaddleOCR
import logging

_PADDLE = None

def get_paddle_ocr():
    """Configuración de PaddleOCR optimizada para texto impreso"""
    global _PADDLE
    if _PADDLE is None:
        _PADDLE = PaddleOCR(lang='es')
    return _PADDLE

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convierte imagen a RGB si es necesario"""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def ocr_with_paddle(image_np: np.ndarray) -> List[dict]:
    """Ejecuta OCR con PaddleOCR y retorna tokens con posiciones"""
    rgb = ensure_rgb(image_np)
    ocr = get_paddle_ocr()
    pred = ocr.predict(rgb)
    items: List[dict] = []
    
    if not pred or len(pred) == 0:
        return items
        
    result = pred[0]
    if not isinstance(result, dict):
        return items
        
    polys = result.get('dt_polys', []) or result.get('rec_polys', [])
    texts = result.get('rec_texts', [])
    scores = result.get('rec_scores', [])
    
    min_len = min(len(polys), len(texts), len(scores))
    
    for i in range(min_len):
        poly = polys[i]
        txt = texts[i]
        score = scores[i]
        
        if len(poly) >= 4:
            # Calcular centro del token
            x_coords = [point[0] for point in poly]
            y_coords = [point[1] for point in poly]
            cx = sum(x_coords) / len(x_coords)
            cy = sum(y_coords) / len(y_coords)
            
            items.append({
                'text': txt.strip(),
                'score': score,
                'cx': cx,
                'cy': cy,
                'bbox': poly
            })
    
    return items

def extract_fecha_impresion(tokens: List[dict]) -> Optional[str]:
    """Extrae fecha de impresión del documento"""
    for token in tokens:
        text = token['text'].lower()
        
        # Buscar "fecha de impresión:" seguido de la fecha
        if 'fecha' in text and 'impresión' in text:
            # Buscar tokens cercanos que contengan fecha
            target_y = token['cy']
            
            for nearby_token in tokens:
                if abs(nearby_token['cy'] - target_y) < 30:  # Misma línea
                    fecha_text = nearby_token['text']
                    
                    # Patrón para fecha DD/MM/YYYY HH:MM
                    fecha_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2})', fecha_text)
                    if fecha_match:
                        return fecha_match.group(1)
                    
                    # Patrón alternativo DD-MM-YYYY HH:MM
                    fecha_match = re.search(r'(\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2})', fecha_text)
                    if fecha_match:
                        return fecha_match.group(1)
    
    return None

def extract_paciente_info(tokens: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    """Extrae número de identificación y nombre del paciente"""
    documento = None
    nombre = None
    
    for i, token in enumerate(tokens):
        text = token['text']
        
        # Buscar patrón "CC XXXXXXXX - NOMBRE"
        cc_match = re.search(r'CC\s*(\d{6,12})\s*-\s*(.+)', text, re.IGNORECASE)
        if cc_match:
            documento = cc_match.group(1).strip()
            nombre = cc_match.group(2).strip()
            return documento, nombre
        
        # Buscar "CC" seguido del número en tokens adyacentes
        if re.search(r'\bCC\b', text, re.IGNORECASE):
            target_y = token['cy']
            target_x = token['cx']
            
            # Buscar tokens a la derecha en la misma línea
            for j in range(i + 1, min(i + 5, len(tokens))):
                nearby_token = tokens[j]
                
                if (abs(nearby_token['cy'] - target_y) < 20 and 
                    nearby_token['cx'] > target_x):
                    
                    nearby_text = nearby_token['text']
                    
                    # Buscar número seguido de nombre
                    doc_match = re.search(r'(\d{6,12})\s*-?\s*(.+)', nearby_text)
                    if doc_match:
                        documento = doc_match.group(1).strip()
                        if len(doc_match.group(2).strip()) > 2:
                            nombre = doc_match.group(2).strip()
                        return documento, nombre
                    
                    # Solo número
                    if re.match(r'^\d{6,12}$', nearby_text.strip()):
                        documento = nearby_text.strip()
                        
                        # Buscar nombre en el siguiente token
                        if j + 1 < len(tokens):
                            next_token = tokens[j + 1]
                            if abs(next_token['cy'] - target_y) < 20:
                                nombre_text = next_token['text'].strip()
                                if len(nombre_text) > 2 and not nombre_text.isdigit():
                                    nombre = nombre_text
                        return documento, nombre
    
    return documento, nombre

def extract_medico_tratante(tokens: List[dict]) -> Optional[str]:
    """Extrae el nombre del médico tratante"""
    for token in tokens:
        text = token['text']
        
        # Caso 1: Etiqueta y valor en el mismo token
        if 'médico' in text.lower() and 'tratante' in text.lower():
            # Extraer todo lo que viene después de "médico tratante:"
            match = re.search(r'médico\s+tratante:\s*(.+)', text, re.IGNORECASE)
            if match:
                medico_value = match.group(1).strip()
                if len(medico_value) > 1:
                    return medico_value
        
        # Caso 2: Solo la etiqueta, buscar en tokens adyacentes
        if ('médico' in text.lower() and 'tratante' in text.lower() and 
            ':' in text and len(text.strip()) < 20):
            target_y = token['cy']
            
            # Buscar tokens a la derecha en la misma línea
            for nearby_token in tokens:
                if (abs(nearby_token['cy'] - target_y) < 30 and 
                    nearby_token['cx'] > token['cx']):
                    
                    medico_text = nearby_token['text'].strip()
                    if len(medico_text) > 1 and not medico_text.isdigit():
                        return medico_text
    
    return None

def extract_lugar(tokens: List[dict]) -> Optional[str]:
    """Extrae el lugar"""
    for token in tokens:
        text = token['text']
        
        # Caso 1: Etiqueta y valor en el mismo token
        if 'lugar:' in text.lower():
            # Extraer todo lo que viene después de "lugar:"
            match = re.search(r'lugar:\s*(.+)', text, re.IGNORECASE)
            if match:
                lugar_value = match.group(1).strip()
                if len(lugar_value) > 1:
                    return lugar_value
        
        # Caso 2: Solo la etiqueta, buscar en tokens adyacentes
        if 'lugar' in text.lower() and ':' in text and len(text.strip()) < 10:
            target_y = token['cy']
            
            # Buscar tokens a la derecha en la misma línea
            for i, nearby_token in enumerate(tokens):
                if (abs(nearby_token['cy'] - target_y) < 30 and 
                    nearby_token['cx'] > token['cx']):
                    
                    lugar_text = nearby_token['text'].strip()
                    if len(lugar_text) > 1 and ':' not in lugar_text:
                        return lugar_text
    
    return None

def extract_tabla_insumos(tokens: List[dict]) -> List[Dict[str, Any]]:
    """Extrae la tabla de insumos médicos"""
    insumos = []
    
    # Como el texto exacto "GASTO QUIRÚRGICO" no aparece en los tokens,
    # vamos a buscar directamente patrones de insumos médicos en todo el documento
    
    # Construir texto completo del documento
    full_text = ' '.join([token['text'] for token in tokens])
    
    # También buscar en tokens individuales que contengan información de insumos
    insumos_text = ""
    
    # Buscar tokens que contengan la etiqueta "GASTO QUIRÚRGICO" o símbolos × con números
    for token in tokens:
        text = token['text']
        # Buscar tokens que contengan × seguido de números, o la etiqueta específica
        if ('gasto' in text.lower() and 'quirúrgico' in text.lower()) or re.search(r'[×x]\s*\d+', text):
            insumos_text += " " + text
    
    # Si encontramos texto relacionado con insumos, parsearlo
    if insumos_text.strip():
        insumos = parse_insumos_text(insumos_text)
    
    # Si no encontramos nada, buscar directamente en el texto completo con patrón genérico
    if not insumos:
        # Patrón completamente genérico: cualquier texto seguido de ×número sin letra
        patron_general = r'([^,×x\n]+)\s*[×x]\s*(\d+)(?![a-zA-Z])'
        matches = re.findall(patron_general, full_text, re.IGNORECASE)
        
        for descripcion, cantidad in matches:
            descripcion = descripcion.strip()
            descripcion = re.sub(r'^[^\w\s]+', '', descripcion).strip()
            
            if len(descripcion) > 3:
                insumos.append({
                    'descripcion': descripcion,
                    'cantidad': int(cantidad)
                })
    
    return insumos

def parse_insumos_text(text: str) -> List[Dict[str, Any]]:
    """Parsea el texto de insumos y extrae los elementos individuales"""
    insumos = []
    
    # Limpiar el texto
    text = text.strip()
    
    # Buscar todas las posiciones donde aparece "GASTO QUIRÚRGICO:"
    import re
    
    # Encontrar todas las ocurrencias de "GASTO QUIRÚRGICO:"
    pattern = re.compile(r'gasto\s+quirúrgico:\s*', re.IGNORECASE)
    matches = list(pattern.finditer(text))
    
    for match in matches:
        start_pos = match.end()  # Posición después de "GASTO QUIRÚRGICO:"
        
        # Buscar el final de esta sección (próxima etiqueta en mayúsculas o final del texto)
        remaining_text = text[start_pos:]
        
        # Encontrar donde termina esta sección
        end_match = re.search(r'\s+[A-Z]{3,}[^a-z]*[A-Z]', remaining_text)
        if end_match:
            end_pos = end_match.start()
            section_text = remaining_text[:end_pos]
        else:
            section_text = remaining_text
        
        # Ahora dividir por comas y procesar cada insumo
        items = [item.strip() for item in section_text.split(',') if item.strip()]
        
        for item in items:
            # Buscar patrón: descripción × cantidad AL FINAL
            # La cantidad real está solo al final del item, no en medio
            match = re.search(r'^(.+?)\s*[×x]\s*(\d+)\s*$', item, re.IGNORECASE)
            
            if match:
                descripcion = match.group(1).strip()
                cantidad = int(match.group(2))
            else:
                # Si no hay cantidad explícita al final, asumir 1 unidad
                # NO asumir contenido específico - usar el item completo como descripción
                descripcion = item.strip()
                cantidad = 1
                
            # NO filtrar por contenido específico - solo verificar que tiene contenido válido
            if len(descripcion) > 3:
                insumos.append({
                    'descripcion': descripcion,
                    'cantidad': cantidad
                })
    
    return insumos

def normalize_fields_anexo3(fields: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Normaliza y limpia los campos extraídos"""
    normalized = {}
    
    for field, value in fields.items():
        if not value or value.strip() == "":
            normalized[field] = "[CAMPO_NO_DETECTADO_POR_OCR]"
        else:
            # Limpiar espacios extra
            cleaned_value = re.sub(r'\s+', ' ', value.strip())
            
            # Limpiezas específicas por campo
            if field == 'documento':
                # Solo números para documento
                cleaned_value = re.sub(r'[^\d]', '', cleaned_value)
                if not cleaned_value:
                    cleaned_value = "[CAMPO_NO_DETECTADO_POR_OCR]"
            
            elif field == 'nombre':
                # Limpiar caracteres especiales del nombre
                cleaned_value = re.sub(r'[^\w\s\.]', ' ', cleaned_value)
                cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
                if len(cleaned_value) < 3:
                    cleaned_value = "[CAMPO_NO_DETECTADO_POR_OCR]"
            
            normalized[field] = cleaned_value
    
    return normalized

def process_image_anexo3(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Procesa una imagen del anexo 3 y extrae todos los datos"""
    
    # Ejecutar OCR
    tokens = ocr_with_paddle(image_bgr)
    
    if not tokens:
        logging.warning("No se detectaron tokens en la imagen")
        return {
            'extracted_fields': {
                'fecha_impresion': "[CAMPO_NO_DETECTADO_POR_OCR]",
                'documento': "[CAMPO_NO_DETECTADO_POR_OCR]",
                'nombre': "[CAMPO_NO_DETECTADO_POR_OCR]",
                'medico_tratante': "[CAMPO_NO_DETECTADO_POR_OCR]",
                'lugar': "[CAMPO_NO_DETECTADO_POR_OCR]"
            },
            'tabla_insumos': [],
            'raw_tokens': []
        }
    
    # Extraer campos básicos
    fecha_impresion = extract_fecha_impresion(tokens)
    documento, nombre = extract_paciente_info(tokens)
    medico_tratante = extract_medico_tratante(tokens)
    lugar = extract_lugar(tokens)
    
    # Extraer tabla de insumos
    tabla_insumos = extract_tabla_insumos(tokens)
    
    # Crear diccionario de campos
    fields = {
        'fecha_impresion': fecha_impresion,
        'documento': documento, 
        'nombre': nombre,
        'medico_tratante': medico_tratante,
        'lugar': lugar
    }
    
    # Normalizar campos
    normalized_fields = normalize_fields_anexo3(fields)
    
    return {
        'extracted_fields': normalized_fields,
        'tabla_insumos': tabla_insumos,
        'raw_tokens': tokens
    }

def process_pdf_anexo3(path) -> Dict[str, Any]:
    """Procesa un PDF del anexo 3 y extrae todos los datos"""
    
    # Convertir a Path si es string
    if isinstance(path, str):
        path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Archivo PDF no encontrado: {path}")
    
    # Abrir PDF
    doc = fitz.open(path)
    
    if len(doc) == 0:
        raise ValueError("El PDF no contiene páginas")
    
    all_results = []
    
    # Procesar TODAS las páginas del PDF
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convertir página a imagen con alta resolución
        mat = fitz.Matrix(2.0, 2.0)  # Escalar 2x para mejor calidad
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convertir a array numpy
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Procesar imagen
        page_result = process_image_anexo3(image)
        page_result['page_number'] = page_num + 1
        all_results.append(page_result)
    
    doc.close()
    
    # Combinar resultados de todas las páginas
    combined_result = combine_page_results(all_results)
    
    return combined_result

def combine_page_results(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combina resultados de múltiples páginas"""
    
    # Usar los campos de la primera página que tenga datos válidos
    final_fields = {
        'fecha_impresion': "[CAMPO_NO_DETECTADO_POR_OCR]",
        'documento': "[CAMPO_NO_DETECTADO_POR_OCR]",
        'nombre': "[CAMPO_NO_DETECTADO_POR_OCR]",
        'medico_tratante': "[CAMPO_NO_DETECTADO_POR_OCR]",
        'lugar': "[CAMPO_NO_DETECTADO_POR_OCR]"
    }
    
    all_insumos = []
    all_tokens = []
    
    for page_result in page_results:
        # Actualizar campos si encontramos datos válidos
        for field, value in page_result['extracted_fields'].items():
            if value != "[CAMPO_NO_DETECTADO_POR_OCR]" and final_fields[field] == "[CAMPO_NO_DETECTADO_POR_OCR]":
                final_fields[field] = value
        
        # Combinar insumos
        all_insumos.extend(page_result['tabla_insumos'])
        
        # Combinar tokens
        all_tokens.extend(page_result['raw_tokens'])
    
    # Eliminar duplicados de insumos - mantener solo UNA instancia de cada insumo único
    unique_insumos = []
    for insumo in all_insumos:
        # Normalizar descripción para comparación (solo espacios y mayúsculas/minúsculas)
        desc_normalizada = re.sub(r'\s+', ' ', insumo['descripcion'].lower().strip())
        
        # Buscar si ya existe un insumo similar
        encontrado = False
        for existing in unique_insumos:
            existing_desc = re.sub(r'\s+', ' ', existing['descripcion'].lower().strip())
            
            if existing_desc == desc_normalizada:
                # Ya existe, no agregar duplicado
                encontrado = True
                break
        
        if not encontrado:
            # Agregar sin modificar la descripción original
            unique_insumos.append({
                'descripcion': insumo['descripcion'],
                'cantidad': insumo['cantidad']
            })
    
    # Limpieza final: revisar que no haya duplicados antes de retornar
    final_insumos = []
    for insumo in unique_insumos:
        # Normalizar descripción para comparación más robusta
        desc_clean = re.sub(r'\s+', ' ', insumo['descripcion'].lower().strip())
        desc_clean = desc_clean.replace('×', 'x')  # Normalizar símbolo multiplicación
        
        # Buscar si ya existe en la lista final
        ya_existe = False
        for existing in final_insumos:
            existing_clean = re.sub(r'\s+', ' ', existing['descripcion'].lower().strip())
            existing_clean = existing_clean.replace('×', 'x')
            
            if existing_clean == desc_clean:
                ya_existe = True
                break
        
        if not ya_existe:
            final_insumos.append(insumo)
    
    return {
        'extracted_fields': final_fields,
        'tabla_insumos': final_insumos,
        'raw_tokens': all_tokens
    }

def main():
    """Función principal para pruebas"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extractor OCR para Anexo 3')
    parser.add_argument('pdf_path', help='Ruta al archivo PDF del anexo 3')
    parser.add_argument('--output', '-o', help='Archivo de salida JSON (opcional)')
    
    args = parser.parse_args()
    
    try:
        # Procesar PDF
        result = process_pdf_anexo3(Path(args.pdf_path))
        
        # Mostrar resultados
        print("=== ANEXO 3 - EXTRACCIÓN OCR ===")
        print(f"Fecha de impresión: {result['extracted_fields']['fecha_impresion']}")
        print(f"Documento: {result['extracted_fields']['documento']}")
        print(f"Nombre: {result['extracted_fields']['nombre']}")
        print(f"Médico tratante: {result['extracted_fields']['medico_tratante']}")
        print(f"Lugar: {result['extracted_fields']['lugar']}")
        print(f"\nInsumos encontrados: {len(result['tabla_insumos'])}")
        
        for i, insumo in enumerate(result['tabla_insumos'], 1):
            print(f"  {i}. {insumo['descripcion']} x{insumo['cantidad']}")
        
        print(f"\nTokens detectados: {len(result['raw_tokens'])}")
        
        # Guardar resultado si se especifica archivo de salida
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nResultado guardado en: {args.output}")
    
    except Exception as e:
        print(f"Error procesando archivo: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())