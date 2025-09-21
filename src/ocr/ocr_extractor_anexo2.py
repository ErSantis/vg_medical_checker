"""
OCR extractor específico para anexo 2 - "Gastos de Materiales de Osteosíntesis"
- Formato manuscrito con campos diferentes al anexo 1
- Extrae campos: fecha, paciente, documento, empresa, cirujano, diagnóstico, procedimiento
- Usa PaddleOCR con configuración optimizada para texto manuscrito
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR

try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

_PADDLE = None

def get_paddle_ocr():
    global _PADDLE
    if _PADDLE is None:
        _PADDLE = PaddleOCR(lang='es')
    return _PADDLE

# ----------------------------
# OCR helpers
# ----------------------------

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def ocr_with_paddle(image_np: np.ndarray) -> List[dict]:
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
        sc = scores[i]
        if hasattr(poly, 'tolist'):
            pts = [(float(x), float(y)) for x, y in poly.tolist()]
        else:
            pts = [(float(x), float(y)) for x, y in poly]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        y_top = min(ys)
        items.append({
            'box': pts,
            'text': str(txt).strip(),
            'score': float(sc),
            'cx': cx,
            'cy': cy,
            'y_top': y_top,
        })
    
    # Filtrar items con score mínimo más bajo para capturar nombres manuscritos
    items = [item for item in items if item['score'] >= 0.1]  # Umbral aún más bajo
    items.sort(key=lambda d: (int(d['y_top'] // 20), d['cx']))
    return items

# ----------------------------
# Labels específicos para anexo 2
# ----------------------------
LABELS_ANEXO2 = {
    'fecha': ['fecha'],
    'paciente': ['nombre y apellido del paciente', 'nombre', 'apellido', 'paciente'],
    'documento': ['documento de identidad', 'documento', 'identidad', 'cc', 'cedula'],
    'empresa': ['empresa proveedora', 'empresa', 'proveedora'],
    'cirujano': ['cirujano', 'medico', 'doctor'],
    'diagnostico': ['diagnostico', 'diagnóstico'],
    'procedimiento': ['procedimiento']
}

def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("ú","u").replace("ó","o").replace("í","i").replace("á","a").replace("é","e").replace("ñ","n")
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()

LABEL_INDEX_ANEXO2: Dict[str, str] = {}
for field, keys in LABELS_ANEXO2.items():
    for k in keys:
        LABEL_INDEX_ANEXO2[_norm(k)] = field

def is_label_token_anexo2(token_text: str) -> Optional[str]:
    """Detecta si un token es una etiqueta para anexo 2"""
    t = _norm(token_text)
    
    # Buscar coincidencias exactas o parciales
    for k, field in LABEL_INDEX_ANEXO2.items():
        if k in t or t in k:
            return field
    
    # Patrones específicos para anexo 2
    if 'fecha' in t:
        return 'fecha'
    if 'nombre' in t and ('apellido' in t or 'paciente' in t):
        return 'paciente'
    if 'documento' in t and 'identidad' in t:
        return 'documento'
    if 'empresa' in t and 'proveedora' in t:
        return 'empresa'
    if 'cirujano' in t:
        return 'cirujano'
    if 'diagnostico' in t:
        return 'diagnostico'
    if 'procedimiento' in t:
        return 'procedimiento'
    
    return None

def _same_line(y1: float, y2: float, tol: int = 25) -> bool:
    """Tolerancia mayor para texto manuscrito"""
    return abs(y1 - y2) <= tol

def group_lines(items: List[dict], tol: int = 25) -> List[List[dict]]:
    lines: List[List[dict]] = []
    for it in items:
        placed = False
        for line in lines:
            if _same_line(line[0]['cy'], it['cy'], tol):
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])
    for line in lines:
        line.sort(key=lambda d: d['cx'])
    lines.sort(key=lambda line: sum([t['cy'] for t in line]) / len(line))
    return lines

def group_by_lines(items: List[dict], y_threshold: int = 25) -> List[List[dict]]:
    """Agrupa tokens por líneas basándose en la posición Y"""
    lines: List[List[dict]] = []
    
    for item in items:
        placed = False
        for line in lines:
            # Verificar si está en la misma línea
            if any(abs(item['y_top'] - existing['y_top']) <= y_threshold for existing in line):
                line.append(item)
                placed = True
                break
        
        if not placed:
            lines.append([item])
    
    # Ordenar cada línea por posición X y las líneas por posición Y
    for line in lines:
        line.sort(key=lambda t: t['cx'])
    
    lines.sort(key=lambda line: min(t['y_top'] for t in line))
    
    return lines

def extract_kv_from_anexo2(items: List[dict]) -> Dict[str, Optional[str]]:
    """Extrae campos clave-valor específicos para anexo 2"""
    result: Dict[str, Optional[str]] = {k: None for k in LABELS_ANEXO2.keys()}
    
    # Filtrar tokens con score decente para texto manuscrito
    good_tokens = [t for t in items if t['score'] >= 0.25]
    
    # También incluir tokens con score muy bajo que podrían ser nombres manuscritos
    low_score_tokens = [t for t in items if 0.1 <= t['score'] < 0.25]  # Umbral más agresivo
    
    # Crear texto completo para buscar patrones
    all_text = " ".join([t['text'] for t in good_tokens]).lower()
    
    # FECHA - buscar patrón después de "FECHA:"
    fecha_match = re.search(r'fecha[:\s_]*([0-9]{1,2}[\/\-\s]*[0-9]{1,2}[\/\-\s]*[0-9]{2,4})', all_text)
    if fecha_match:
        fecha_raw = fecha_match.group(1)
        # Limpiar espacios y formatear
        fecha_clean = re.sub(r'\s+', '', fecha_raw).replace('/', '-')
        result['fecha'] = fecha_clean
    
    # DOCUMENTO DE IDENTIDAD - buscar números después de "documento de identidad"
    doc_match = re.search(r'documento[^:]*identidad[:\s_]*([0-9\s]{8,})', all_text)
    if doc_match:
        doc_raw = doc_match.group(1)
        # Limpiar espacios
        result['documento'] = re.sub(r'\s+', '', doc_raw)
    
    # PACIENTE - buscar después de "nombre y apellido del paciente" o similar
    paciente_patterns = [
        r'nombre[^:]*apellido[^:]*paciente[:\s]*([a-zA-ZÀ-ÿ\s]{5,30}?)(?=documento|empresa|cc|123|$)',
        r'paciente[:\s]*([a-zA-ZÀ-ÿ\s]{5,30}?)(?=documento|empresa|cc|123|$)',
        r'nombre[:\s]*([a-zA-ZÀ-ÿ\s]{5,30}?)(?=documento|empresa|cc|123|$)'
    ]
    for pattern in paciente_patterns:
        match = re.search(pattern, all_text)
        if match:
            result['paciente'] = match.group(1).strip()
            break
    
    # FALLBACK: Si detectamos el label pero no el texto manuscrito, 
    # buscar cualquier token en la zona del paciente
    if not result['paciente'] and 'nombre' in all_text and 'apellido' in all_text and 'paciente' in all_text:
        # Sabemos que existe el campo, pero el texto manuscrito no se detectó
        # Buscar tokens manuscritos en esa zona
        for token in good_tokens + low_score_tokens:
            y = token.get('y_top', 0)
            text = token['text'].strip()
            
            # Buscar en zona entre fecha (515) y documento (678)
            if (530 <= y <= 670 and 
                len(text) >= 3 and 
                not text.isdigit() and
                'fecha' not in text.lower() and
                'documento' not in text.lower() and
                'nombre' not in text.lower() and
                'apellido' not in text.lower() and
                'paciente' not in text.lower() and
                any(c.isalpha() for c in text)):
                result['paciente'] = text
                break
        
    # ESTRATEGIA ESPECIAL: Si no encontramos el label del paciente pero sí FECHA y DOCUMENTO,
    # es muy probable que el OCR no detectó esa línea completa por contraste bajo
    if not result['paciente']:
        # Verificar que tenemos fecha y documento pero no paciente
        tiene_fecha = any('fecha' in t['text'].lower() for t in good_tokens)
        tiene_documento = any('documento' in t['text'].lower() for t in good_tokens)
        tiene_label_paciente = any(any(word in t['text'].lower() for word in ['nombre', 'apellido', 'paciente']) for t in good_tokens)
        
        if tiene_fecha and tiene_documento and not tiene_label_paciente:
            # El campo de paciente existe pero no fue detectado por OCR
            # Buscar cualquier token suelto en la zona que podría ser el nombre
            for token in good_tokens + low_score_tokens:
                y = token.get('y_top', 0)
                text = token['text'].strip()
                
                # Buscar en zona entre fecha y documento (con margen más amplio)
                if (520 <= y <= 675 and 
                    len(text) >= 3 and 
                    not text.isdigit() and
                    'fecha' not in text.lower() and
                    'documento' not in text.lower() and
                    'identidad' not in text.lower() and
                    any(c.isalpha() for c in text)):
                    result['paciente'] = text
                    break
            
            # Si no encontramos ningún texto en esa zona, es probable que no se detectó
            if not result['paciente']:
                result['paciente'] = "[No detectado]"
    
    # Si no se encontró paciente en el texto principal, buscar en tokens con score bajo
    # que podrían ser nombres manuscritos mal reconocidos
    if not result['paciente']:
        # ESTRATEGIA 1: Buscar tokens entre fecha y documento que podrían ser nombres
        if low_score_tokens:
            fecha_y = None
            doc_y = None
            
            for token in good_tokens:
                if 'fecha' in token['text'].lower():
                    fecha_y = token.get('y_top', 0)
                elif 'documento' in token['text'].lower():
                    doc_y = token.get('y_top', 0)
            
            # Si encontramos las posiciones Y, buscar nombres entre ellas
            if fecha_y and doc_y:
                for token in low_score_tokens:
                    token_y = token.get('y_top', 0)
                    if fecha_y < token_y < doc_y:
                        # Es un candidato para nombre del paciente
                        text = token['text'].strip()
                        if len(text) >= 3 and not text.isdigit():
                            result['paciente'] = text
                            break
            
            # También buscar tokens con score bajo que parezcan nombres (no dígitos)
            if not result['paciente']:
                for token in low_score_tokens:
                    text = token['text'].strip()
                    if (len(text) >= 4 and 
                        not text.isdigit() and 
                        any(c.isalpha() for c in text) and
                        token['score'] > 0.15):  # Un poco más restrictivo
                        result['paciente'] = text
                        break
        
        # ESTRATEGIA 2: Buscar por posición aproximada basándose en la imagen
        # El campo de paciente debería estar entre Y:600-650 aproximadamente 
        if not result['paciente']:
            all_tokens = good_tokens + low_score_tokens
            patient_candidates = []
            
            for token in all_tokens:
                y = token.get('y_top', 0)
                # Buscar en la zona donde debería estar el nombre del paciente
                if 580 <= y <= 660:  # Zona entre fecha y documento
                    text = token['text'].strip()
                    if (len(text) >= 3 and 
                        not text.isdigit() and 
                        any(c.isalpha() for c in text) and
                        'fecha' not in text.lower() and
                        'documento' not in text.lower()):
                        patient_candidates.append((token, text))
            
            # Si encontramos candidatos, tomar el que tenga mejor score
            if patient_candidates:
                best_candidate = max(patient_candidates, key=lambda x: x[0]['score'])
                result['paciente'] = best_candidate[1]
    
    
    # CIRUJANO - buscar después de "cirujano:" hasta antes de "diagnóstico"
    cirujano_match = re.search(r'cirujano[:\s_]*([a-zA-Z\s]{5,25})(?:\s+diagn|$)', all_text)
    if cirujano_match:
        result['cirujano'] = cirujano_match.group(1).strip()
    
    # DIAGNÓSTICO - buscar después de "diagnóstico:" hasta antes de "procedimiento"
    diagnostico_match = re.search(r'diagn[oó]stico[:\s]*([^:]{5,50}?)(?=procedimiento|$)', all_text)
    if diagnostico_match:
        result['diagnostico'] = diagnostico_match.group(1).strip()
    
    # PROCEDIMIENTO - buscar después de "procedimiento:" hasta antes de "cantidad" o texto de tabla
    procedimiento_match = re.search(r'procedimiento[:\s]*([^:]{10,200}?)(?=cantidad|descripción|redaccion|oseo|$)', all_text)
    if procedimiento_match:
        result['procedimiento'] = procedimiento_match.group(1).strip()
    
    return result

def extract_table_anexo2(items: List[dict]) -> List[Dict[str, str]]:
    """Extrae datos de la tabla de materiales del anexo 2"""
    # Buscar tokens que indiquen el inicio de la tabla
    table_start_y = None
    table_end_y = None
    
    for item in items:
        text = item['text'].lower()
        y = item['y_top']
        
        # Buscar encabezados de tabla
        if 'cantidad' in text and 'descripción' in text:
            table_start_y = y + 50  # Comenzar después del encabezado
        elif 'cantidad' in text or 'descripción' in text or 'observaciones' in text:
            if table_start_y is None:
                table_start_y = y + 30
        
        # Buscar fin de tabla (cuando aparezcan otros elementos del formulario)
        if any(word in text for word in ['especialista', 'instrumentador', 'elaboró']) and table_start_y:
            table_end_y = y - 50
            break
    
    if not table_start_y:
        return []
    
    # Si no encontramos el final, usar un valor predeterminado más restrictivo
    if not table_end_y:
        table_end_y = table_start_y + 300  # Tabla más pequeña, más restrictiva
    
    # Filtrar tokens que están en el área de la tabla
    table_tokens = []
    for item in items:
        y = item['y_top']
        # Más restrictivo: solo tokens dentro del rango de tabla Y con score decente
        if (table_start_y <= y <= table_end_y and 
            item['score'] >= 0.25 and 
            y < 1600):  # Corte adicional para evitar tokens muy abajo
            table_tokens.append(item)
    
    if not table_tokens:
        return []
    
    # Agrupar tokens por filas (usando clustering por Y) - más restrictivo
    rows = group_by_lines(table_tokens, y_threshold=20)  # Umbral más estricto
    
    # Procesar cada fila para extraer cantidad, descripción y observaciones
    table_data = []
    
    for row_tokens in rows:
        if len(row_tokens) == 0:
            continue
            
        # Ordenar tokens por X (de izquierda a derecha)
        row_tokens.sort(key=lambda t: t['cx'])
        
        # Extraer texto de la fila
        row_text = " ".join([t['text'] for t in row_tokens])
        
        # Saltear filas que no contienen datos útiles
        if (len(row_text.strip()) < 3 or 
            any(header in row_text.lower() for header in ['cantidad', 'descripción', 'observaciones']) or
            'especialista' in row_text.lower() or
            'instrumentador' in row_text.lower() or
            'ldun' in row_text.lower() or  # Filtrar este token específico
            'elaboró' in row_text.lower() or
            'revisó' in row_text.lower() or
            'aprobó' in row_text.lower() or
            len(row_text.strip()) < 5):  # Filtrar tokens muy cortos
            continue
        
        # Intentar extraer cantidad (primer número/texto)
        cantidad = ""
        descripcion = ""
        observaciones = ""
        
        # Buscar cantidad (usualmente al inicio) - mejorada
        # Primero intentar detectar números claros
        cantidad_match = re.match(r'^(\d+)\s*', row_text)
        if cantidad_match:
            cantidad = cantidad_match.group(1)
            descripcion = row_text[len(cantidad_match.group(0)):].strip()
        else:
            # Si no hay número claro, buscar letras que podrían ser números mal leídos
            # Buscar patrones como "V", "l", "I" que podrían ser "1"
            cantidad_letter_match = re.match(r'^([VvIl1])\s*', row_text)
            if cantidad_letter_match:
                letra = cantidad_letter_match.group(1)
                # Convertir letras mal leídas a números
                if letra.upper() in ['V', 'I', 'L']:
                    cantidad = "1"
                else:
                    cantidad = letra
                descripcion = row_text[len(cantidad_letter_match.group(0)):].strip()
            else:
                # Si no encontramos cantidad al inicio, asumir que toda la línea es descripción
                descripcion = row_text.strip()
                
                # Como último recurso, buscar si hay números dispersos en la línea
                # que podrían indicar cantidad
                nums_in_line = re.findall(r'\b(\d+)\b', row_text)
                if nums_in_line:
                    # Tomar el primer número como posible cantidad
                    potential_qty = nums_in_line[0]
                    # Solo si es un número pequeño (probable cantidad)
                    if int(potential_qty) <= 10:
                        cantidad = potential_qty
                        # Remover ese número de la descripción
                        descripcion = re.sub(r'\b' + potential_qty + r'\b', '', row_text, 1).strip()
                        
        # CORRECCIÓN ESPECÍFICA: Si en la descripción vemos "× 1" o "x 1", 
        # la cantidad probablemente sea 1, no lo que detectamos
        if re.search(r'[×x]\s*1(?:Ader|$|\s)', descripcion, re.IGNORECASE):
            cantidad = "1"
            # Limpiar la descripción quitando números erróneos al inicio
            descripcion = re.sub(r'^\d+\s*', '', descripcion).strip()
        
        # Limpiar descripción de espacios extra
        descripcion = re.sub(r'\s+', ' ', descripcion).strip()
        
        # Para anexo 2, generalmente no hay observaciones separadas
        # La descripción incluye toda la información del producto
        
        if cantidad or descripcion:
            table_data.append({
                'cantidad': cantidad,
                'descripcion': descripcion,
                'observaciones': observaciones
            })
    
    return table_data

def normalize_fields_anexo2(fields: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Normalización específica para anexo 2"""
    out = dict(fields)
    
    # Normalizar fecha
    if out.get('fecha'):
        fecha_str = str(out['fecha'])
        nums = re.findall(r"\d+", fecha_str)
        if len(nums) >= 3:
            dd, mm, yy = nums[:3]
            try:
                if len(yy) == 2:
                    yy = ('20' + yy) if int(yy) < 50 else ('19' + yy)
                out['fecha'] = f"{int(dd):02d}-{int(mm):02d}-{yy}"
            except Exception:
                pass
    
    # Limpiar espacios en todos los campos de texto
    for k in out.keys():
        if k in out and out[k] is not None:
            s = str(out[k])
            s = re.sub(r"\s+", " ", s).strip(" \\t:.-")
            out[k] = s if s else None
    
    return out

def process_image_anexo2(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Procesa imagen específica para anexo 2"""
    items = ocr_with_paddle(image_bgr)
    kv = extract_kv_from_anexo2(items)
    kv = normalize_fields_anexo2(kv)
    
    # Extraer datos de la tabla
    table_data = extract_table_anexo2(items)
    
    return {
        'extracted_fields': kv,
        'table_data': table_data,
    }

def process_pdf_anexo2(path: Path) -> Dict[str, Any]:
    """Procesa PDF específico para anexo 2"""
    if not HAS_PDF:
        raise RuntimeError("PyMuPDF no instalado.")
    doc = fitz.open(str(path))
    results = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(3,3))
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        res = process_image_anexo2(img)
        res['page'] = i+1
        results.append(res)
    doc.close()
    if len(results) == 1:
        return results[0]
    return {'pages': results}

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description='Extractor OCR específico para anexo 2')
    ap.add_argument('input', type=str, help='Ruta a imagen (jpg/png) o PDF del anexo 2')
    ap.add_argument('--out_json', type=str, default=None, help='Guardar salida en JSON')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    if in_path.suffix.lower() in {'.pdf'}:
        payload = process_pdf_anexo2(in_path)
    else:
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('No se pudo abrir la imagen')
        payload = process_image_anexo2(img)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()