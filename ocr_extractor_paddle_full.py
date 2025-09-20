"""
OCR de cabecera para "Reporte de Gasto Quirúrgico" usando PaddleOCR (ES)
- Detecta y reconoce texto
- Extrae campos clave por proximidad a etiquetas: fecha, paciente, identificación, ciudad,
  especialista, procedimiento, institución, cliente, remisión
- Soporta imagen (JPG/PNG) o PDF (con PyMuPDF)
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
    items.sort(key=lambda d: (int(d['y_top'] // 20), d['cx']))
    return items

# ----------------------------
# Agrupar líneas y extracción KV
# ----------------------------
LABELS = {
    'fecha':        ['fecha'],
    'paciente':     ['paciente'],
    'identificacion':['identificación','identificacion','cc','doc','documento'],
    'ciudad':       ['ciudad','cludad','municipio'],
    'especialista': ['especialista','médico','medico','doctor'],
    'procedimiento':['procedimiento','procedimlento'],
    'institucion':  ['institución','institucion'],
    'cliente':      ['cliente'],
    'remision':     ['no.remisión','no remisión','n° remisión','remisión','remision','n°','no.']
}

def _same_line(y1: float, y2: float, tol: int = 18) -> bool:
    return abs(y1 - y2) <= tol

def group_lines(items: List[dict], tol: int = 18) -> List[List[dict]]:
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

def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("ú","u").replace("ó","o").replace("í","i").replace("á","a").replace("é","e").replace("ñ","n")
    s = s.replace("cludad","ciudad")
    s = re.sub(r"[^a-z0-9\.]", "", s)
    return s

LABEL_INDEX: Dict[str, str] = {}
for field, keys in LABELS.items():
    for k in keys:
        LABEL_INDEX[_norm(k)] = field

def is_label_token(token_text: str) -> Optional[str]:
    t = _norm(token_text)
    # Evitar tokens que contengan "firma" o "sello" para especialista
    if 'firma' in t or 'sello' in t:
        return None
    # Dr. y Dr son valores, no etiquetas - no filtrarlos
    for k, field in LABEL_INDEX.items():
        if t.startswith(k):
            return field
    return None

def extract_table_data(items: List[dict]) -> List[dict]:
    """Extrae los datos de la tabla de insumos capturando solo las 7 filas reales"""
    # Filtrar tokens con buena confianza
    good_tokens = [t for t in items if t['score'] >= 0.35]
    
    # Buscar límites más precisos de la tabla
    table_start_y = None
    table_end_y = None
    cantidad_header_y = None
    
    for token in good_tokens:
        text = token['text'].lower()
        # Buscar header "Cantidad" para delimitar mejor
        if 'cantidad' in text and cantidad_header_y is None:
            cantidad_header_y = token['y_top']
            table_start_y = token['y_top'] + 40  # Después del header
        # Buscar "Sticker:" como límite inferior
        if 'sticker' in text and table_end_y is None:
            table_end_y = token['y_top'] - 20
    
    if not table_start_y or not table_end_y:
        return []
    
    # Filtrar tokens en área de tabla con límites más estrictos
    table_tokens = [t for t in good_tokens 
                   if table_start_y <= t['y_top'] <= table_end_y
                   and t['cx'] >= 80]  # Excluir tokens muy a la izquierda
    
    # Agrupar por líneas con tolerancia ajustada
    lines = group_lines(table_tokens, tol=25)
    
    def extract_referencia_lote_combined(text):
        """Extrae referencia+lote como un campo combinado"""
        # Patrón principal: 5 dígitos - seguido de más dígitos
        pattern = r'(\d{5}-\d+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        
        # Patrón alternativo: 4 dígitos - seguido de dígitos
        alt_pattern = r'(\d{4,5}-\d+)'
        alt_match = re.search(alt_pattern, text)
        if alt_match:
            return alt_match.group(1)
        
        return None
    
    def is_valid_table_row(line):
        """Valida que una línea sea realmente una fila de tabla válida"""
        # Debe tener al menos 3 tokens
        if len(line) < 3:
            return False
        
        # Debe tener un número de fila al inicio (cx < 150)
        has_row_number = any(token['cx'] < 150 and re.match(r'^\d{1,2}$', token['text'].strip()) 
                           for token in line)
        
        # Debe tener una cantidad al final (cx > 1600)
        has_quantity = any(token['cx'] > 1600 and re.match(r'^\d{1,3}$', token['text'].strip()) 
                         for token in line)
        
        # Debe tener algún patrón de referencia-lote
        has_reference = any(extract_referencia_lote_combined(token['text']) 
                          for token in line)
        
        return has_row_number and has_quantity and has_reference
    
    table_data = []
    
    # Filtrar solo las líneas que son filas válidas de tabla
    valid_lines = [line for line in lines if is_valid_table_row(line)]
    
    # Procesar solo las primeras 7 líneas válidas (las filas reales de la tabla)
    for line_idx, line in enumerate(valid_lines[:7]):
        try:
            # Ordenar tokens por posición horizontal
            line.sort(key=lambda x: x['cx'])
            
            referencia_lote = None
            descripcion_parts = []
            cantidad = None
            
            for token in line:
                text = token['text'].strip()
                cx = token['cx']
                
                # Columna 1: N° (cx < 150) - ignorar números de fila
                if cx < 150 and re.match(r'^\d{1,2}$', text):
                    continue
                
                # Buscar referencia+lote en cualquier posición
                if not referencia_lote:
                    ref_lote = extract_referencia_lote_combined(text)
                    if ref_lote:
                        referencia_lote = ref_lote
                        # Limpiar la descripción removiendo la referencia+lote
                        desc_text = text.replace(ref_lote, '', 1).strip()
                        desc_text = re.sub(r'^[^\w]+', '', desc_text).strip()
                        if desc_text and len(desc_text) > 3:
                            descripcion_parts.append(desc_text)
                        continue
                
                # Buscar descripción (área central de la tabla)
                if 400 <= cx <= 1600:
                    if len(text) > 1 and not re.match(r'^\d{1,3}$', text):
                        # Solo agregar si no contiene la referencia+lote ya extraída
                        if not (referencia_lote and referencia_lote in text):
                            descripcion_parts.append(text)
                
                # Columna cantidad (cx > 1600)
                elif cx > 1600 and re.match(r'^\d{1,3}$', text):
                    try:
                        cantidad = int(text)
                    except:
                        continue
            
            # Construir descripción final
            descripcion = ' '.join(descripcion_parts).strip() if descripcion_parts else None
            
            # Solo agregar si tenemos datos completos
            if referencia_lote and cantidad is not None:
                table_data.append({
                    'referencia_lote': referencia_lote,
                    'descripcion': descripcion,
                    'cantidad': cantidad
                })
                
        except Exception as e:
            # Si hay error en una línea, continuar con la siguiente
            continue    
    
    return table_data

def extract_kv_from_paddle(items: List[dict]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {k: None for k in LABELS.keys()}
    lines = group_lines(items)
    for li, line in enumerate(lines):
        labels_in_line = []
        for idx, tok in enumerate(line):
            field = is_label_token(tok['text'])
            if field is not None:
                labels_in_line.append((idx, field))
        if not labels_in_line:
            continue
        for pos, field in labels_in_line:
            if result[field] is not None:
                continue
            next_label_pos = None
            for (p2, _f2) in labels_in_line:
                if p2 > pos:
                    next_label_pos = p2
                    break
            right_tokens = line[pos+1: next_label_pos] if next_label_pos else line[pos+1:]
            right_tokens = [t for t in right_tokens if t['score'] >= 0.40]
            value = " ".join([t['text'] for t in right_tokens]).strip()
            if not value and li + 1 < len(lines):
                next_line = lines[li+1]
                if not next_line or is_label_token(next_line[0]['text']):
                    pass
                else:
                    value = " ".join([t['text'] for t in next_line if t['score'] >= 0.40]).strip()
            
            if value:
                parts = []
                for tok in value.split():
                    if is_label_token(tok):
                        break
                    parts.append(tok)
                value = " ".join(parts).strip(": ")
                if field == 'fecha':
                    m = re.search(r"\b(\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4})\b", value)
                    if m:
                        value = m.group(1).replace('/', '-')
                if field in ('identificacion','remision'):
                    m = re.search(r"\d{3,}", value)
                    if m:
                        value = m.group(0)
                if field == 'paciente':
                    # Si el valor contiene "identificación" o "identificcion", separarlo completamente
                    if 'identific' in value.lower():
                        # Dividir por cualquier variante de "identific" seguido de dos puntos
                        parts = re.split(r'identific[a-záéíóúñ]*\s*:\s*', value, flags=re.I)
                        if len(parts) > 1:
                            value = parts[0].strip(' :')
                            # Extraer la identificación si no la tenemos aún
                            if result['identificacion'] is None:
                                remainder = parts[1]
                                id_match = re.search(r'\d{3,}', remainder)
                                if id_match:
                                    result['identificacion'] = id_match.group(0)
                value = re.sub(r"^(Ciudad|Cludad|Instituci[oó]n)\s*:\s*", "", value, flags=re.I).strip()
                if value:
                    result[field] = value
    return result

# ----------------------------
# Fallback: franjas naranjas
# ----------------------------

def find_orange_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ranges = [
        (np.array([5,  30,  80], np.uint8), np.array([25, 255, 255], np.uint8)),
        (np.array([8,  60, 100], np.uint8), np.array([28, 255, 255], np.uint8)),
        (np.array([10, 80, 120], np.uint8), np.array([30, 255, 255], np.uint8)),
    ]
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    return mask

def ocr_orange_regions(image_bgr: np.ndarray) -> List[dict]:
    mask = find_orange_mask(image_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items_all: List[dict] = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 2000 or h < 18:
            continue
        pad = 10
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(image_bgr.shape[1], x + w + pad)
        y1 = min(image_bgr.shape[0], y + h + pad)
        roi = image_bgr[y0:y1, x0:x1]
        items = ocr_with_paddle(roi)
        for it in items:
            it['cx'] += x0
            it['cy'] += y0
            it['y_top'] += y0
        items_all.extend(items)
    items_all.sort(key=lambda d: (int(d['y_top'] // 20), d['cx']))
    return items_all

# ----------------------------
# Normalización mínima
# ----------------------------

def normalize_fields(fields: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Normalización mínima: solo formatea fecha y limpia espacios.
    No corrige typos ni cambia el texto.
    """
    out = dict(fields)
    if out.get('fecha'):
        nums = re.findall(r"\d+", str(out['fecha']))
        if len(nums) >= 3:
            dd, mm, yy = nums[:3]
            try:
                if len(yy) == 2:
                    yy = ('20' + yy) if int(yy) < 50 else ('19' + yy)
                out['fecha'] = f"{int(dd):02d}-{int(mm):02d}-{yy}"
            except Exception:
                pass
    for k in ('paciente','ciudad','especialista','procedimiento','cliente','institucion','identificacion','remision'):
        if k in out and out[k] is not None:
            s = str(out[k])
            s = re.sub(r"\s+", " ", s).strip(" \\t:.-")
            out[k] = s
    return out

# ----------------------------
# Pipeline principal
# ----------------------------

def process_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    items = ocr_with_paddle(image_bgr)
    kv = extract_kv_from_paddle(items)
    if not any(v for v in kv.values()):
        orange_items = ocr_orange_regions(image_bgr)
        if orange_items:
            kv = extract_kv_from_paddle(orange_items)
            items.extend(orange_items)  # Agregar también los tokens de regiones naranjas
    
    # Extraer datos de la tabla
    table_data = extract_table_data(items)
    
    kv = normalize_fields(kv)
    return {
        'extracted_fields': kv,
        'table_data': table_data,
        'raw_tokens': items,
    }

def process_pdf(path: Path) -> Dict[str, Any]:
    if not HAS_PDF:
        raise RuntimeError("PyMuPDF no instalado.")
    doc = fitz.open(str(path))
    results = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(3,3))
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        res = process_image(img)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=str, help='Ruta a imagen (jpg/png) o PDF')
    ap.add_argument('--out_json', type=str, default=None, help='Guardar salida en JSON')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    if in_path.suffix.lower() in {'.pdf'}:
        payload = process_pdf(in_path)
    else:
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('No se pudo abrir la imagen')
        payload = process_image(img)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()