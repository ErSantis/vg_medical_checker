#!/usr/bin/env python3
"""Debug script para entender la estructura de líneas y tokens"""

import cv2
import json
import numpy as np
from pathlib import Path
from ocr_extractor_paddle_full import ocr_with_paddle, group_lines, is_label_token

# Cargar la primera página del PDF como imagen
import fitz
doc = fitz.open("anexo1.pdf")
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(3,3))
img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

# Extraer tokens
import numpy as np
items = ocr_with_paddle(img)
lines = group_lines(items)

print("=== ANÁLISIS DE LÍNEAS ===")
for li, line in enumerate(lines):
    line_texts = [t['text'] for t in line]
    print(f"Línea {li}: {line_texts}")
    
    # Buscar líneas que contengan etiquetas específicas
    for idx, tok in enumerate(line):
        field = is_label_token(tok['text'])
        if field in ['paciente', 'especialista', 'identificacion']:
            print(f"  -> ETIQUETA ENCONTRADA: '{tok['text']}' -> campo '{field}' en posición {idx}")
            print(f"     Tokens a la derecha en la misma línea: {[t['text'] for t in line[idx+1:]]}")
            if li + 1 < len(lines):
                print(f"     Tokens en la línea siguiente: {[t['text'] for t in lines[li+1]]}")
            if li + 2 < len(lines):
                print(f"     Tokens en la línea +2: {[t['text'] for t in lines[li+2]]}")

print("\n=== TOKENS ESPECÍFICOS ===")
for item in items:
    text = item['text']
    if any(word in text.lower() for word in ['andrea', 'cecilia', 'dr', 'florentino', 'rosas', 'especialista', 'paciente']):
        print(f"Token relevante: '{text}' (score: {item['score']:.3f}, y_top: {item['y_top']:.0f})")