# src/vgmedical_checker/ocr/ocr_pipeline.py
"""
OCR para formularios médicos (VG Medical)
- PyMuPDF (texto embebido) + raster + Tesseract
- Preprocesados para manuscritos y fondos coloreados
- Extracción por LAYOUT: detecta etiquetas (Fecha, Paciente, Identificación, Ciudad, Especialista, Procedimiento)
  y toma el texto a la derecha en la misma línea
- Re-OCR dirigido de ROI para números/fechas (whitelist)
- Fallback por regex genéricas
- CLI: python -m vgmedical_checker.ocr.ocr_pipeline path/al/archivo.(pdf|png|jpg) --lang spa+eng
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io
import os
import re
import difflib
import argparse

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import pytesseract
from pytesseract import Output

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

COMMON_TESSERACT_PATHS = [
    r"C:\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

CONFIDENCE_THRESHOLD = 35.0
MIN_TEXT_LENGTH = 2

DEFAULT_PSM_LIST_AGGR = [6, 8, 13, 7, 3]
DEFAULT_PSM_LIST_HANDWRITING = [8, 6, 13, 7]

LABELS_CANON = {
    "fecha": ["fecha"],
    "paciente": ["paciente", "nombre", "nombre paciente"],
    "identificacion": ["identificacion", "identificación", "cedula", "cédula", "no ident", "nro ident", "documento"],
    "ciudad": ["ciudad", "municipio"],
    "medico": ["especialista", "médico", "doctor", "dr", "dra"],
    "procedimiento": ["procedimiento"],
    "institucion": ["institucion", "institución"],
    "cliente": ["cliente"],
    "remision": ["no", "remision", "remisión", "no remisión", "no. remisión", "nro remision"],
}

# ---------------------------------------------------------------------
# Helpers Tesseract / Paths
# ---------------------------------------------------------------------
def _ensure_tesseract(tesseract_cmd: Optional[str], tessdata_prefix: Optional[str]):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        for p in COMMON_TESSERACT_PATHS:
            if Path(p).exists():
                pytesseract.pytesseract.tesseract_cmd = p
                break
    if tessdata_prefix:
        os.environ["TESSDATA_PREFIX"] = tessdata_prefix

# ---------------------------------------------------------------------
# PIL <-> OpenCV conversions and pixmap
# ---------------------------------------------------------------------
def _pil_from_pixmap(pix) -> Image.Image:
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def _pil_to_cv(img_pil: Image.Image):
    arr = np.array(img_pil)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# ---------------------------------------------------------------------
# Preprocesados (selección breve y efectiva)
# ---------------------------------------------------------------------
def enhance_handwriting_sauvola(img_pil: Image.Image) -> Image.Image:
    img_cv = _pil_to_cv(img_pil)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    adaptive = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize=21, C=10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    return Image.fromarray(cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB))

def enhance_handwriting_contrast(img_pil: Image.Image) -> Image.Image:
    img_cv = _pil_to_cv(img_pil)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    contrast_enhanced = clahe.apply(gray)
    gamma_corrected = np.power(contrast_enhanced / 255.0, 0.7) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    adaptive = cv2.adaptiveThreshold(gamma_corrected, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize=11, C=15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
    return Image.fromarray(cv2.cvtColor(connected, cv2.COLOR_GRAY2RGB))

def enhance_colored_background_removal(img_pil: Image.Image) -> Image.Image:
    """Optimizado para campos naranja/amarillos (formularios)."""
    img_cv = _pil_to_cv(img_pil)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 20, 100])
    upper_orange = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    result = gray.copy()
    if np.any(mask):
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gamma_corrected = np.power(enhanced / 255.0, 0.6) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)
        result[mask > 0] = gamma_corrected[mask > 0]
        bg = cv2.dilate(mask, kernel, iterations=1)
        result[bg > 0] = np.maximum(result[bg > 0], 200)
    adaptive = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize=11, C=8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel2)
    return Image.fromarray(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB))

def enhance_form_fields_detection(img_pil: Image.Image) -> Image.Image:
    img_cv = _pil_to_cv(img_pil)
    b, g, _ = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    blue_enhanced = clahe.apply(b)
    gray_combined = cv2.addWeighted(blue_enhanced, 0.6, g, 0.4, 0)
    bilateral = cv2.bilateralFilter(gray_combined, d=9, sigmaColor=75, sigmaSpace=75)
    adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize=17, C=7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    return Image.fromarray(cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB))

# ---------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------
def normalize_ocr_text(s: str) -> str:
    if not s:
        return s
    repl = {"¦": "|", "—": "-", "–": "-", "´": "'", "‘": "'", "’": "'",
            "“": '"', "”": '"', "ﬁ": "fi", "ﬂ": "fl", "\x0c": "", "\u200b": ""}
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"[ \t]+", " ", s).strip()
    s = re.sub(r"(?<=\d)O(?=\d)", "0", s)
    s = re.sub(r"(?<=\D)I(?=\d)", "1", s)
    s = s.replace("||", "|")
    return s

def avg_confidence(img_pil: Image.Image, config: str) -> float:
    try:
        data = pytesseract.image_to_data(img_pil, output_type=Output.DICT, config=config)
    except Exception:
        return 0.0
    confs = []
    for c in data.get("conf", []):
        try:
            v = float(c)
            if v >= 0:
                confs.append(v)
        except Exception:
            continue
    if not confs:
        return 0.0
    return float(sum(confs)) / len(confs)

def ocr_try_multiple_psm(img_pil: Image.Image, lang: str, psm_list: List[int], base_oem: int = 1):
    best_text = ""
    best_conf = -1.0
    for psm in psm_list:
        cfg = f"--oem {base_oem} --psm {psm} -l {lang}"
        try:
            text = pytesseract.image_to_string(img_pil, config=cfg)
        except Exception:
            text = ""
        conf = avg_confidence(img_pil, config=cfg)
        if conf > best_conf:
            best_conf = conf
            best_text = text
    return normalize_ocr_text(best_text), best_conf

# ---------------------------------------------------------------------
# Extracción por Layout
# ---------------------------------------------------------------------
def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _is_label_token(tok: str, field: str, thr=0.74) -> bool:
    for cand in LABELS_CANON[field]:
        if _similar(tok, cand) >= thr:
            return True
    return False

def _ocr_words_with_boxes(img_pil: Image.Image, lang="spa+eng"):
    cfg = f"--oem 1 --psm 6 -l {lang}"
    data = pytesseract.image_to_data(img_pil, output_type=Output.DICT, config=cfg)
    words = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = float(data["conf"][i]) if str(data["conf"][i]).replace('.', '', 1).isdigit() else -1
        if txt and conf >= 0:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append({"text": txt, "conf": conf, "x": x, "y": y, "w": w, "h": h, "cx": x + w/2, "cy": y + h/2})
    words.sort(key=lambda t: (t["cy"], t["x"]))
    return words

def _collect_right_of_label(line_words, label_idx, next_label_x=None):
    val_tokens = []
    label_x_end = line_words[label_idx]["x"] + line_words[label_idx]["w"]
    for j in range(label_idx + 1, len(line_words)):
        wj = line_words[j]
        if next_label_x is not None and wj["x"] >= next_label_x:
            break
        if wj["x"] >= label_x_end - 2:
            val_tokens.append(wj["text"])
    return _norm(" ".join(val_tokens))

def extract_fields_by_layout(img_pil: Image.Image, lang="spa+eng") -> Dict[str, Optional[str]]:
    words = _ocr_words_with_boxes(img_pil, lang=lang)
    if not words:
        return {}

    # agrupar por línea (tolerancia vertical)
    lines: List[List[dict]] = []
    current = []
    last_cy = None
    for w in words:
        if last_cy is None or abs(w["cy"] - last_cy) <= 8:
            current.append(w)
            last_cy = w["cy"] if last_cy is None else (last_cy*0.7 + w["cy"]*0.3)
        else:
            lines.append(sorted(current, key=lambda t: t["x"]))
            current = [w]
            last_cy = w["cy"]
    if current:
        lines.append(sorted(current, key=lambda t: t["x"]))

    fields = {k: None for k in LABELS_CANON.keys()}

    for line in lines:
        label_hits = []
        for i, tok in enumerate(line):
            tnorm = tok["text"].lower().rstrip(":")
            for field in LABELS_CANON.keys():
                if _is_label_token(tnorm, field):
                    label_hits.append((i, field))
        if not label_hits:
            continue
        label_hits.sort(key=lambda x: line[x[0]]["x"])
        for k, (idx, fname) in enumerate(label_hits):
            next_x = line[label_hits[k+1][0]]["x"] if k+1 < len(label_hits) else None
            candidate = _collect_right_of_label(line, idx, next_x)

            if fname == "identificacion":
                cand_num = re.sub(r"[^\d]", "", candidate)
                if len(cand_num) < 7:
                    x = line[idx]["x"] + line[idx]["w"]
                    y = max(0, line[idx]["y"] - 4)
                    w = (next_x - x) if next_x else 420
                    h = max(line[idx]["h"] + 8, 36)
                    roi = img_pil.crop((x, y, x + w, y + h))
                    cfg = f"--oem 1 --psm 7 -l {lang} -c tessedit_char_whitelist=0123456789"
                    cand2 = pytesseract.image_to_string(roi, config=cfg)
                    cand2 = re.sub(r"[^\d]", "", cand2 or "")
                    if len(cand2) > len(cand_num):
                        cand_num = cand2
                candidate = cand_num

            if fname == "fecha":
                m = re.search(r"([0-3]?\d)[\-/\. ]+([01]?\d)[\-/\. ]+([12]?\d{1,3})", candidate)
                if m:
                    d, mth, y = m.groups()
                    y = ("20" + y) if len(y) == 2 else y
                    candidate = f"{int(d):02d}-{int(mth):02d}-{int(y)%100:02d}"

            if candidate and (fields.get(fname) is None or len(candidate) > len(fields[fname] or "")):
                fields[fname] = candidate

    # post-normalización
    if fields.get("paciente"):
        fields["paciente"] = " ".join(s.capitalize() for s in re.split(r"\s+", fields["paciente"]))
    if fields.get("medico"):
        mtxt = fields["medico"]
        if not mtxt.lower().startswith(("dr", "dra")):
            mtxt = "Dr. " + mtxt
        fields["medico"] = " ".join(w.capitalize() for w in mtxt.split())
    if fields.get("procedimiento"):
        fields["procedimiento"] = _norm(fields["procedimiento"])
    if fields.get("ciudad"):
        fields["ciudad"] = fields["ciudad"].title()
    return fields

# ---------------------------------------------------------------------
# Fallback por texto plano (regex genéricas, sin datos hardcodeados)
# ---------------------------------------------------------------------
def extract_fields_from_text(full_text: str) -> Dict[str, Optional[str]]:
    text = " " + re.sub(r"\s+", " ", full_text) + " "
    out = {
        "fecha": None, "paciente": None, "identificacion": None,
        "ciudad": None, "medico": None, "procedimiento": None,
        "institucion": None, "cliente": None, "remision": None
    }

    # Fecha
    m = re.search(r"(?:fecha[:\s]*)?([0-3]?\d)[\-/\. ]+([01]?\d)[\-/\. ]+([12]?\d{1,3})", text, re.I)
    if m:
        d, mth, y = m.groups()
        y = ("20" + y) if len(y) == 2 else y
        out["fecha"] = f"{int(d):02d}-{int(mth):02d}-{int(y)%100:02d}"

    # Identificación
    m = re.search(r"(?:identificaci[óo]n|cedula|c[eé]dula|cc|documento)\s*[:\-]?\s*([\d\.\s]{7,})", text, re.I)
    if m:
        out["identificacion"] = re.sub(r"[^\d]", "", m.group(1))
    else:
        m = re.search(r"\b(\d{8,12})\b", text)
        if m:
            out["identificacion"] = m.group(1)

    # Paciente
    m = re.search(r"(?:paciente|nombre(?:\s+paciente)?)\s*[:\-]\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ\s\.]{4,80})", text, re.I)
    if m:
        cand = m.group(1).strip()
        out["paciente"] = " ".join(w.capitalize() for w in cand.split())

    # Ciudad
    m = re.search(r"(?:ciudad|municipio)\s*[:\-]\s*([A-Za-zÁÉÍÓÚáéíóúñ\s]{3,40})", text, re.I)
    if m:
        out["ciudad"] = m.group(1).strip().title()

    # Médico
    m = re.search(r"(?:especialista|m[eé]dico|doctor)\s*[:\-]\s*(dr\.?\s*)?([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñ]+)", text, re.I)
    if m:
        out["medico"] = "Dr. " + " ".join(w.capitalize() for w in m.group(2).split())

    # Procedimiento
    m = re.search(r"(?:procedimiento|cirug[íi]a|intervenci[óo]n)\s*[:\-]\s*([A-Za-zÁÉÍÓÚáéíóúñ\s]{5,80})", text, re.I)
    if m:
        out["procedimiento"] = m.group(1).strip()

    # Institución / Cliente / Remisión (opcionales)
    m = re.search(r"(?:instituci[oó]n)\s*[:\-]\s*([A-Za-z0-9\-\s]{2,60})", text, re.I)
    if m: out["institucion"] = m.group(1).strip()
    m = re.search(r"(?:cliente)\s*[:\-]\s*([A-Za-z0-9\-\s]{2,60})", text, re.I)
    if m: out["cliente"] = m.group(1).strip()
    m = re.search(r"(?:remisi[oó]n|no\.?\s*remisi[oó]n|nro\s*remisi[oó]n)\s*[:\-]?\s*([A-Za-z0-9\-]{1,20})", text, re.I)
    if m: out["remision"] = m.group(1).strip()

    return out

# ---------------------------------------------------------------------
# Re-OCR header rápido (opcional)
# ---------------------------------------------------------------------
def header_reocr(img: Image.Image, lang="spa+eng") -> str:
    w, h = img.size
    header_h = int(max(120, h * 0.18))
    header = img.crop((0, 0, w, header_h)).resize((w*2, header_h*2), Image.BICUBIC)
    texts = []
    for proc in (enhance_handwriting_contrast, enhance_handwriting_sauvola, enhance_form_fields_detection):
        t, _ = ocr_try_multiple_psm(proc(header), lang=lang, psm_list=DEFAULT_PSM_LIST_HANDWRITING)
        texts.append(t)
    return normalize_ocr_text("\n".join(texts))

# ---------------------------------------------------------------------
# Core: PDF/Image -> texto + campos
# ---------------------------------------------------------------------
def pdf_or_image_to_text_and_fields(
    path: str,
    lang: str = "spa+eng",
    tesseract_cmd: Optional[str] = None,
    tessdata_prefix: Optional[str] = None,
    fast_mode: bool = True
) -> Dict:
    _ensure_tesseract(tesseract_cmd, tessdata_prefix)
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"No existe: {fpath}")

    pages_info: List[Dict] = []
    full_text_chunks: List[str] = []

    def _process_pil(img: Image.Image, page_idx: int):
        # estrategia de página
        strategies = [
            ("colored_bg_removal", enhance_colored_background_removal(img)),
            ("form_fields", enhance_form_fields_detection(img)),
            ("sauvola", enhance_handwriting_sauvola(img)),
            ("contrast", enhance_handwriting_contrast(img)),
            ("original", img),
        ] if not fast_mode else [
            ("colored_bg_removal", enhance_colored_background_removal(img)),
            ("original", img),
        ]
        best_text, best_conf, best_mode = "", -1.0, None
        for name, pim in strategies:
            psm = DEFAULT_PSM_LIST_HANDWRITING if name in ("sauvola", "contrast") else [6, 7]
            txt, conf = ocr_try_multiple_psm(pim, lang=lang, psm_list=psm)
            if conf > best_conf and len((txt or "").strip()) >= MIN_TEXT_LENGTH:
                best_text, best_conf, best_mode = txt, conf, name
            if conf >= 50 and len((txt or "")) > 20:
                break

        # layout fields de la página
        layout_fields = extract_fields_by_layout(img, lang=lang)

        pages_info.append({
            "page": page_idx + 1,
            "text": best_text.strip(),
            "avg_conf": float(best_conf),
            "mode": best_mode or "unknown",
            "layout_fields": layout_fields,
        })
        full_text_chunks.append(best_text)

    # PDF o imagen suelta
    if fpath.suffix.lower() in (".pdf",):
        doc = fitz.open(str(fpath))
        for i, page in enumerate(doc):
            # embedded
            try:
                p_text = normalize_ocr_text(page.get_text())
            except Exception:
                p_text = ""
            if p_text and len(p_text) > 5:
                # aún así rasterizamos para layout
                mat = fitz.Matrix(3.0, 3.0)
                pix = page.get_pixmap(matrix=mat)
                img = _pil_from_pixmap(pix)
                _process_pil(img, i)
                # combinar texto embebido
                pages_info[-1]["text"] = p_text
                full_text_chunks[-1] = p_text
            else:
                mat = fitz.Matrix(3.0, 3.0)
                pix = page.get_pixmap(matrix=mat)
                img = _pil_from_pixmap(pix)
                _process_pil(img, i)
        doc.close()
    else:
        # imagen
        img = Image.open(fpath).convert("RGB")
        _process_pil(img, 0)

    full_text = "\n\n".join([f"--- Página {p['page']} ({p.get('mode','')}) ---\n{p['text']}" for p in pages_info]).strip()

    # merge de campos: preferir layout; complementar con regex del texto total + header re-OCR
    merged_fields = {k: None for k in LABELS_CANON.keys()}
    # 1) layout (por página en orden)
    for p in pages_info:
        lf = p.get("layout_fields") or {}
        for k, v in lf.items():
            if v and not merged_fields.get(k):
                merged_fields[k] = v
    # 2) header re-OCR como apoyo
    try:
        if fpath.suffix.lower() == ".pdf":
            # primer header del primer raster
            pass  # ya cubierto por layout por página
        else:
            img = Image.open(fpath).convert("RGB")
            htxt = header_reocr(img, lang=lang)
            h_fields = extract_fields_from_text(htxt)
            for k, v in h_fields.items():
                if v and not merged_fields.get(k):
                    merged_fields[k] = v
    except Exception:
        pass
    # 3) regex global de texto
    t_fields = extract_fields_from_text("\n".join(full_text_chunks))
    for k, v in t_fields.items():
        if v and not merged_fields.get(k):
            merged_fields[k] = v

    # salida compacta (solo claves principales)
    final_fields = {
        "fecha": merged_fields.get("fecha"),
        "paciente": merged_fields.get("paciente"),
        "identificacion": merged_fields.get("identificacion"),
        "ciudad": merged_fields.get("ciudad"),
        "medico": merged_fields.get("medico"),
        "procedimiento": merged_fields.get("procedimiento"),
        # opcionales útiles
        "institucion": merged_fields.get("institucion"),
        "cliente": merged_fields.get("cliente"),
        "remision": merged_fields.get("remision"),
    }

    return {"full_text": full_text, "fields": final_fields, "pages_info": pages_info}

# ---------------------------------------------------------------------
# CLI para pruebas rápidas
# ---------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description="OCR VG Medical - extracción de campos clave")
    ap.add_argument("input_path", help="Ruta al PDF o imagen")
    ap.add_argument("--lang", default="spa+eng", help="Lenguajes Tesseract (p.ej. spa, spa+eng)")
    ap.add_argument("--tesseract_cmd", default=None, help="Ruta al binario de tesseract")
    ap.add_argument("--tessdata_prefix", default=None, help="Ruta TESSDATA_PREFIX")
    ap.add_argument("--fast", action="store_true", help="Modo rápido")
    args = ap.parse_args()

    res = pdf_or_image_to_text_and_fields(
        args.input_path,
        lang=args.lang,
        tesseract_cmd=args.tesseract_cmd,
        tessdata_prefix=args.tessdata_prefix,
        fast_mode=args.fast
    )
    import json
    print(json.dumps(res["fields"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _cli()
