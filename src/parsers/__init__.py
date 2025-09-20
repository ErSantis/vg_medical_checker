import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Tuple, List

from rapidfuzz import process, fuzz


# ===============================
# Modelos de salida
# ===============================

@dataclass
class FieldValue:
    value: Optional[str]
    confidence: Optional[float]  # 1.0 exacto, 0.8 fuzzy>=90, 0.6 fuzzy>=85, None si no encontrado
    source: Optional[str]        # "label", "fuzzy:<etiqueta>", "regex_scan_date", None


@dataclass
class BasicFieldsDetailed:
    paciente: FieldValue
    identificacion: FieldValue
    fecha: FieldValue
    ciudad: FieldValue
    medico: FieldValue
    procedimiento: FieldValue

    def to_basic_dict(self) -> Dict[str, Optional[str]]:
        """Solo los valores (para capas posteriores)."""
        return {
            "paciente": self.paciente.value,
            "identificacion": self.identificacion.value,
            "fecha": self.fecha.value,
            "ciudad": self.ciudad.value,
            "medico": self.medico.value,
            "procedimiento": self.procedimiento.value,
        }


# ===============================
# Config / labels
# ===============================

LABELS: Dict[str, List[str]] = {
    "paciente": ["paciente", "nombre del paciente", "nombre paciente"],
    "identificacion": ["identificación", "identificacion", "cedula", "cédula", "cc", "id", "documento"],
    "fecha": ["fecha", "fecha del procedimiento", "f. cirugia", "f. cirugía"],
    "ciudad": ["ciudad", "municipio", "lugar"],
    "medico": ["médico", "medico", "doctor", "dr", "cirujano"],
    "procedimiento": [
        "procedimiento", "cirugia", "cirugía", "dx/procedimiento",
        "descripcion del procedimiento", "descripción del procedimiento"
    ],
}

SEPARATORS_REGEX = r"[:\-–—]"  # :, -, –, —


# ===============================
# Utilidades de normalización
# ===============================

def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def _norm(s: str) -> str:
    s2 = _strip_accents(s)
    s2 = re.sub(r"[ \t]+", " ", s2)
    return s2.lower().strip()

def _clean_value(val: str) -> str:
    v = val.strip()
    v = v.splitlines()[0].strip()       # corta en fin de línea
    v = re.sub(r"[|•·]+$", "", v).strip()
    v = re.sub(r"^[\s:.\-–—]+", "", v)  # separadores al inicio
    return v


# ===============================
# Extracción por línea
# ===============================

def _try_exact_label(line: str, label_variants: List[str]) -> Optional[str]:
    """Busca 'Etiqueta <sep> valor' de forma exacta (case-insensitive)."""
    for lab in label_variants:
        m = re.search(rf"(?i)\b{re.escape(lab)}\b\s*{SEPARATORS_REGEX}\s*(.+)$", line.strip())
        if m:
            return _clean_value(m.group(1))
    return None

def _try_fuzzy_label(line: str, label_variants: List[str], threshold: int = 85) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Si no hay match exacto, intenta fuzzy:
    - Compara la parte izquierda de la línea con las etiquetas.
    - Si supera threshold, devuelve el valor a la derecha.
    """
    parts = re.split(SEPARATORS_REGEX, line, maxsplit=1)
    left = parts[0]
    best = process.extractOne(left, label_variants, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= threshold and len(parts) == 2:
        return _clean_value(parts[1]), int(best[1]), str(best[0])
    return None, None, None


# ===============================
# Heurísticas por campo
# ===============================

def _normalize_ident(v: Optional[str]) -> Optional[str]:
    if not v:
        return v
    v = v.strip()
    # conservar dígitos, letras y guiones (por si hay NIT/ID con letra)
    v = re.sub(r"[^\dA-Za-z-]", "", v)
    return v or None

def _normalize_medico(v: Optional[str]) -> Optional[str]:
    if not v:
        return v
    v = re.sub(r"^\s*(dr\.?|doctor)\s+", "", v, flags=re.IGNORECASE).strip()
    # capitalización simple
    return " ".join(w.capitalize() for w in v.split()) or None

def _extract_date_iso_any(text: str) -> Optional[str]:
    """Busca fechas comunes en todo el texto; devuelve ISO si posible."""
    if not text:
        return None
    # ISO: yyyy-mm-dd
    m_iso = re.search(r"\b(20\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b", text)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"
    # dd-mm-yyyy / dd/mm/yyyy / dd.mm.yyyy  (también d-m-yy)
    m = re.search(r"\b(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.]((?:20)?\d{2})\b", text)
    if m:
        d, mm, yy = m.group(1), m.group(2), m.group(3)
        if len(yy) == 2:
            yy = f"20{yy}"
        try:
            dt = datetime(int(yy), int(mm), int(d))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


# ===============================
# Extracción principal
# ===============================

def parse_basic_fields(text: str) -> BasicFieldsDetailed:
    """
    Extrae los 6 datos básicos con dos intentos por campo:
      1) Búsqueda exacta 'Etiqueta: Valor'
      2) Fallback fuzzy (rapidfuzz) si el OCR deformó la etiqueta
    Devuelve valor + confianza + fuente.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result: Dict[str, FieldValue] = {}

    def extract_field(field_name: str, post_clean=None, long_value=False) -> FieldValue:
        labels = LABELS[field_name]

        # 1) Intento exacto
        for ln in lines:
            exact = _try_exact_label(ln, labels)
            if exact:
                val = exact if not long_value else exact
                if post_clean:
                    val = post_clean(val)
                return FieldValue(val, 1.0, "label")

        # 2) Fallback fuzzy
        best_val, best_score, best_lab = None, None, None
        for ln in lines:
            val, score, lab = _try_fuzzy_label(ln, labels, threshold=85)
            if val is not None and (best_score is None or score > best_score):
                best_val, best_score, best_lab = val, score, lab

        if best_val is not None:
            val = post_clean(best_val) if post_clean else best_val
            conf = 0.8 if best_score >= 90 else 0.6
            return FieldValue(val, conf, f"fuzzy:{best_lab}")

        return FieldValue(None, None, None)

    # Campos con normalizaciones específicas
    result["paciente"] = extract_field("paciente")
    result["identificacion"] = extract_field("identificacion", post_clean=_normalize_ident)
    result["ciudad"] = extract_field("ciudad")
    result["medico"] = extract_field("medico", post_clean=_normalize_medico)
    result["procedimiento"] = extract_field("procedimiento", long_value=True)

    # Fecha: primero por etiqueta/fuzzy; si no, escaneo global regex
    date_field = extract_field("fecha")
    if date_field.value:
        iso = _extract_date_iso_any(date_field.value)
        if iso:
            result["fecha"] = FieldValue(iso, date_field.confidence, f"{date_field.source}+regex")
        else:
            iso2 = _extract_date_iso_any(text)
            if iso2:
                result["fecha"] = FieldValue(iso2, 1.0, "regex_scan_date")
            else:
                result["fecha"] = date_field
    else:
        iso2 = _extract_date_iso_any(text)
        result["fecha"] = FieldValue(iso2, 1.0 if iso2 else None, "regex_scan_date" if iso2 else None)

    return BasicFieldsDetailed(
        paciente=result["paciente"],
        identificacion=result["identificacion"],
        fecha=result["fecha"],
        ciudad=result["ciudad"],
        medico=result["medico"],
        procedimiento=result["procedimiento"],
    )
