"""
OCR para Anexo 1: Reporte de Gasto Quirúrgico.
- Extrae cabecera (fecha, paciente, identificación, ciudad, médico, procedimiento, etc.)
- Detecta tabla de insumos
- Detecta sección de trazabilidad (stickers UDI)
- Marca presencia de firmas
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io
import re
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from PIL import ImageEnhance, ImageFilter
import logging
import warnings
warnings.filterwarnings('ignore')

# AI OCR imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR no disponible")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠️ PaddleOCR no disponible")

try:
    from scipy import ndimage
    from skimage import restoration, morphology, filters
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False
    print("⚠️ Procesamiento avanzado no disponible")

# Configurar rutas comunes de Tesseract en Windows
COMMON_TESSERACT_PATHS = [
    r"C:\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe", 
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

# Intentar configurar Tesseract automáticamente
def _setup_tesseract():
    """Configurar Tesseract automáticamente"""
    for path in COMMON_TESSERACT_PATHS:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    return False

# Configurar al importar
_setup_tesseract()

# ----------------------------
# VISIÓN POR COMPUTADORA AVANZADA
# ----------------------------

def detect_orange_fields_with_computer_vision(image_np):
    """Detecta campos naranjas usando visión por computadora avanzada - MEJORADO"""
    print("🔍 Iniciando detección con visión por computadora...")
    
    # Convertir a RGB si es necesario
    if len(image_np.shape) == 3:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    # Convertir a HSV para detectar naranja
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Rangos de naranja BALANCEADOS
    orange_ranges = [
        ([8, 60, 100], [28, 255, 255]),    # Naranja principal
        ([5, 40, 80], [25, 255, 255]),     # Naranja claro
        ([10, 80, 120], [30, 255, 255]),   # Naranja saturado
    ]
    
    # Crear máscara combinada HSV
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in orange_ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    final_mask = combined_mask
    
    # Operaciones morfológicas BALANCEADAS para unir fragmentos de cada campo
    # 1. Cerrar huecos pequeños dentro de cada campo
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # 2. Dilatar moderadamente para unir texto manuscrito de cada campo
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=2)
    
    # 3. Abrir para eliminar ruido pequeño
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    print(f"🎯 Máscara naranja mejorada: {np.sum(final_mask > 0)} píxeles detectados")
    
    return final_mask, image_rgb

def segment_individual_fields(mask, image_rgb):
    """Segmentar campos optimizado para detectar exactamente los 6 campos requeridos"""
    print("✂️ Segmentando campos específicos (Fecha, Paciente, ID, Ciudad, Especialista, Procedimiento)...")
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fields = []
    valid_fields = 0
    
    # Analizar cada contorno con filtros más permisivos
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filtros MENOS ESTRICTOS para debugging - detectar campos grandes
        if area < 1000:  # Área mínima reducida
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Filtros muy permisivos para debugging
        if (aspect_ratio >= 1.0 and aspect_ratio <= 20 and  # Muy amplio
            w >= 80 and h >= 30 and  # Mínimos reducidos
            w <= 800 and h <= 150):  # Máximos amplios
            
            # Expansión con margen más grande para mejor captura
            padding = 12
            x_expanded = max(0, x - padding)
            y_expanded = max(0, y - padding)
            w_expanded = min(image_rgb.shape[1] - x_expanded, w + 2*padding)
            h_expanded = min(image_rgb.shape[0] - y_expanded, h + 2*padding)
            
            # Extraer ROI
            roi = image_rgb[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
            
            if roi.size > 0:
                fields.append({
                    'roi': roi,
                    'bbox': (x_expanded, y_expanded, w_expanded, h_expanded),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center_x': x + w//2,
                    'center_y': y + h//2,
                    'index': i
                })
                valid_fields += 1
                print(f"   📋 Campo {valid_fields}: {w_expanded}x{h_expanded}px, área={area:.0f}, ratio={aspect_ratio:.1f}, pos=({x},{y})")
    
    # Ordenar campos por posición (de arriba a abajo, izquierda a derecha)
    fields.sort(key=lambda f: (f['bbox'][1], f['bbox'][0]))
    
    print(f"✅ Detectados {len(fields)} campos válidos para OCR")
    return fields

def computer_vision_pipeline(image_np):
    """Pipeline completo de visión por computadora para OCR de campos"""
    print("\n🤖 INICIANDO PIPELINE DE VISIÓN POR COMPUTADORA")
    
    # 1. Detectar áreas naranjas
    mask, image_rgb = detect_orange_fields_with_computer_vision(image_np)
    
    # 2. Segmentar campos individuales
    fields = segment_individual_fields(mask, image_rgb)
    
    if not fields:
        print("❌ No se detectaron campos válidos")
        return {}
    
    # 3. OCR en cada campo individual
    field_results = {}
    all_extracted_text = []
    
    for i, field in enumerate(fields):
        field_num = i + 1
        roi = field['roi']
        
        print(f"\n🔬 OCR en campo #{field_num} ({roi.shape[1]}x{roi.shape[0]}px)")
        
        # Usar el OCR ultra agresivo en el ROI específico
        result_text = ultra_aggressive_ocr_engines(roi)
        
        if result_text:
            cleaned_text = intelligent_text_correction(result_text)
            print(f"   📝 Campo #{field_num} resultado: '{cleaned_text}'")
            
            field_results[f'campo_{field_num}'] = {
                'texto': cleaned_text,
                'bbox': field['bbox'],
                'area': field['area'],
                'aspect_ratio': field['aspect_ratio']
            }
            all_extracted_text.append(cleaned_text)
        else:
            print(f"   ❌ Campo #{field_num}: no se extrajo texto")
    
    # 4. Combinar todos los textos para análisis de campos
    combined_text = ' '.join(all_extracted_text)
    print(f"\n📋 TEXTO COMBINADO DE TODOS LOS CAMPOS: '{combined_text}'")
    
    # 5. Extraer campos específicos del texto combinado
    extracted_fields = extract_manuscript_fields(combined_text)
    
    return {
        'individual_fields': field_results,
        'combined_text': combined_text,
        'extracted_fields': extracted_fields,
        'total_fields_detected': len(fields)
    }

# ----------------------------
# Preprocesamiento SÚPER AGRESIVO para campos naranjas
# ----------------------------

def _pil_to_cv(img_pil):
    """Convertir PIL a OpenCV"""
    arr = np.array(img_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _cv_to_pil(img_cv):
    """Convertir OpenCV a PIL"""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def enhance_orange_fields_extreme(img: Image.Image) -> List[Image.Image]:
    """Preprocesamiento extremo específico para campos resaltados en naranja"""
    results = []
    img_cv = _pil_to_cv(img)
    
    # Técnica 1: Detección específica de naranjas con múltiples rangos HSV
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Rangos específicos para diferentes tonos de naranja/amarillo
    orange_ranges = [
        # Naranja claro (como en la imagen)
        (np.array([5, 30, 100]), np.array([25, 255, 255])),
        # Naranja medio
        (np.array([8, 50, 120]), np.array([22, 255, 255])),
        # Amarillo-naranja
        (np.array([15, 40, 150]), np.array([35, 255, 255])),
        # Naranja saturado
        (np.array([3, 100, 100]), np.array([18, 255, 255])),
    ]
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    for i, (lower, upper) in enumerate(orange_ranges):
        # Crear máscara para el color
        mask = cv2.inRange(hsv, lower, upper)
        
        if np.sum(mask) < 50:  # Si no hay suficiente color, saltar
            continue
        
        # Expandir máscara para capturar bordes del texto
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask_expanded = cv2.dilate(mask, kernel, iterations=1)
        
        # Aplicar mejoras específicas en regiones naranjas
        enhanced_region = gray.copy()
        
        # CLAHE súper agresivo en regiones naranjas
        clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(4, 4))
        clahe_applied = clahe.apply(gray)
        enhanced_region[mask_expanded > 0] = clahe_applied[mask_expanded > 0]
        
        # Gamma correction específica para texto en fondo naranja
        for gamma in [0.3, 0.5, 0.7]:
            gamma_corrected = np.power(enhanced_region / 255.0, gamma) * 255
            gamma_corrected = gamma_corrected.astype(np.uint8)
            
            # Realzar el fondo para hacer el texto más visible
            background_mask = cv2.dilate(mask, kernel, iterations=2)
            gamma_corrected[background_mask > 0] = np.maximum(gamma_corrected[background_mask > 0], 180)
            
            # Binarización adaptativa específica
            binary = cv2.adaptiveThreshold(gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 9, 3)
            
            # Limpieza morfológica para manuscritos
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)
            
            results.append(_cv_to_pil(cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2RGB)))
    
    # Técnica 2: Procesamiento por canales específico para naranjas
    b, g, r = cv2.split(img_cv)
    
    # El canal azul suele tener mejor contraste en fondos naranjas
    for channel_name, channel in [('blue', b), ('green', g)]:
        # CLAHE en el canal
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(6, 6))
        channel_enhanced = clahe.apply(channel)
        
        # Inversión si el texto es más oscuro
        inverted = 255 - channel_enhanced
        
        for variant in [channel_enhanced, inverted]:
            # Múltiples binarizaciones
            for block_size, C in [(9, 2), (11, 4), (15, 6)]:
                binary = cv2.adaptiveThreshold(variant, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, block_size, C)
                results.append(_cv_to_pil(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)))
    
    # Técnica 3: Sharpening extremo para manuscritos
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Binarización del sharpened
    binary_sharp = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 5)
    results.append(_cv_to_pil(cv2.cvtColor(binary_sharp, cv2.COLOR_GRAY2RGB)))
    
    return results[:10]  # Limitar a las 10 mejores variantes

def enhance_contrast_extreme(img: Image.Image) -> List[Image.Image]:
    """Mejoras de contraste extremas con PIL"""
    results = []
    
    # Múltiples factores de contraste
    for contrast in [0.5, 1.5, 2.0, 2.5, 3.0]:
        try:
            enhanced = ImageEnhance.Contrast(img).enhance(contrast)
            results.append(enhanced)
        except:
            continue
    
    # Múltiples factores de brillo
    for brightness in [0.6, 0.8, 1.2, 1.4]:
        try:
            enhanced = ImageEnhance.Brightness(img).enhance(brightness)
            results.append(enhanced)
        except:
            continue
    
    # Sharpness
    for sharpness in [1.5, 2.0, 2.5]:
        try:
            enhanced = ImageEnhance.Sharpness(img).enhance(sharpness)
            results.append(enhanced)
        except:
            continue
    
    return results[:8]

# ----------------------------
# Helpers básicos
# ----------------------------
def _pil_from_pixmap(pix) -> Image.Image:
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def _pil_to_cv(img_pil: Image.Image):
    arr = np.array(img_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x0c", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

# ----------------------------
# OCR helpers
# ----------------------------
def enhance_orange_fields_for_manuscripts(img: Image.Image) -> List[Image.Image]:
    """Detección ULTRA PRECISA de campos naranjas manuscritos"""
    variants = []
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 1. MÚLTIPLES RANGOS HSV para máxima cobertura de naranjas
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    
    orange_ranges = [
        ([5, 30, 80], [25, 255, 255]),    # Naranja muy claro
        ([8, 50, 100], [30, 255, 255]),   # Naranja medio
        ([10, 70, 120], [35, 255, 255]),  # Naranja saturado
        ([3, 40, 60], [28, 255, 255]),    # Naranja muy amplio
        ([12, 80, 140], [40, 255, 255]),  # Naranja intenso
    ]
    
    all_masks = []
    for lower, upper in orange_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if np.sum(mask) > 20:  # Si hay algo de naranja
            all_masks.append(mask)
    
    if not all_masks:
        return [img]  # No hay naranjas, usar imagen original
    
    # Combinar todas las máscaras de naranja
    combined_mask = np.zeros_like(all_masks[0])
    for mask in all_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 2. EXPANSIÓN AGRESIVA para capturar texto manuscrito completo
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),   # Expansión fuerte
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), # Expansión suave
        cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)),   # Expansión en cruz
    ]
    
    for kernel in kernels:
        mask_expanded = cv2.dilate(combined_mask, kernel, iterations=3)
        
        # Crear imagen con SOLO campos naranjas
        orange_isolated = np.full_like(cv_img, 255)  # Fondo blanco puro
        orange_isolated[mask_expanded > 0] = cv_img[mask_expanded > 0]
        
        # 3. MÚLTIPLES PROCESAMIENTOS para mejor OCR
        gray = cv2.cvtColor(orange_isolated, cv2.COLOR_BGR2GRAY)
        
        # 3a. Versión binaria directa
        _, binary_direct = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        variants.append(Image.fromarray(cv2.cvtColor(binary_direct, cv2.COLOR_GRAY2RGB)))
        
        # 3b. Versión con CLAHE agresivo
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        clahe_enhanced = clahe.apply(gray)
        _, binary_clahe = cv2.threshold(clahe_enhanced, 160, 255, cv2.THRESH_BINARY_INV)
        variants.append(Image.fromarray(cv2.cvtColor(binary_clahe, cv2.COLOR_GRAY2RGB)))
        
        # 3c. Versión con binarización adaptativa
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        variants.append(Image.fromarray(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)))
    
    # 4. DETECCIÓN POR COLOR RGB directo (para naranjas que HSV no captura)
    # Detectar píxeles naranjas por intensidad RGB
    r, g, b = cv2.split(cv_img)
    
    # Condiciones para naranja: R > G > B y suficiente intensidad
    orange_rgb_mask = (
        (r > g + 15) &  # Más rojo que verde
        (g > b + 10) &  # Más verde que azul (característico naranja)
        (r > 120) &     # Suficiente intensidad de rojo
        (g > 80)        # Suficiente verde para naranja
    ).astype(np.uint8) * 255
    
    if np.sum(orange_rgb_mask) > 50:
        # Expansión del RGB
        kernel_rgb = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        rgb_expanded = cv2.dilate(orange_rgb_mask, kernel_rgb, iterations=2)
        
        # Aplicar máscara RGB
        rgb_isolated = np.full_like(cv_img, 255)
        rgb_isolated[rgb_expanded > 0] = cv_img[rgb_expanded > 0]
        
        gray_rgb = cv2.cvtColor(rgb_isolated, cv2.COLOR_BGR2GRAY)
        _, binary_rgb = cv2.threshold(gray_rgb, 170, 255, cv2.THRESH_BINARY_INV)
        variants.append(Image.fromarray(cv2.cvtColor(binary_rgb, cv2.COLOR_GRAY2RGB)))
    
    return variants[:8]  # Las 8 mejores variantes

def ocr_comparison_easy_tesseract(field_image):
    """OCR OPTIMIZADO: PSM-8-OEM-3 como configuración principal"""
    print("� INICIANDO OCR OPTIMIZADO: PSM-8-OEM-3 Principal")
    
    results = []
    
    # ====================
    # 1. TESSERACT PSM-8-OEM-3 (PRINCIPAL)
    # ====================
    print("🔥 === TESSERACT PSM-8-OEM-3 PRINCIPAL ===")
    
    # Aplicar preprocesamiento BALANCEADO para manuscritos
    try:
        if len(field_image.shape) == 3:
            gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = field_image.copy()
        
        # Preprocesamiento SIMPLIFICADO pero efectivo
        # 1. Suavizado bilateral preservando bordes
        enhanced = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # 2. CLAHE moderado para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)
        
        # 3. Normalización suave
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        # Usar imagen en escala de grises mejorada (NO binarizada)
        final_image = enhanced
        
        # Configuraciones optimizadas - PSM-8-OEM-3 como principal
        tesseract_configs = [
            ('PSM-8-OEM-3-MAIN', '--psm 8 --oem 3', '🎯 CONFIGURACIÓN PRINCIPAL'),
            ('PSM-8-OEM-3-CLEAN', '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789áéíóúñÁÉÍÓÚÑ.: -', 'Con filtro de caracteres'),
            ('PSM-7-OEM-3', '--psm 7 --oem 3', 'Backup línea'),
            ('PSM-6-OEM-3', '--psm 6 --oem 3', 'Backup bloque'),
        ]
        
        # Probar cada configuración con la imagen preprocesada final
        for config_name, config_params, description in tesseract_configs:
            try:
                text = pytesseract.image_to_string(final_image, config=config_params, lang='spa+eng').strip()
                if text and len(text) > 1:
                    score = len(text) + sum(1 for c in text if c.isalnum()) * 2
                    results.append((config_name, text, score, 0.0))
                    print(f"   🔤 {config_name} ({description}): '{text}'")
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"❌ Error en Tesseract: {e}")
    
    # ====================
    # 2. EASYOCR (PRINCIPAL - mejor con manuscritos)
    # ====================
    print("🧠 === EASYOCR PRINCIPAL ===")
    try:
        easyocr_reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
        
        # Solo líneas individuales para EasyOCR con threshold más bajo
        try:
            easy_results = easyocr_reader.readtext(field_image, detail=1, paragraph=False)
            if easy_results:
                text = ' '.join([result[1] for result in easy_results if result[2] > 0.2])
                confidence = sum([result[2] for result in easy_results]) / len(easy_results)
                if text.strip():
                    # SCORE MUCHO MÁS ALTO para EasyOCR (prioridad sobre Tesseract)
                    score = len(text) * 6 + sum(1 for c in text if c.isalnum()) * 4
                    results.append(('EasyOCR-Primary', text, score, confidence))
                    print(f"   📝 EasyOCR Primary (conf:{confidence:.2f}): '{text}'")
        except Exception as e:
            print(f"   ❌ Error EasyOCR: {e}")
            
    except Exception as e:
        print(f"❌ Error inicializando EasyOCR: {e}")
    
    return results

def ultra_aggressive_ocr_engines(field_image):
    """Wrapper que usa la función de comparación EasyOCR vs Tesseract"""
    results = ocr_comparison_easy_tesseract(field_image)
    
    if not results:
        return "NO_TEXT_DETECTED"
    
    # Ordenar por score y tomar el mejor
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Mostrar top resultados
    print(f"\n🏆 TOP 5 MEJORES RESULTADOS:")
    for i, (engine, text, score, conf) in enumerate(results[:5]):
        conf_str = f" conf:{conf:.2f}" if conf > 0 else ""
        print(f"   🥇 #{i+1} {engine}{conf_str} (score:{score}): '{text}'")
    
    best_result = results[0][1]
    print(f"\n💎 RESULTADO FINAL: '{best_result[:50]}...'")
    return best_result

def intelligent_text_correction(text: str) -> str:
    """Post-procesamiento básico SOLO para limpiar caracteres extraños de OCR"""
    
    if not text or len(text.strip()) < 2:
        return text
    
    # SOLO limpiar caracteres extraños y espacios múltiples
    cleaned = re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ.,:]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # SOLO correcciones técnicas de OCR (NO correcciones basadas en valores conocidos)
    basic_ocr_fixes = {
        # Solo caracteres obviamente mal interpretados
        'rn': 'm',  # muy común en OCR
        'ri': 'n',  # muy común en OCR
        'cl': 'd',  # muy común en OCR
    }
    
    # Aplicar SOLO correcciones técnicas básicas
    corrected = cleaned
    for pattern, replacement in basic_ocr_fixes.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    return corrected

def ocr_manuscript_fields(orange_variants: List[Image.Image], lang="spa+eng") -> str:
    """Wrapper para compatibilidad con la función anterior"""
    
    # Convertir la primera variante PIL a numpy
    if orange_variants:
        # Convertir PIL a numpy array
        pil_img = orange_variants[0]
        image_np = np.array(pil_img)
        
        # Usar el nuevo OCR ultra agresivo
        return ultra_aggressive_ocr_engines(image_np)
    else:
        print("⚠️ No hay variantes de imagen para procesar")
        return ""
    """Post-procesamiento básico SOLO para limpiar caracteres extraños de OCR"""
    
    if not text or len(text.strip()) < 2:
        return text
    
    # SOLO limpiar caracteres extraños y espacios múltiples
    cleaned = re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ.,:]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # SOLO correcciones técnicas de OCR (NO correcciones basadas en valores conocidos)
    basic_ocr_fixes = {
        # Solo caracteres obviamente mal interpretados
        'rn': 'm',  # muy común en OCR
        'ri': 'n',  # muy común en OCR
        'cl': 'd',  # muy común en OCR
    }
    
    # Aplicar SOLO correcciones técnicas básicas
    corrected = cleaned
    for pattern, replacement in basic_ocr_fixes.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    # NO aplicar correcciones de nombres, ciudades, ni valores específicos
    # El sistema debe extraer lo que genuinamente interpreta
    
    return corrected

def extract_manuscript_fields(text: str) -> dict:
    """Extracción GENUINA de 6 campos SIN asumir valores específicos conocidos"""
    fields = {
        'fecha': None,
        'paciente': None,
        'identificacion': None,
        'ciudad': None,
        'especialista': None,
        'procedimiento': None
    }
    
    print(f"\n🔍 ANÁLISIS de texto para 6 campos específicos: '{text[:100]}...'")
    
    # Limpiar y preparar texto
    clean_text = ' '.join(text.split())
    print(f"📝 Texto para análisis: '{clean_text[:100]}...'")
    
    # 1. FECHA (patrones generales SIN valores específicos)
    fecha_patterns = [
        r'\b(\d{1,2}[\-\s/]\d{1,2}[\-\s/]\d{2,4})\b',  # Fechas generales
        r'\b(\d{1,2}[\-\s]+\d{1,2}[\-\s]+\d{2,4})\b',  # Con espacios
    ]
    
    for pattern in fecha_patterns:
        match = re.search(pattern, clean_text)
        if match:
            fields['fecha'] = match.group(1).replace(' ', '-')
            print(f"📅 FECHA detectada: {fields['fecha']}")
            break
    
    # 2. IDENTIFICACIÓN (números y letras mezclados - GENÉRICO)
    id_patterns = [
        r'\b(\d{8,12})\b',  # Solo números largos
        r'\b([0-9A-Z]{8,12})\b',  # Alfanumérico largo genérico
        r'\b(\d{7,10})\b',  # Números medianos
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, clean_text)
        if match:
            fields['identificacion'] = match.group(1)
            print(f"🆔 IDENTIFICACIÓN detectada: {fields['identificacion']}")
            break
    
    # 3. PACIENTE (nombres completos GENÉRICOS) 
    paciente_patterns = [
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # Cuatro palabras
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # Tres palabras
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # Dos palabras
    ]
    
    for pattern in paciente_patterns:
        matches = re.findall(pattern, clean_text)
        if matches:
            for match in matches:
                # Evitar confundir con médico (si tiene "Dr" cerca)
                if 'Dr' not in match and 'D.' not in match:
                    fields['paciente'] = match.strip()
                    print(f"👤 PACIENTE detectado: {fields['paciente']}")
                    break
            if fields['paciente']:
                break
    
    # 4. CIUDAD (palabras que podrían ser ciudades - GENÉRICO)
    ciudad_patterns = [
        r'\b([A-Z][a-z]{5,15})\b',  # Palabras capitalizadas medianas
        r'\b([A-Z][a-z]{4,12})\b',  # Palabras capitalizadas cortas
    ]
    
    for pattern in ciudad_patterns:
        matches = re.findall(pattern, clean_text)
        if matches:
            for match in matches:
                # Filtrar palabras que obviamente no son ciudades
                if (len(match) >= 4 and 
                    'dr' not in match.lower() and
                    match not in ['SOAT']):  # Solo filtros muy básicos
                    fields['ciudad'] = match
                    print(f"🏙️ CIUDAD detectada: {fields['ciudad']}")
                    break
            if fields['ciudad']:
                break
    
    # 5. ESPECIALISTA/MÉDICO (títulos médicos GENÉRICOS)
    medico_patterns = [
        r'\b(Dr[A-Z][a-z]+\s+Dr\s+[A-Z][a-z]+)\b',  # Patrón Dr+palabra+Dr+palabra
        r'\b(Dr\.?\s*[A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # Dr + dos palabras
        r'\b(D\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # D. + dos palabras
    ]
    
    for pattern in medico_patterns:
        match = re.search(pattern, clean_text)
        if match:
            fields['especialista'] = match.group(1)
            print(f"👨‍⚕️ ESPECIALISTA detectado: {fields['especialista']}")
            break
    
    # 6. PROCEDIMIENTO (términos médicos GENÉRICOS)
    proc_patterns = [
        r'\b([a-z]{8,}\s+[a-z]{5,})\b',  # Dos palabras largas juntas
        r'\b([A-Za-z]{7,}s[íi]s\s+[a-z]{5,})\b',  # Palabra terminada en -sis + otra palabra
        r'\b([a-z]{6,}sis\s+[a-z]{4,})\b',  # Términos médicos con -sis
    ]
    
    for pattern in proc_patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            fields['procedimiento'] = match.group(1)
            print(f"🔬 PROCEDIMIENTO detectado: {fields['procedimiento']}")
            break
    
    return fields
   


def ocr_image_aggressive(img: Image.Image, lang="spa+eng") -> str:
    """OCR súper agresivo con múltiples estrategias para campos naranjas y manuscritos"""
    
    all_results = []
    
    try:
        # Probar si Tesseract funciona
        test_config = f"--oem 1 --psm 6 -l {lang}"
        pytesseract.image_to_string(img.crop((0, 0, 50, 50)), config=test_config)
        tesseract_available = True
        print("✅ Tesseract disponible - usando OCR real")
    except Exception:
        print("⚠️  Tesseract no disponible" )
      
    # Código original de OCR si Tesseract está disponible
    # Estrategia 1: OCR con imagen original
    try:
        for psm in [6, 7, 8, 13]:  # Múltiples PSM
            cfg = f"--oem 1 --psm {psm} -l {lang}"
            text = pytesseract.image_to_string(img, config=cfg)
            if text.strip():
                all_results.append((normalize_text(text), len(text), "original"))
    except Exception as e:
        print(f"Error OCR original: {e}")
    
    # Estrategia 2: Preprocesamiento para campos naranjas
    try:
        orange_variants = enhance_orange_fields_extreme(img)
        for i, variant in enumerate(orange_variants[:5]):  # Top 5
            try:
                for psm in [7, 8, 6]:  # PSM optimizados para manuscritos
                    cfg = f"--oem 1 --psm {psm} -l {lang}"
                    text = pytesseract.image_to_string(variant, config=cfg)
                    if text.strip():
                        all_results.append((normalize_text(text), len(text), f"orange_{i}_psm{psm}"))
            except Exception:
                continue
    except Exception as e:
        print(f"Error preprocesamiento naranja: {e}")
    
    # Estrategia 3: Mejoras de contraste
    try:
        contrast_variants = enhance_contrast_extreme(img)
        for i, variant in enumerate(contrast_variants[:3]):  # Top 3
            try:
                cfg = f"--oem 1 --psm 6 -l {lang}"
                text = pytesseract.image_to_string(variant, config=cfg)
                if text.strip():
                    all_results.append((normalize_text(text), len(text), f"contrast_{i}"))
            except Exception:
                continue
    except Exception as e:
        print(f"Error mejoras contraste: {e}")
    
    # Estrategia 4: Configuraciones específicas para manuscritos
    try:
        manuscript_configs = [
            "--oem 1 --psm 8 -c edges_use_new_outline_complexity=0",  # Manuscritos
            "--oem 1 --psm 7 -c textord_min_linesize=0.5",  # Líneas pequeñas
            "--oem 1 --psm 6 -c preserve_interword_spaces=1",  # Espacios
        ]
        
        for config in manuscript_configs:
            try:
                full_config = f"{config} -l {lang}"
                text = pytesseract.image_to_string(img, config=full_config)
                if text.strip():
                    all_results.append((normalize_text(text), len(text), f"manuscript_config"))
            except Exception:
                continue
    except Exception as e:
        print(f"Error configuraciones manuscritos: {e}")
    
    # Seleccionar el mejor resultado
    if not all_results:
        return "Error: No se pudo extraer texto"
    
    # Ordenar por longitud (más texto = mejor) y tomar el mejor
    all_results.sort(key=lambda x: x[1], reverse=True)
    best_result = all_results[0]
    
    return best_result[0]

def ocr_image(img: Image.Image, lang="spa+eng", psm=6, whitelist: Optional[str] = None) -> str:
    """OCR con manejo de errores robusto - ahora usa el método agresivo"""
    try:
        # Usar el método agresivo por defecto
        return ocr_image_aggressive(img, lang)
    except Exception as e:
        print(f"Error en OCR agresivo: {e}")
        # Fallback al método simple
        try:
            cfg = f"--oem 1 --psm {psm} -l {lang}"
            if whitelist:
                cfg += f" -c tessedit_char_whitelist={whitelist}"
            text = pytesseract.image_to_string(img, config=cfg)
            return normalize_text(text)
        except Exception as e2:
            print(f"Error en OCR fallback: {e2}")
            return "Error: Tesseract no disponible. Instalar desde: https://github.com/UB-Mannheim/tesseract/wiki"

def ocr_data(img: Image.Image, lang="spa+eng", psm=6) -> dict:
    """OCR data con manejo de errores"""
    try:
        cfg = f"--oem 1 --psm {psm} -l {lang}"
        return pytesseract.image_to_data(img, output_type=Output.DICT, config=cfg)
    except Exception as e:
        print(f"Error en OCR data: {e}")
        return {"text": [], "conf": []}

# ----------------------------
# Extracción de campos cabecera
# ----------------------------
def extract_header_fields(text: str) -> Dict[str, Optional[str]]:
    """Extracción SÚPER AGRESIVA de campos de cabecera con patrones específicos"""
    fields = {
        "fecha": None,
        "paciente": None,
        "identificacion": None,
        "ciudad": None,
        "medico": None,
        "procedimiento": None,
        "institucion": None,
        "cliente": None,
        "remision": None,
    }

    print(f"Texto a procesar: {text[:500]}...")  # Debug

    # FECHA - Patrones múltiples y más permisivos
    fecha_patterns = [
        r"(?:Fecha[:\s]*)?([0-3]?\d)[-/\.\s]*([01]?\d)[-/\.\s]*([12]?\d{2,4})",
        r"([0-3]?\d)[-/\.\s]+([01]?\d)[-/\.\s]+([12]?\d{2,4})",
        r"(\d{1,2})[-/\.\s]+(\d{1,2})[-/\.\s]+(\d{2,4})",
    ]
    
    for pattern in fecha_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            d, mth, y = m.groups()
            # Normalizar año
            y = y if len(y) == 4 else f"20{y}" if int(y) < 50 else f"19{y}"
            try:
                fields["fecha"] = f"{int(d):02d}-{int(mth):02d}-{y}"
                print(f"Fecha detectada: {fields['fecha']}")
                break
            except ValueError:
                continue

    # IDENTIFICACIÓN - Múltiples patrones para números largos
    id_patterns = [
        r"(?:Identificaci[oó]n[:\s]*)?(\d{8,12})",
        r"(?:C[eé]dula[:\s]*)?(\d{8,12})",
        r"(?:CC[:\s]*)?(\d{8,12})",
        r"(?:Documento[:\s]*)?(\d{8,12})",
        r"\b(\d{8,12})\b",  # Cualquier número de 8-12 dígitos
    ]
    
    for pattern in id_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_id = re.sub(r"[^\d]", "", match)
            if 8 <= len(clean_id) <= 12:
                fields["identificacion"] = clean_id
                print(f"Identificación detectada: {fields['identificacion']}")
                break
        if fields["identificacion"]:
            break

    # PACIENTE - Nombres completos más agresivos
    paciente_patterns = [
        r"(?:Paciente[:\s]*)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,4})",
        r"(?:Nombre[:\s]*)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,4})",
        r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)",
        r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)",
    ]
    
    for pattern in paciente_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Filtrar nombres que no sean solo palabras comunes
            words = match.split()
            if (len(words) >= 2 and 
                all(len(w) >= 2 for w in words) and
                not any(w.lower() in ['fecha', 'ciudad', 'doctor', 'procedimiento'] for w in words)):
                fields["paciente"] = match.title()
                print(f"Paciente detectado: {fields['paciente']}")
                break
        if fields["paciente"]:
            break

    # CIUDAD - Patrones más amplios
    ciudad_patterns = [
        r"(?:Ciudad[:\s]*)?([A-Za-záéíóúñ]{3,20})",
        r"(?:Municipio[:\s]*)?([A-Za-záéíóúñ]{3,20})",
        r"\b(Bucaramanga|Bogotá|Medellín|Cali|Barranquilla|Cartagena)\b",
    ]
    
    for pattern in ciudad_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            ciudad_candidate = m.group(1).strip()
            # Filtrar palabras que claramente no son ciudades
            if (len(ciudad_candidate) >= 3 and 
                ciudad_candidate.lower() not in ['fecha', 'paciente', 'doctor', 'procedimiento']):
                fields["ciudad"] = ciudad_candidate.title()
                print(f"Ciudad detectada: {fields['ciudad']}")
                break

    # MÉDICO/ESPECIALISTA - Más agresivo
    medico_patterns = [
        r"(?:Especialista[:\s]*)?(?:Dr\.?\s*)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)",
        r"(?:M[eé]dico[:\s]*)?(?:Dr\.?\s*)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)",
        r"(?:Doctor[:\s]*)?(?:Dr\.?\s*)?([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)",
        r"Dr\.?\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)",
    ]
    
    for pattern in medico_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            medico_name = m.group(1).strip()
            if len(medico_name) > 5:  # Nombre razonable
                fields["medico"] = f"Dr. {medico_name.title()}"
                print(f"Médico detectado: {fields['medico']}")
                break

    # PROCEDIMIENTO - Más específico para términos médicos
    procedimiento_patterns = [
        r"(?:Procedimiento[:\s]*)?([A-Za-záéíóúñ\s]{10,80})",
        r"\b(osteo\w+|artro\w+|endo\w+|\w*sis\s+\w+|cirugía\s+\w+)",
    ]
    
    for pattern in procedimiento_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            proc = m.group(1).strip()
            # Filtrar procedimientos que parezcan válidos
            if len(proc) >= 10 and any(term in proc.lower() for term in ['osteo', 'sis', 'cirugía', 'ectomía']):
                fields["procedimiento"] = proc.title()
                print(f"Procedimiento detectado: {fields['procedimiento']}")
                break

    # INSTITUCIÓN
    inst_patterns = [
        r"(?:Instituci[oó]n[:\s]*)?([A-Z]{2,10})",
        r"\b(HUC|IPS|EPS|HOSPITAL)\b",
    ]
    
    for pattern in inst_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            fields["institucion"] = m.group(1).upper()
            break

    # CLIENTE  
    cliente_patterns = [
        r"(?:Cliente[:\s]*)?([A-Za-z]{3,20})",
    ]
    
    for pattern in cliente_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            fields["cliente"] = m.group(1).title()
            break

    # REMISIÓN
    remision_patterns = [
        r"(?:No\.?\s*Remisi[oó]n[:\s]*)?(\d{1,4})",
        r"(?:Remisi[oó]n[:\s]*)?(\d{1,4})",
        r"\b(\d{3})\b",  # Números de 3 dígitos
    ]
    
    for pattern in remision_patterns:
        m = re.search(pattern, text)
        if m:
            fields["remision"] = m.group(1)
            break

    return fields

# ----------------------------
# Extracción de tabla insumos
# ----------------------------
def extract_table_insumos(text: str) -> List[Dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if re.search(r"\d{4,}-?\d*", line) and re.search(r"\b\d+\b", line):
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 2:
                rows.append({
                    "codigo": parts[0],
                    "descripcion": " ".join(parts[1:-1]) if len(parts) > 2 else parts[1],
                    "cantidad": parts[-1] if parts[-1].isdigit() else "1"
                })
    return rows

# ----------------------------
# Extracción de trazabilidad
# ----------------------------
def extract_trazabilidad(text: str) -> List[Dict[str, str]]:
    trazas = []
    for block in re.findall(r"UDI.*?(?=(UDI|$))", text, flags=re.S):
        udi = re.search(r"UDI.*?(\d{8,})", block)
        lote = re.search(r"Lote[:\- ]?(\w+)", block, re.I)
        venc = re.search(r"(\d{4}-\d{2}-\d{2})", block)
        trazas.append({
            "udi": udi.group(1) if udi else None,
            "lote": lote.group(1) if lote else None,
            "vencimiento": venc.group(1) if venc else None,
        })
    return trazas

# ----------------------------
# Detección de firmas
# ----------------------------
def detect_firmas(img: Image.Image) -> bool:
    img_cv = _pil_to_cv(img)
    h, w = img_cv.shape[:2]
    roi = img_cv[int(h*0.85):, :]  # zona inferior del documento
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ratio = np.sum(th > 0) / th.size
    return ratio > 0.01  # hay trazos suficientes

# ----------------------------
# Core
# ----------------------------
def pdf_to_structured(path: str, lang="spa+eng") -> Dict:
    """
    Extracción con VISIÓN POR COMPUTADORA + OCR ultra agresivo
    Detecta automáticamente campos naranjas individuales y aplica OCR específico a cada uno
    """
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(path)

    doc = fitz.open(str(fpath))
    results = []

    for i, page in enumerate(doc):
        print(f"\n🔍 Procesando página {i+1} con VISIÓN POR COMPUTADORA...")
        
        # FORZAR rasterización a alta resolución para visión por computadora
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom para mejor detección de contornos
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convertir a numpy array para visión por computadora
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        print(f"   📐 Imagen generada: {image.shape[1]}x{image.shape[0]} píxeles")
        
        # USAR VISIÓN POR COMPUTADORA para detectar y procesar campos
        print("   🤖 Aplicando visión por computadora...")
        cv_results = computer_vision_pipeline(image)
        
        if cv_results and cv_results.get('extracted_fields'):
            page_result = {
                'page': i + 1,
                'method': 'computer_vision_ocr',
                'extracted_fields': cv_results['extracted_fields'],
                'individual_fields': cv_results.get('individual_fields', {}),
                'total_fields_detected': cv_results.get('total_fields_detected', 0),
                'combined_text': cv_results.get('combined_text', '')
            }
            
            # Mostrar resumen de campos extraídos
            extracted = cv_results['extracted_fields']
            found_fields = [k for k, v in extracted.items() if v is not None]
            print(f"\n✅ CAMPOS EXTRAÍDOS CON VISIÓN POR COMPUTADORA: {len(found_fields)}/{len(extracted)}")
            
            for field_name, value in extracted.items():
                if value is not None:
                    print(f"   ✓ {field_name}: {value}")
            
            results.append(page_result)
        else:
            print("❌ Visión por computadora no extrajo campos válidos")
            # Fallback al método tradicional
            print("   🔄 Usando método tradicional como fallback...")
            
            # Tu implementación anterior como fallback
            img = _pil_from_pixmap(pix)
            orange_variants = enhance_orange_fields_for_manuscripts(img)
            
            if orange_variants:
                manuscript_text = ocr_manuscript_fields(orange_variants)
                if manuscript_text:
                    extracted_fields = extract_manuscript_fields(manuscript_text)
                    page_result = {
                        'page': i + 1,
                        'method': 'fallback_traditional_ocr',
                        'extracted_fields': extracted_fields,
                        'raw_text': manuscript_text
                    }
                    results.append(page_result)
                else:
                    results.append({'page': i + 1, 'error': 'No text extracted'})
            else:
                results.append({'page': i + 1, 'error': 'No orange fields detected'})

    doc.close()
    
    # Combinar resultados de todas las páginas
    if len(results) == 1:
        return results[0]
    else:
        return {"pages": results}
        