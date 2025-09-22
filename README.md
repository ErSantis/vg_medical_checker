# VG MedicalEl sistema compara automáticamente los campos básicos entre los tres documentos y verifica la consistencia de la tabla de insumos entre los anexos, proporcionando reportes claros sobre discrepancias y recomendaciones de revisión.

## ⚡ Inicio Rápido

```bash
# 1. Clonar repositorio
git clone <url-del-repositorio>
cd vg_medical_checker

# 2. Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. 🚀 EJECUTAR TODO EL SISTEMA
python tests/test_document_matcher.py
```

¡Eso es todo! El comando anterior procesará automáticamente los 3 anexos y generará reportes completos.

## 🏗️ Arquitectura de la Soluciónecker - Sistema de Verificación de Documentos Médicos

## 📋 Descripción General

VG Medical Checker es un sistema automatizado de verificación de documentos médicos que utiliza tecnología OCR (Reconocimiento Óptico de Caracteres) para extraer y comparar información de tres tipos de anexos médicos:

- **Anexo 1**: REPORTE DE GASTO QUIRÚRGICO (INTERNO)

- **Anexo 2**: REPORTE DE GASTO QUIRÚRGICO (HOSPITAL)

- **Anexo 3**: DESCRIPCIÓN QUIRÚRGICA (DOCTOR)

El sistema compara automáticamente los campos básicos entre los tres documentos y verifica la consistencia de la tabla de insumos entre los anexos, proporcionando reportes claros sobre discrepancias y recomendaciones de revisión.

## 🏗️ Arquitectura de la Solución

```
VG Medical Checker/
├── src/
│   ├── ocr/                    # Extractores OCR por tipo de anexo
│   │   ├── ocr_extractor_anexo1.py
│   │   ├── ocr_extractor_anexo2.py
│   │   └── ocr_extractor_anexo3.py
│   ├── matcher/                # Motor de comparación
│   │   └── document_matcher.py
│   ├── parsers/               # Parsers de campos específicos
│   │   └── basic_fields.py
│   └── db/                    # Base de datos de sinónimos
│       └── synonyms.json
├── tests/                     # Pruebas del sistema
├── output/                    # Archivos de salida
└── requirements.txt           # Dependencias Python
```

## 🧠 Enfoque Técnico

### 1. Extracción OCR Especializada

**Tecnología**: PaddlePaddle OCR con modelos preentrenados en español/latín
- **PP-OCRv5**: Para detección y reconocimiento de texto
- **UVDoc**: Para corrección de distorsión de documentos
- **PP-LCNet**: Para orientación y clasificación de texto

**Procesamiento por Anexo**:
- Cada tipo de anexo tiene un extractor especializado
- Configuraciones de OCR optimizadas por formato de documento
- Parsers específicos para estructuras de datos diferentes

### 2. Sistema de Equivalencias de Nombres

**Archivo**: `src/db/synonyms.json`

El sistema incluye una base de datos de sinónimos médicos para manejar variaciones en nombres de insumos:

```json
{
  "PLACA DE TITANIO": [
    "placa titanio",
    "placa de titanium",
    "placa metalica",
    "implante placa"
  ],
  "TORNILLO CORTICAL": [
    "tornillo corticoesponjoso",
    "screw cortical",
    "tornillo 3.5mm"
  ]
}
```

**Algoritmo de Matching**:
1. **Fuzzy Matching** con RapidFuzz (umbral: 65%)
2. **Normalización de texto** sin acentos ni caracteres especiales
3. **Comparación directa** cuando no hay sinónimos disponibles
4. **Score de confianza** basado en similitud promedio

### 3. Manejo de Errores OCR

**Estrategias Implementadas**:
- **Campos no detectados**: Marcados como `[CAMPO_NO_DETECTADO_POR_OCR]`
- **Normalización de fechas**: Extracción de números y formato estándar DD-MM-YYYY
- **Limpieza de documentos**: Solo números para cédulas/documentos
- **Cantidades corregidas**: Conversión de letras mal leídas (V→1, I→1, L→1)

### 4. Lógica de Comparación

**Campos Básicos** (entre 3 documentos):
- Comparación por pares (1↔2, 2↔3, 1↔3)
- Identificación del anexo problemático
- Umbrales de similitud por tipo de campo:
  - Fechas: 85%
  - Nombres: 70%
  - Documentos: 100% (exacto)
  - Médicos: 65%
  - Procedimientos: 60%

**Tabla de Insumos** (solo Anexo 1 ↔ Anexo 2):
- Matching de productos con sinónimos
- Algoritmo greedy para mejores coincidencias
- Ponderación: 30% cantidad + 70% descripción

## 🚀 Instalación y Configuración

### Prerequisitos

- **Python 3.8 o superior**
- **Windows/Linux/macOS**
- **4GB RAM mínimo** (recomendado 8GB)
- **2GB espacio en disco** para modelos PaddlePaddle

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd vgmedical_checker
```

### 2. Crear Entorno Virtual

**Windows (PowerShell)**:
```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Si hay error de políticas de ejecución:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/macOS**:
```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate
```

### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 4. Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

### 5. Verificar Instalación de PaddlePaddle

PaddlePaddle se instala automáticamente con `requirements.txt`, pero puedes verificar:

```python
python -c "import paddle; print('PaddlePaddle versión:', paddle.__version__)"
```

**Si hay problemas con PaddlePaddle**:
```bash
# Desinstalar versión actual
pip uninstall paddlepaddle

# Instalar versión específica
pip install paddlepaddle==2.6.1
```

### 6. Verificar Instalación Completa

```bash
# Ejecutar test simple de un extractor
python tests/test_ocr_extractor_anexo3.py

# 🚀 O MEJOR AÚN: Ejecutar todo el sistema completo
python tests/test_document_matcher.py

# Si funciona correctamente, verás reportes detallados y archivos en output/
```

## 📖 Guía de Uso

### 1. Extracción Individual de Anexos

**Anexo 1** 
```bash
python src/ocr/ocr_extractor_anexo1.py ruta/a/anexo1.pdf --out_json output/anexo1_resultado.json
```

**Anexo 2** 
```bash
python src/ocr/ocr_extractor_anexo2.py ruta/a/anexo2.pdf --out_json output/anexo2_resultado.json
```

**Anexo 3** 
```bash
python src/ocr/ocr_extractor_anexo3.py ruta/a/anexo3.pdf --out_json output/anexo3_resultado.json
```

### 2. Comparación de 2 Documentos (Anexo 1 vs Anexo 2)

```bash
python src/matcher/document_matcher.py output/anexo1_resultado.json output/anexo2_resultado.json --output output/comparacion_2docs.json --verbose
```

### 3. Comparación de 3 Documentos

```bash
python src/matcher/document_matcher.py output/anexo1_resultado.json output/anexo2_resultado.json --anexo3 output/anexo3_resultado.json --output output/comparacion_3docs.json --verbose
```

### 4. 🚀 **EJECUTAR TODO EL SISTEMA COMPLETO (RECOMENDADO)**

**Para probar todo el sistema de una vez** (extracción de 3 anexos + comparación completa):

```bash
# ⭐ COMANDO PRINCIPAL - Ejecuta todo el flujo completo
python tests/test_document_matcher.py
```

Este comando hace automáticamente:
- ✅ Extrae datos de anexo1.pdf, anexo2.pdf y anexo3.pdf
- ✅ Compara campos básicos entre los 3 documentos
- ✅ Compara tabla de insumos entre anexo 1 y anexo 2
- ✅ Genera reportes completos en `output/`
- ✅ Muestra resultados claros en pantalla

### 5. Ejecutar Pruebas Individuales

```bash
# Test de extractor específico
python tests/test_ocr_extractor_anexo3.py

# Test de comparación manual
python tests/test_document_matcher.py
```

## 📄 Estructura de Salidas

### Extracción Individual

```json
{
  "extracted_fields": {
    "fecha": "24-12-2024",
    "paciente": "JUAN PEREZ GARCIA",
    "documento": "12345678",
    "medico": "DR. MARIA RODRIGUEZ",
    "procedimiento": "IMPLANTE DE PLACA"
  },
  "table_data": [
    {
      "descripcion": "PLACA DE TITANIO",
      "cantidad": "1"
    }
  ]
}
```

### Reporte de Comparación Simplificado

```json
{
  "ESTADO_GENERAL": "✅ TODOS LOS DOCUMENTOS ESTÁN CORRECTOS",
  "CAMPOS_COMPARADOS": {
    "paciente": {
      "anexo1_value": "JUAN PEREZ",
      "anexo2_value": "JUAN PEREZ GARCIA", 
      "anexo3_value": "JUAN PEREZ GARCIA",
      "status": "✅ CORRECTO",
      "problema_detectado": "Sin discrepancias"
    }
  },
  "TABLA_INSUMOS": {
    "status": "✅ TABLA CORRECTA",
    "similitud_total": 0.879,
    "items_coincidentes": 6,
    "nota": "Comparación entre anexo 1 y anexo 2 únicamente"
  }
}
```

## 🎯 Interpretación de Resultados

### Estados de Campos

- **✅ CORRECTO**: Los valores coinciden entre documentos
- **⚠️ REVISAR**: Hay discrepancias que requieren atención manual

### Estados de Tabla

- **✅ TABLA CORRECTA**: Similitud ≥ 80% y pocos items no coincidentes
- **⚠️ TABLA REQUIERE REVISIÓN**: Similitud < 80% o muchos items no coincidentes

### Tipos de Discrepancias

1. **Sin discrepancias**: Todos los valores coinciden
2. **Revisar anexo X**: Un anexo específico tiene valores diferentes
3. **Múltiples discrepancias**: Varios anexos tienen valores diferentes

## 🔧 Configuración Avanzada

### Ajustar Umbrales de Similitud

Editar `src/matcher/document_matcher.py`:

```python
self.SIMILARITY_THRESHOLDS = {
    'fecha': 0.85,          # Fechas (85%)
    'paciente': 0.70,       # Nombres (70%)
    'documento': 1.0,       # Documentos (100% exacto)
    'cirujano': 0.65,       # Médicos (65%)
    'procedimiento': 0.60   # Procedimientos (60%)
}
```

### Agregar Sinónimos Médicos

Editar `src/db/synonyms.json`:

```json
{
  "NUEVO_INSUMO": [
    "sinonimo1",
    "sinonimo2",
    "variante_nombre"
  ]
}
```

### Configurar Parámetros OCR

En cada extractor, ajustar configuración PaddlePaddle:

```python
# Cambiar modelo OCR
ocr = PPStructure(
    lang='es',  # Idioma español
    layout=False,
    table=True,
    ocr=True,
    show_log=False
)
```

## 🐛 Solución de Problemas Comunes

### Error: "No module named 'paddle'"

```bash
pip install paddlepaddle==2.6.1
# O para GPU:
pip install paddlepaddle-gpu==2.6.1
```

### Error: "Permission denied" al activar entorno virtual

**Windows**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: Memoria insuficiente durante OCR

1. Reducir resolución de imágenes
2. Procesar páginas una por una
3. Aumentar memoria virtual del sistema

### OCR no detecta texto correctamente

1. Verificar calidad del PDF (300 DPI mínimo)
2. Asegurar que el texto no esté rotado
3. Verificar que no sean imágenes escaneadas de baja calidad

### Comparación da muchas discrepancias

1. Revisar archivo de sinónimos `src/db/synonyms.json`
2. Ajustar umbrales de similitud
3. Verificar calidad de extracción OCR

## 📈 Rendimiento y Optimización

### Tiempos Esperados

- **Extracción por página**: 5-15 segundos
- **Comparación de 3 documentos**: 2-5 segundos
- **Primera ejecución**: +60 segundos (descarga de modelos)

### Optimizaciones

1. **Caché de modelos**: Los modelos se descargan una sola vez
2. **Procesamiento en lotes**: Para múltiples documentos
3. **Configuración de memoria**: Ajustar según hardware disponible

## 📝 Registro de Cambios

### Versión Actual
- ✅ Extracción OCR especializada por anexo
- ✅ Comparación de campos básicos entre 3 documentos
- ✅ Comparación de tabla de insumos solo entre anexo 1 y 2
- ✅ Sistema de sinónimos médicos con fuzzy matching
- ✅ Reportes simplificados y claros
- ✅ Manejo robusto de errores OCR
- ✅ Tests automatizados completos
