"""
Matcher para comparar datos extraídos de anexos 1, 2 y 3
Compara campos clave entre documentos y detecta similitudes/discrepancias
usando fuzzy matching para campos de texto y validación específica para documentos.

Para campos básicos: compara entre los 3 documentos como antes.
Para tabla de insumos: compara únicamente anexo 1 con anexo 2.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from rapidfuzz import fuzz
import re
import logging

# Cargar sinónimos
def load_synonyms(synonyms_file: str = "src/db/synonyms.json") -> Dict[str, List[str]]:
    """Carga la base de datos de sinónimos simplificada"""
    try:
        with open(synonyms_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Archivo de sinónimos {synonyms_file} no encontrado")
        return {}

def find_synonym_match(description, synonyms_dict, threshold=0.65):
    """
    Busca el mejor match de sinónimos usando SOLO fuzzy matching.
    NO aplica ninguna normalización porque no sabemos qué errores OCR ocurrirán.
    """
    if not description:
        return None, 0.0
    
    # Solo limpieza básica de espacios - SIN normalización de errores OCR
    clean_desc = description.strip()
    best_match = None
    best_score = 0.0
    
    # Buscar en todas las entradas de sinónimos
    for main_product, synonym_list in synonyms_dict.items():
        # Comparar con el producto principal usando fuzzy matching directo
        score = fuzz.ratio(clean_desc, main_product) / 100.0
        if score > best_score:
            best_score = score
            best_match = main_product
        
        # Comparar con cada sinónimo usando fuzzy matching directo
        for synonym in synonym_list:
            score = fuzz.ratio(clean_desc, synonym) / 100.0
            if score > best_score:
                best_score = score
                best_match = main_product
    
    if best_score >= threshold:
        return best_match, best_match, best_score
    return None, None, best_score

def are_products_synonymous(desc1: str, desc2: str, synonyms_dict: Dict[str, List[str]]) -> Tuple[bool, float]:
    """
    Determina si dos descripciones son sinónimos del mismo producto usando SOLO fuzzy matching
    
    Returns:
        Tuple[bool, float]: (son_sinonimos, similarity_score)
    """
    # Usar la nueva función de sinónimos
    match1, canonical1, score1 = find_synonym_match(desc1, synonyms_dict)
    match2, canonical2, score2 = find_synonym_match(desc2, synonyms_dict)
    
    # Si ambos tienen match y apuntan al mismo producto canónico
    if match1 and match2 and canonical1 == canonical2:
        avg_score = (score1 + score2) / 2
        return True, avg_score
    
    # Si no hay matches, usar comparación directa fuzzy
    direct_score = fuzz.ratio(desc1.strip(), desc2.strip()) / 100.0
    return direct_score > 0.7, direct_score
    

@dataclass
class MatchResult:
    field: str
    anexo1_value: str
    anexo2_value: str
    similarity_score: float
    status: str  # "Correcto", "Requiere revision"
    reason: str

@dataclass
class TableItemMatch:
    anexo1_item: Dict[str, Any]
    anexo2_item: Dict[str, Any]
    cantidad_similarity: float
    descripcion_similarity: float
    overall_similarity: float
    status: str
    reason: str

@dataclass
class TableComparison:
    matched_items: List[TableItemMatch]
    anexo1_only: List[Dict[str, Any]]  # Items solo en anexo1
    anexo2_only: List[Dict[str, Any]]  # Items solo en anexo2
    total_similarity: float
    status: str
    summary: Dict[str, int]

@dataclass
class DocumentComparison:
    fecha: MatchResult
    paciente: MatchResult
    documento: MatchResult
    cirujano: MatchResult
    procedimiento: MatchResult
    table_comparison: TableComparison
    overall_status: str
    confidence_score: float

@dataclass
class ThreeWayFieldComparison:
    """Comparación de un campo entre 3 documentos usando comparaciones por pares"""
    field: str
    anexo1_value: str
    anexo2_value: str
    anexo3_value: str
    pair_1_2: MatchResult  # Anexo 1 vs 2
    pair_2_3: MatchResult  # Anexo 2 vs 3
    pair_1_3: MatchResult  # Anexo 1 vs 3
    discrepancy_analysis: str  # "Sin discrepancias", "Revisar anexo X", "Múltiples discrepancias"
    recommendation: str

@dataclass
class ThreeWayDocumentComparison:
    """Comparación completa entre 3 documentos"""
    fecha: ThreeWayFieldComparison
    paciente: ThreeWayFieldComparison
    identificacion: ThreeWayFieldComparison
    ciudad: ThreeWayFieldComparison
    medico: ThreeWayFieldComparison
    procedimiento: ThreeWayFieldComparison
    table_comparison: Dict[str, Any]  # Comparación de tablas de insumos
    overall_status: str
    summary: Dict[str, int]  # Conteo de discrepancias por anexo

class DocumentMatcher:
    """Comparador de documentos médicos con fuzzy matching y sinónimos"""
    
    def __init__(self, synonyms_file: str = "src/db/synonyms.json"):
        # Umbrales de similitud
        self.SIMILARITY_THRESHOLDS = {
            'fecha': 0.85,          # Fechas deben ser muy similares
            'paciente': 0.70,       # Nombres pueden tener variaciones
            'documento': 1.0,       # Documentos deben ser exactos
            'cirujano': 0.65,       # Nombres de médicos pueden variar
            'procedimiento': 0.60   # Procedimientos pueden tener descripciones diferentes
        }
        
        # Campos que requieren coincidencia exacta
        self.STRICT_FIELDS = {'documento'}
        
        # Cargar sinónimos
        self.synonyms = load_synonyms(synonyms_file)
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
    
    def normalize_text(self, text: str) -> str:
        """Normaliza texto para comparación"""
        if not text or text == "[CAMPO_NO_DETECTADO_POR_OCR]":
            return ""
        
        # Convertir a minúsculas
        text = text.lower().strip()
        
        # Remover acentos
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
            'ñ': 'n'
        }
        for accented, normal in replacements.items():
            text = text.replace(accented, normal)
        
        # Limpiar caracteres especiales pero mantener espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_document_id(self, doc_id: str) -> str:
        """Normaliza número de documento removiendo espacios y caracteres"""
        if not doc_id or doc_id == "[No detectado]":
            return ""
        
        # Solo números
        return re.sub(r'[^\d]', '', doc_id)
    
    def load_synonyms(self, synonyms_file: str) -> Dict:
        """Carga archivo de sinónimos"""
        try:
            synonyms_path = Path(synonyms_file)
            if not synonyms_path.exists():
                logging.warning(f"Archivo de sinónimos {synonyms_file} no encontrado. Usando comparación directa.")
                return {}
            
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error cargando sinónimos: {e}")
            return {}
    
    def find_product_match(self, description: str) -> Optional[Tuple[str, str, float]]:
        """Encuentra el producto más similar en los sinónimos usando SOLO fuzzy matching
        
        Returns:
            Tuple[product_type, canonical_name, confidence] o None
        """
        if not self.synonyms:
            return None
        
        # Solo limpieza básica - SIN normalización predefinida
        clean_desc = description.strip()
        best_match = None
        best_score = 0.0
        threshold = 0.65
        
        for canonical_product, synonym_list in self.synonyms.items():
            # Comparar con el producto canónico
            score = fuzz.ratio(clean_desc, canonical_product) / 100.0
            if score > best_score:
                best_match = (canonical_product, canonical_product, score)
                best_score = score
            
            # Comparar con cada sinónimo
            for synonym in synonym_list:
                score = fuzz.ratio(clean_desc, synonym) / 100.0
                if score > best_score:
                    best_match = (canonical_product, canonical_product, score)
                    best_score = score
        
        if best_score >= threshold:
            return best_match
        
        return None
    
    def compare_products_with_synonyms(self, desc1: str, desc2: str) -> Tuple[float, str]:
        """Compara dos descripciones de productos usando sinónimos con fuzzy matching puro
        
        Returns:
            Tuple[similarity_score, reason]
        """
        # Usar la función simplificada de sinónimos
        are_synonyms, similarity = are_products_synonymous(desc1, desc2, self.synonyms)
        
        if are_synonyms:
            return similarity, f"Productos sinónimos detectados (similitud: {similarity:.1%})"
        
        # Si no son sinónimos, usar fuzzy matching directo
        direct_score = fuzz.ratio(desc1, desc2) / 100.0
        return direct_score, f"Comparación fuzzy directa (similitud: {direct_score:.1%})"
    def normalize_date(self, date_str: str) -> str:
        """Normaliza fecha a formato estándar"""
        if not date_str or date_str == "[No detectado]":
            return ""
        
        # Extraer números de la fecha
        nums = re.findall(r'\d+', date_str)
        if len(nums) >= 3:
            day, month, year = nums[:3]
            # Asegurar formato de 4 dígitos para el año
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            return f"{int(day):02d}-{int(month):02d}-{year}"
        
        return date_str
    
    def calculate_similarity(self, text1: str, text2: str, field: str) -> float:
        """Calcula similitud entre dos textos según el tipo de campo"""
        
        # Manejar campos vacíos o no detectados
        if not text1 or not text2:
            return 0.0

        if text1 == "[No detectado]" or text2 == "[No detectado]":
            return 0.0
        
        # Normalizar según el tipo de campo
        if field == 'documento':
            norm1 = self.normalize_document_id(text1)
            norm2 = self.normalize_document_id(text2)
            # Para documentos, debe ser coincidencia exacta
            return 1.0 if norm1 == norm2 else 0.0
        
        elif field == 'fecha':
            norm1 = self.normalize_date(text1)
            norm2 = self.normalize_date(text2)
        else:
            norm1 = self.normalize_text(text1)
            norm2 = self.normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Usar diferentes algoritmos según el campo
        if field in ['paciente', 'cirujano']:
            # Para nombres, usar token_sort_ratio que es más tolerante al orden
            return fuzz.token_sort_ratio(norm1, norm2) / 100.0
        elif field == 'procedimiento':
            # Para procedimientos, usar partial_ratio para coincidencias parciales
            return max(
                fuzz.partial_ratio(norm1, norm2),
                fuzz.token_set_ratio(norm1, norm2)
            ) / 100.0
        else:
            # Para otros campos, usar ratio simple
            return fuzz.ratio(norm1, norm2) / 100.0
    

    
    def normalize_quantity(self, qty: Any) -> str:
        """Normaliza cantidad para comparación"""
        if qty is None:
            return ""
        
        qty_str = str(qty).strip()
        
        # Convertir letras mal leídas a números
        if qty_str.upper() in ['V', 'I', 'L']:
            return "1"
        
        # Solo números
        qty_clean = re.sub(r'[^\d]', '', qty_str)
        return qty_clean if qty_clean else "0"
    
    def calculate_table_item_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calcula similitud entre dos items de tabla usando sinónimos"""
        
        # Extraer cantidades (diferentes formatos en anexo1 vs anexo2)
        qty1 = item1.get('cantidad', item1.get('cantidad', ''))
        qty2 = item2.get('cantidad', '')
        
        # Extraer descripciones (diferentes formatos en anexo1 vs anexo2)
        desc1 = item1.get('descripcion', item1.get('descripcion', ''))
        desc2 = item2.get('descripcion', '')
        
        # Normalizar cantidades
        norm_qty1 = self.normalize_quantity(qty1)
        norm_qty2 = self.normalize_quantity(qty2)
        
        # Calcular similitud de cantidad (exacta o 0)
        qty_similarity = 1.0 if norm_qty1 == norm_qty2 and norm_qty1 != "" else 0.0
        
        # Calcular similitud de descripción usando sinónimos
        if desc1 and desc2:
            desc_similarity, reason = self.compare_products_with_synonyms(desc1, desc2)
            logging.debug(f"Comparando '{desc1}' vs '{desc2}': {desc_similarity:.3f} - {reason}")
        else:
            desc_similarity = 0.0
        
        # Similitud general (promedio ponderado: cantidad 30%, descripción 70%)
        overall_similarity = (0.3 * qty_similarity) + (0.7 * desc_similarity)
        
        return qty_similarity, desc_similarity, overall_similarity
    
    def compare_products_with_synonyms(self, desc1: str, desc2: str) -> Tuple[float, str]:
        """
        Compara dos descripciones usando el sistema de sinónimos
        
        Returns:
            Tuple[float, str]: (similarity_score, reason)
        """
        # Usar el sistema de sinónimos
        are_synonymous, synonym_score = are_products_synonymous(desc1, desc2, self.synonyms)
        
        if are_synonymous:
            return synonym_score, f"Productos sinónimos identificados (similitud: {synonym_score:.1%})"
        
        # Si no son sinónimos, usar fuzzy matching directo
        direct_score = fuzz.ratio(desc1, desc2) / 100.0
        return direct_score, f"Comparación fuzzy directa (similitud: {direct_score:.1%})"
    
    def find_best_table_matches(self, table1: List[Dict], table2: List[Dict]) -> List[Tuple[int, int, float]]:
        """Encuentra las mejores coincidencias entre items de dos tablas"""
        matches = []
        
        # Calcular matriz de similitudes
        similarity_matrix = []
        for i, item1 in enumerate(table1):
            row = []
            for j, item2 in enumerate(table2):
                _, _, overall_sim = self.calculate_table_item_similarity(item1, item2)
                row.append(overall_sim)
            similarity_matrix.append(row)
        
        # Algoritmo greedy para encontrar mejores matches
        used_i = set()
        used_j = set()
        
        # Ordenar por similitud descendente
        all_matches = []
        for i in range(len(table1)):
            for j in range(len(table2)):
                all_matches.append((i, j, similarity_matrix[i][j]))
        
        all_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Seleccionar matches sin repetir índices
        for i, j, sim in all_matches:
            if i not in used_i and j not in used_j and sim >= 0.4:  # Umbral mínimo
                matches.append((i, j, sim))
                used_i.add(i)
                used_j.add(j)
        
        return matches
    
    def compare_tables(self, table1: List[Dict], table2: List[Dict]) -> TableComparison:
        """Compara las tablas de materiales de ambos anexos"""
        
        if not table1 and not table2:
            return TableComparison(
                matched_items=[],
                anexo1_only=[],
                anexo2_only=[],
                total_similarity=1.0,
                status="Correcto",
                summary={"matches": 0, "anexo1_only": 0, "anexo2_only": 0}
            )
        
        if not table1 or not table2:
            return TableComparison(
                matched_items=[],
                anexo1_only=table1 or [],
                anexo2_only=table2 or [],
                total_similarity=0.0,
                status="Requiere revision",
                summary={"matches": 0, "anexo1_only": len(table1 or []), "anexo2_only": len(table2 or [])}
            )
        
        # Encontrar mejores matches
        best_matches = self.find_best_table_matches(table1, table2)
        
        # Crear objetos TableItemMatch
        matched_items = []
        matched_indices_1 = set()
        matched_indices_2 = set()
        
        for i, j, overall_sim in best_matches:
            item1 = table1[i]
            item2 = table2[j]
            
            qty_sim, desc_sim, _ = self.calculate_table_item_similarity(item1, item2)
            
            # Determinar status del match
            if overall_sim >= 0.7:
                status = "Correcto"
                reason = f"Similitud alta ({overall_sim:.1%})"
            elif overall_sim >= 0.4:
                status = "Requiere revision"
                reason = f"Similitud media ({overall_sim:.1%})"
            else:
                status = "Requiere revision"
                reason = f"Similitud baja ({overall_sim:.1%})"
            
            match = TableItemMatch(
                anexo1_item=item1,
                anexo2_item=item2,
                cantidad_similarity=qty_sim,
                descripcion_similarity=desc_sim,
                overall_similarity=overall_sim,
                status=status,
                reason=reason
            )
            
            matched_items.append(match)
            matched_indices_1.add(i)
            matched_indices_2.add(j)
        
        # Items solo en anexo1
        anexo1_only = [table1[i] for i in range(len(table1)) if i not in matched_indices_1]
        
        # Items solo en anexo2
        anexo2_only = [table2[j] for j in range(len(table2)) if j not in matched_indices_2]
        
        # Calcular similitud total
        if matched_items:
            total_similarity = sum(item.overall_similarity for item in matched_items) / len(matched_items)
        else:
            total_similarity = 0.0
        
        # Determinar status general de la tabla
        if total_similarity >= 0.8 and len(anexo1_only) == 0 and len(anexo2_only) == 0:
            table_status = "Correcto"
        elif total_similarity >= 0.6 and len(anexo1_only) <= 1 and len(anexo2_only) <= 1:
            table_status = "Correcto"
        else:
            table_status = "Requiere revision"
        
        return TableComparison(
            matched_items=matched_items,
            anexo1_only=anexo1_only,
            anexo2_only=anexo2_only,
            total_similarity=total_similarity,
            status=table_status,
            summary={
                "matches": len(matched_items),
                "anexo1_only": len(anexo1_only),
                "anexo2_only": len(anexo2_only)
            }
        )
    
    def compare_field(self, field: str, value1: str, value2: str) -> MatchResult:
        """Compara un campo específico entre dos documentos"""
        
        # Calcular similitud
        similarity = self.calculate_similarity(value1, value2, field)
        threshold = self.SIMILARITY_THRESHOLDS.get(field, 0.7)
        
        # Determinar status
        if not value1 or not value2:
            status = "Requiere revision"
            reason = "Campo faltante en uno o ambos documentos"
        elif value1 == "[No detectado]" or value2 == "[No detectado]":
            status = "Requiere revision"
            reason = "Campo no detectado por OCR"
        elif field == 'documento':
            # Documentos requieren coincidencia exacta
            if similarity == 1.0:
                status = "Correcto"
                reason = "Documento de identidad coincide exactamente"
            else:
                status = "Requiere revision"
                reason = "Documentos de identidad no coinciden - verificar manualmente"
        elif similarity >= threshold:
            status = "Correcto"
            reason = f"Similitud alta ({similarity:.2%})"
        else:
            status = "Requiere revision"
            reason = f"Similitud baja ({similarity:.2%}), por debajo del umbral ({threshold:.2%})"
        
        return MatchResult(
            field=field,
            anexo1_value=value1,
            anexo2_value=value2,
            similarity_score=similarity,
            status=status,
            reason=reason
        )
    
    def compare_documents(self, anexo1_data: Dict[str, Any], anexo2_data: Dict[str, Any]) -> DocumentComparison:
        """Compara dos documentos completos incluyendo tablas"""
        
        # Extraer campos de ambos documentos
        a1_fields = anexo1_data.get('extracted_fields', {})
        a2_fields = anexo2_data.get('extracted_fields', {})
        
        # Mapear campos equivalentes (anexo1_field, anexo2_field)
        field_mappings = {
            'fecha': ('fecha', 'fecha'),
            'paciente': ('paciente', 'paciente'),
            'documento': ('identificacion', 'documento'),  # anexo1 usa 'identificacion'
            'cirujano': ('especialista', 'cirujano'),      # anexo1 usa 'especialista'
            'procedimiento': ('procedimiento', 'procedimiento')
        }
        
        results = {}
        total_score = 0
        valid_comparisons = 0
        
        # Comparar cada campo
        for field, (a1_key, a2_key) in field_mappings.items():
            a1_value = str(a1_fields.get(a1_key, ""))
            a2_value = str(a2_fields.get(a2_key, ""))
            
            result = self.compare_field(field, a1_value, a2_value)
            results[field] = result
            
            # Solo contar campos válidos para el score general
            if result.similarity_score > 0:
                total_score += result.similarity_score
                valid_comparisons += 1
        
        # Comparar tablas
        table1 = anexo1_data.get('table_data', [])
        table2 = anexo2_data.get('table_data', [])
        table_comparison = self.compare_tables(table1, table2)
        
        # Incluir similitud de tabla en el score general
        if table_comparison.total_similarity > 0:
            total_score += table_comparison.total_similarity
            valid_comparisons += 1
        
        # Calcular score de confianza general
        confidence_score = total_score / valid_comparisons if valid_comparisons > 0 else 0.0
        
        # Determinar status general
        critical_fields_ok = all(
            results[field].status == "Correcto" 
            for field in ['documento', 'paciente']
            if results[field].similarity_score > 0
        )
        
        table_ok = table_comparison.status == "Correcto"
        
        if confidence_score >= 0.75 and critical_fields_ok and table_ok:
            overall_status = "Correcto"
        else:
            overall_status = "Requiere revision"
        
        return DocumentComparison(
            fecha=results['fecha'],
            paciente=results['paciente'],
            documento=results['documento'],
            cirujano=results['cirujano'],
            procedimiento=results['procedimiento'],
            table_comparison=table_comparison,
            overall_status=overall_status,
            confidence_score=confidence_score
        )
    
    def generate_report(self, comparison: DocumentComparison) -> Dict[str, Any]:
        """Genera reporte detallado de la comparación incluyendo tablas"""
        
        fields_report = {}
        for field in ['fecha', 'paciente', 'documento', 'cirujano', 'procedimiento']:
            result = getattr(comparison, field)
            fields_report[field] = {
                'anexo1_value': result.anexo1_value,
                'anexo2_value': result.anexo2_value,
                'similarity_score': round(result.similarity_score, 3),
                'status': result.status,
                'reason': result.reason
            }
        
        # Reporte de tabla
        table_report = {
            'total_similarity': round(comparison.table_comparison.total_similarity, 3),
            'status': comparison.table_comparison.status,
            'summary': comparison.table_comparison.summary,
            'matched_items': [],
            'anexo1_only': comparison.table_comparison.anexo1_only,
            'anexo2_only': comparison.table_comparison.anexo2_only
        }
        
        # Detalles de items coincidentes
        for match in comparison.table_comparison.matched_items:
            table_report['matched_items'].append({
                'anexo1_item': match.anexo1_item,
                'anexo2_item': match.anexo2_item,
                'cantidad_similarity': round(match.cantidad_similarity, 3),
                'descripcion_similarity': round(match.descripcion_similarity, 3),
                'overall_similarity': round(match.overall_similarity, 3),
                'status': match.status,
                'reason': match.reason
            })
        
        return {
            'overall_status': comparison.overall_status,
            'confidence_score': round(comparison.confidence_score, 3),
           'field_comparisons': fields_report,
            'table_comparison': table_report,
            'summary': {
                'correcto_count': sum(1 for field in fields_report.values() if field['status'] == 'Correcto'),
                'revision_count': sum(1 for field in fields_report.values() if field['status'] == 'Requiere revision'),
                'total_fields': len(fields_report),
                'table_status': table_report['status'],
                'table_matches': len(table_report['matched_items']),
                'items_only_anexo1': len(table_report['anexo1_only']),
                'items_only_anexo2': len(table_report['anexo2_only'])
            }
        }

    def compare_field_three_way(self, field: str, value1: str, value2: str, value3: str) -> ThreeWayFieldComparison:
        """Compara un campo entre 3 documentos usando comparaciones por pares"""
        
        # Realizar comparaciones por pares
        pair_1_2 = self.compare_field(field, value1, value2)
        pair_2_3 = self.compare_field(field, value2, value3)
        pair_1_3 = self.compare_field(field, value1, value3)
        
        # Analizar discrepancias
        results = [pair_1_2, pair_2_3, pair_1_3]
        correct_count = sum(1 for r in results if r.status == "Correcto")
        
        if correct_count == 3:
            # Todos coinciden
            discrepancy_analysis = "Sin discrepancias"
            recommendation = "Todos los valores coinciden correctamente"
        elif correct_count == 2:
            # Solo uno no coincide - identificar cuál
            if pair_1_2.status == "Correcto" and pair_1_3.status == "Correcto":
                # 1 y 2 coinciden, 1 y 3 coinciden, entonces 2 es el problema
                discrepancy_analysis = "Revisar anexo 2"
                recommendation = f"El valor en anexo 2 ('{value2}') no coincide con anexos 1 y 3"
            elif pair_1_2.status == "Correcto" and pair_2_3.status == "Correcto":
                # 1 y 2 coinciden, 2 y 3 coinciden, entonces 3 es el problema
                discrepancy_analysis = "Revisar anexo 3"
                recommendation = f"El valor en anexo 3 ('{value3}') no coincide con anexos 1 y 2"
            elif pair_1_3.status == "Correcto" and pair_2_3.status == "Correcto":
                # 1 y 3 coinciden, 2 y 3 coinciden, entonces 1 es el problema
                discrepancy_analysis = "Revisar anexo 1"
                recommendation = f"El valor en anexo 1 ('{value1}') no coincide con anexos 2 y 3"
            else:
                # Caso extraño - múltiples discrepancias
                discrepancy_analysis = "Múltiples discrepancias"
                recommendation = "Se requiere revisión manual de todos los documentos"
        elif correct_count == 1:
            # Solo uno coincide
            discrepancy_analysis = "Múltiples discrepancias"
            recommendation = "Se requiere revisión manual - múltiples valores diferentes"
        else:
            # Ninguno coincide
            discrepancy_analysis = "Múltiples discrepancias"
            recommendation = "Ningún documento coincide - revisar todos los valores"
        
        return ThreeWayFieldComparison(
            field=field,
            anexo1_value=value1,
            anexo2_value=value2,
            anexo3_value=value3,
            pair_1_2=pair_1_2,
            pair_2_3=pair_2_3,
            pair_1_3=pair_1_3,
            discrepancy_analysis=discrepancy_analysis,
            recommendation=recommendation
        )

    def compare_three_tables(self, table1: List[Dict], table2: List[Dict], table3: List[Dict]) -> Dict[str, Any]:
        """Compara tablas de insumos de 3 documentos usando comparaciones por pares"""
        
        # Normalizar estructuras de tabla (diferentes nombres de campos en cada anexo)
        def normalize_table_item(item, source):
            if source == 'anexo1':
                return {
                    'descripcion': item.get('descripcion', ''),
                    'cantidad': item.get('cantidad', 0)
                }
            elif source == 'anexo2': 
                return {
                    'descripcion': item.get('descripcion', ''),
                    'cantidad': int(item.get('cantidad', 0)) if str(item.get('cantidad', 0)).isdigit() else 0
                }
            else:  # anexo3
                return {
                    'descripcion': item.get('descripcion', ''),
                    'cantidad': item.get('cantidad', 0)
                }
        
        # Normalizar todas las tablas
        norm_table1 = [normalize_table_item(item, 'anexo1') for item in table1]
        norm_table2 = [normalize_table_item(item, 'anexo2') for item in table2] 
        norm_table3 = [normalize_table_item(item, 'anexo3') for item in table3]
        
        # Realizar comparaciones por pares usando el método existente
        comp_1_2 = self.compare_tables(table1, table2)
        comp_2_3 = self.compare_tables(norm_table2, norm_table3) 
        comp_1_3 = self.compare_tables(norm_table1, norm_table3)
        
        # Analizar concordancia entre las 3 tablas
        total_items = len(norm_table1) + len(norm_table2) + len(norm_table3)
        
        # Contar items únicos en cada anexo
        all_descriptions = set()
        for table in [norm_table1, norm_table2, norm_table3]:
            for item in table:
                all_descriptions.add(item['descripcion'].lower().strip())
        
        # Análisis de concordancia
        if comp_1_2.status == "Correcto" and comp_2_3.status == "Correcto" and comp_1_3.status == "Correcto":
            overall_status = "Todas las tablas coinciden"
            discrepancy_analysis = "Sin discrepancias"
        elif comp_1_2.status == "Correcto":
            overall_status = "Revisar anexo 3"
            discrepancy_analysis = "Anexo 3 tiene diferencias"
        elif comp_2_3.status == "Correcto":
            overall_status = "Revisar anexo 1" 
            discrepancy_analysis = "Anexo 1 tiene diferencias"
        elif comp_1_3.status == "Correcto":
            overall_status = "Revisar anexo 2"
            discrepancy_analysis = "Anexo 2 tiene diferencias"
        else:
            overall_status = "Múltiples discrepancias en tablas"
            discrepancy_analysis = "Múltiples discrepancias"
        
        return {
            'status': overall_status,
            'discrepancy_analysis': discrepancy_analysis,
            'pairwise_comparisons': {
                'anexo1_vs_anexo2': {
                    'status': comp_1_2.status,
                    'similarity': comp_1_2.total_similarity,
                    'matches': len(comp_1_2.matched_items),
                    'anexo1_only': len(comp_1_2.anexo1_only),
                    'anexo2_only': len(comp_1_2.anexo2_only)
                },
                'anexo2_vs_anexo3': {
                    'status': comp_2_3.status,
                    'similarity': comp_2_3.total_similarity,
                    'matches': len(comp_2_3.matched_items),
                    'anexo2_only': len(comp_2_3.anexo1_only),  # En esta comparación anexo2 es el "anexo1"
                    'anexo3_only': len(comp_2_3.anexo2_only)
                },
                'anexo1_vs_anexo3': {
                    'status': comp_1_3.status,
                    'similarity': comp_1_3.total_similarity,
                    'matches': len(comp_1_3.matched_items),
                    'anexo1_only': len(comp_1_3.anexo1_only),
                    'anexo3_only': len(comp_1_3.anexo2_only)
                }
            },
            'summary': {
                'total_items_anexo1': len(norm_table1),
                'total_items_anexo2': len(norm_table2),
                'total_items_anexo3': len(norm_table3),
                'unique_descriptions': len(all_descriptions)
            }
        }

    def compare_three_documents(self, anexo1_data: Dict[str, Any], anexo2_data: Dict[str, Any], anexo3_data: Dict[str, Any]) -> ThreeWayDocumentComparison:
        """
        Compara 3 documentos usando comparaciones por pares para campos básicos
        y solo compara tabla de insumos entre anexo 1 y anexo 2
        """
        
        # Extraer campos de cada anexo
        anexo1_fields = anexo1_data.get('extracted_fields', {})
        anexo2_fields = anexo2_data.get('extracted_fields', {})
        anexo3_fields = anexo3_data.get('extracted_fields', {})
        
        # Mapear campos entre anexos (cada anexo puede tener nombres de campo diferentes)
        field_mappings = {
            'fecha': {
                'anexo1': 'fecha',
                'anexo2': 'fecha',
                'anexo3': 'fecha_impresion'
            },
            'paciente': {
                'anexo1': 'paciente',
                'anexo2': 'paciente', 
                'anexo3': 'nombre'
            },
            'identificacion': {
                'anexo1': 'identificacion',
                'anexo2': 'documento',
                'anexo3': 'documento'
            },
            'ciudad': {
                'anexo1': 'ciudad',
                'anexo2': None,  # Anexo 2 no tiene campo ciudad
                'anexo3': 'lugar'
            },
            'medico': {
                'anexo1': 'especialista',
                'anexo2': 'cirujano',
                'anexo3': 'medico_tratante'
            },
            'procedimiento': {
                'anexo1': 'procedimiento',
                'anexo2': 'procedimiento',
                'anexo3': None  # Anexo 3 no tiene procedimiento específico
            }
        }
        
        # Comparar cada campo entre los 3 documentos
        comparisons = {}
        for field, mappings in field_mappings.items():
            # Obtener valores con manejo de campos faltantes
            value1 = anexo1_fields.get(mappings['anexo1']) if mappings['anexo1'] else None
            value2 = anexo2_fields.get(mappings['anexo2']) if mappings['anexo2'] else None  
            value3 = anexo3_fields.get(mappings['anexo3']) if mappings['anexo3'] else None
            
            # Convertir None a mensaje de campo no disponible
            if value1 is None:
                value1 = '[CAMPO NO DISPONIBLE EN ANEXO 1]'
            elif value1 == '[CAMPO_NO_DETECTADO_POR_OCR]':
                value1 = '[NO DETECTADO POR OCR]'
                
            if value2 is None:
                value2 = '[CAMPO NO DISPONIBLE EN ANEXO 2]'
            elif value2 == '[CAMPO_NO_DETECTADO_POR_OCR]':
                value2 = '[NO DETECTADO POR OCR]'
                
            if value3 is None:
                value3 = '[CAMPO NO DISPONIBLE EN ANEXO 3]'
            elif value3 == '[CAMPO_NO_DETECTADO_POR_OCR]':
                value3 = '[NO DETECTADO POR OCR]'
            
            comparisons[field] = self.compare_field_three_way(field, value1, value2, value3)
        
        # NUEVA LÓGICA: Comparar tabla de insumos SOLO entre anexo 1 y anexo 2
        # Se mantiene la comparación de campos básicos entre los 3 documentos,
        # pero la tabla de insumos solo se compara entre anexo 1 y anexo 2
        table1 = anexo1_data.get('table_data', [])
        table2 = anexo2_data.get('table_data', [])
        
        # Usar método de comparación de 2 documentos para la tabla
        table_comparison = self.compare_tables(table1, table2)
        
        # Convertir resultado de comparación de 2 documentos a formato compatible
        table_comparison_result = {
            'status': table_comparison.status,
            'total_similarity': table_comparison.total_similarity,
            'matched_items': table_comparison.matched_items,
            'anexo1_only': table_comparison.anexo1_only,
            'anexo2_only': table_comparison.anexo2_only,
            'summary': table_comparison.summary,
            'comparison_scope': 'anexo1_vs_anexo2_only',  # Indicador de que solo comparamos 1 vs 2
            'note': 'Tabla de insumos comparada únicamente entre anexo 1 y anexo 2 (anexo 3 excluido)'
        }
        
        # Calcular resumen de discrepancias
        discrepancy_summary = {'anexo1': 0, 'anexo2': 0, 'anexo3': 0, 'multiples': 0}
        overall_issues = 0
        
        for comp in comparisons.values():
            if comp.discrepancy_analysis == "Revisar anexo 1":
                discrepancy_summary['anexo1'] += 1
                overall_issues += 1
            elif comp.discrepancy_analysis == "Revisar anexo 2":
                discrepancy_summary['anexo2'] += 1
                overall_issues += 1
            elif comp.discrepancy_analysis == "Revisar anexo 3":
                discrepancy_summary['anexo3'] += 1
                overall_issues += 1
            elif comp.discrepancy_analysis == "Múltiples discrepancias":
                discrepancy_summary['multiples'] += 1
                overall_issues += 1
        
        # Incluir resultado de tabla en el análisis general
        if table_comparison.status == "Requiere revision":
            overall_issues += 1
            discrepancy_summary['anexo2'] += 1  # Asignar a anexo2 por defecto
        
        # Determinar estado general
        if overall_issues == 0:
            overall_status = "Todos los documentos coinciden"
        elif discrepancy_summary['multiples'] > 0:
            overall_status = "Múltiples discrepancias requieren revisión manual"
        else:
            max_issues = max(discrepancy_summary['anexo1'], discrepancy_summary['anexo2'], discrepancy_summary['anexo3'])
            if discrepancy_summary['anexo1'] == max_issues:
                overall_status = f"Anexo 1 requiere revisión ({max_issues} discrepancias)"
            elif discrepancy_summary['anexo2'] == max_issues:
                overall_status = f"Anexo 2 requiere revisión ({max_issues} discrepancias)"
            else:
                overall_status = f"Anexo 3 requiere revisión ({max_issues} discrepancias)"
        
        return ThreeWayDocumentComparison(
            fecha=comparisons['fecha'],
            paciente=comparisons['paciente'],
            identificacion=comparisons['identificacion'],
            ciudad=comparisons['ciudad'],
            medico=comparisons['medico'],
            procedimiento=comparisons['procedimiento'],
            table_comparison=table_comparison_result,
            overall_status=overall_status,
            summary=discrepancy_summary
        )

    def format_three_way_report(self, comparison: ThreeWayDocumentComparison) -> Dict[str, Any]:
        """
        Genera reporte CLARO y SIMPLE de comparación de 3 documentos
        Campos básicos: comparación entre 3 documentos
        Tabla de insumos: comparación solo entre anexo 1 y anexo 2
        """
        
        fields_report = {}
        
        for field_name, field_comp in [
            ('fecha', comparison.fecha),
            ('paciente', comparison.paciente), 
            ('identificacion', comparison.identificacion),
            ('ciudad', comparison.ciudad),
            ('medico', comparison.medico),
            ('procedimiento', comparison.procedimiento),
        ]:
            # Reporte SIMPLIFICADO - sin pair_comparisons complejos
            status_simple = "✅ CORRECTO" if field_comp.discrepancy_analysis == "Sin discrepancias" else "⚠️ REVISAR"
            
            fields_report[field_name] = {
                'anexo1_value': field_comp.anexo1_value,
                'anexo2_value': field_comp.anexo2_value,
                'anexo3_value': field_comp.anexo3_value,
                'status': status_simple,
                'discrepancy_analysis': field_comp.discrepancy_analysis,  # Para compatibilidad con tests
                'recommendation': field_comp.recommendation,  # Para compatibilidad con tests
                'problema_detectado': field_comp.discrepancy_analysis,
                'accion_requerida': field_comp.recommendation,
                # Calculamos un score general simple basado en las comparaciones por pares
                'confianza_general': round((
                    field_comp.pair_1_2.similarity_score + 
                    field_comp.pair_2_3.similarity_score + 
                    field_comp.pair_1_3.similarity_score
                ) / 3, 2)
            }
        
        # Manejar el reporte de tabla que ahora solo compara anexo 1 vs anexo 2
        table_comparison = comparison.table_comparison
        
        # Reporte SIMPLIFICADO de tabla
        table_status_simple = "✅ TABLA CORRECTA" if table_comparison['status'] == "Correcto" else "⚠️ TABLA REQUIERE REVISIÓN"
        
        table_report = {
            'status': table_status_simple,
            'total_similarity': table_comparison['total_similarity'],  # Para compatibilidad con tests
            'similarity': table_comparison['total_similarity'],  # Alias
            'similitud_total': f"{round(table_comparison['total_similarity'] * 100, 1)}%",
            'mensaje': table_comparison.get('note', 'Tabla comparada solo entre anexo 1 y anexo 2'),
            'comparison_scope': table_comparison.get('comparison_scope', 'anexo1_vs_anexo2_only'),  # Para compatibilidad con tests
            'note': table_comparison.get('note', 'Tabla comparada solo entre anexo 1 y anexo 2'),  # Para compatibilidad con tests
            'summary': table_comparison['summary'],  # Para compatibilidad con tests
            'resumen_items': {
                'items_que_coinciden': table_comparison['summary']['matches'],
                'items_solo_en_anexo1': table_comparison['summary']['anexo1_only'],
                'items_solo_en_anexo2': table_comparison['summary']['anexo2_only']
            },
            'matched_items': [],  # Se llenará más abajo
            'anexo1_only': table_comparison['anexo1_only'],  # Para compatibilidad con tests
            'anexo2_only': table_comparison['anexo2_only'],  # Para compatibilidad con tests
            'items_coincidentes': [],
            'items_anexo1_unicos': table_comparison['anexo1_only'],
            'items_anexo2_unicos': table_comparison['anexo2_only']
        }
        
        # Detalles SIMPLIFICADOS de items coincidentes
        matched_items_simple = []
        for match in table_comparison['matched_items']:
            item_status = "✅ COINCIDE" if match.overall_similarity >= 0.7 else "⚠️ REVISAR"
            matched_items_simple.append({
                'anexo1_item': match.anexo1_item,
                'anexo2_item': match.anexo2_item,
                'cantidad_similarity': round(match.cantidad_similarity, 3),
                'descripcion_similarity': round(match.descripcion_similarity, 3),
                'overall_similarity': round(match.overall_similarity, 3),
                'status': match.status,
                'reason': match.reason
            })
            table_report['items_coincidentes'].append({
                'anexo1': f"Cant: {match.anexo1_item.get('cantidad', 'N/A')} - {match.anexo1_item.get('descripcion', 'N/A')}",
                'anexo2': f"Cant: {match.anexo2_item.get('cantidad', 'N/A')} - {match.anexo2_item.get('descripcion', 'N/A')}",
                'status': item_status,
                'similitud': f"{round(match.overall_similarity * 100, 1)}%"
            })
        
        # Actualizar campo matched_items para compatibilidad con tests
        table_report['matched_items'] = matched_items_simple
        
        # RESUMEN GENERAL SIMPLIFICADO
        campos_correctos = len([f for f in fields_report.values() if f['status'] == '✅ CORRECTO'])
        campos_con_problemas = len([f for f in fields_report.values() if f['status'] == '⚠️ REVISAR'])
        
        # Determinar qué anexos tienen más problemas
        anexos_con_problemas = []
        if comparison.summary['anexo1'] > 0:
            anexos_con_problemas.append(f"Anexo 1 ({comparison.summary['anexo1']} problemas)")
        if comparison.summary['anexo2'] > 0:
            anexos_con_problemas.append(f"Anexo 2 ({comparison.summary['anexo2']} problemas)")
        if comparison.summary['anexo3'] > 0:
            anexos_con_problemas.append(f"Anexo 3 ({comparison.summary['anexo3']} problemas)")
        
        estado_general = "✅ TODOS LOS DOCUMENTOS ESTÁN CORRECTOS" if not anexos_con_problemas else f"⚠️ REVISAR: {', '.join(anexos_con_problemas)}"
        
        return {
            'overall_status': estado_general,  # Para compatibilidad con tests
            'ESTADO_GENERAL': estado_general,
            'field_comparisons': fields_report,  # Para compatibilidad con tests
            'CAMPOS_COMPARADOS': fields_report,
            'table_comparison': table_report,  # Para compatibilidad con tests
            'TABLA_INSUMOS': table_report,
            'summary': {  # Para compatibilidad con tests
                'total_fields': len(fields_report),
                'anexo1_issues': comparison.summary['anexo1'],
                'anexo2_issues': comparison.summary['anexo2'], 
                'anexo3_issues': comparison.summary['anexo3'],
                'multiple_discrepancies': comparison.summary['multiples'],
                'fields_without_issues': len([f for f in fields_report.values() if f['problema_detectado'] == 'Sin discrepancias']),
                'table_comparison_note': 'Tabla de insumos comparada únicamente entre anexo 1 y anexo 2'
            },
            'RESUMEN_EJECUTIVO': {
                'total_campos_revisados': len(fields_report),
                'campos_correctos': campos_correctos,
                'campos_con_problemas': campos_con_problemas,
                'tabla_status': table_status_simple,
                'anexos_que_requieren_atencion': anexos_con_problemas,
                'nota_importante': 'Los campos básicos se comparan entre los 3 documentos. La tabla de insumos se compara únicamente entre anexo 1 y anexo 2.'
            }
        }

def main():
    parser = argparse.ArgumentParser(description='Comparar datos extraídos entre anexos médicos')
    parser.add_argument('anexo1_json', help='Archivo JSON con datos del anexo 1')
    parser.add_argument('anexo2_json', help='Archivo JSON con datos del anexo 2')
    parser.add_argument('--anexo3', help='Archivo JSON con datos del anexo 3 (opcional para comparación de 3 documentos)')
    parser.add_argument('--output', '-o', help='Archivo de salida para el reporte de comparación')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mostrar detalles adicionales')
    
    args = parser.parse_args()
    
    # Cargar datos
    try:
        with open(args.anexo1_json, 'r', encoding='utf-8') as f:
            anexo1_data = json.load(f)
        
        with open(args.anexo2_json, 'r', encoding='utf-8') as f:
            anexo2_data = json.load(f)
        
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        return 1
    
    print(f"Anexo 1 cargado exitosamente:")
    print(f"  - Número de expediente: {anexo1_data.get('numero_expediente', 'N/A')}")
    print(f"  - Paciente: {anexo1_data.get('nombre_paciente', 'N/A')}")
    print(f"  - Items en tabla: {len(anexo1_data.get('insumos', []))}")
    print()
    
    print(f"Anexo 2 cargado exitosamente:")
    print(f"  - Número de expediente: {anexo2_data.get('numero_expediente', 'N/A')}")
    print(f"  - Paciente: {anexo2_data.get('nombre_paciente', 'N/A')}")
    print(f"  - Items en tabla: {len(anexo2_data.get('insumos', []))}")
    
    # Manejar comparación de 3 documentos si se proporciona anexo3
    if args.anexo3:
        print()
        
        # Cargar anexo 3
        try:
            with open(args.anexo3, 'r', encoding='utf-8') as f:
                anexo3_data = json.load(f)
        except Exception as e:
            print(f"Error cargando anexo 3: {e}")
            return 1
        
        print(f"Anexo 3 cargado exitosamente:")
        print(f"  - Número de expediente: {anexo3_data.get('numero_expediente', 'N/A')}")
        print(f"  - Paciente: {anexo3_data.get('nombre_paciente', 'N/A')}")
        print(f"  - Items en tabla: {len(anexo3_data.get('insumos', []))}")
        print()
        
        # Realizar comparación de 3 documentos
        matcher = DocumentMatcher()
        three_way_comparison = matcher.compare_three_documents(anexo1_data, anexo2_data, anexo3_data)
        report = matcher.format_three_way_report(three_way_comparison)
        
        # Mostrar resultados de comparación de 3 documentos de manera CLARA
        print("=" * 80)
        print("🔍 REPORTE DE COMPARACIÓN DE DOCUMENTOS MÉDICOS")
        print("=" * 80)
        print(f"📋 {report['ESTADO_GENERAL']}")
        print()
        
        print("📊 RESUMEN EJECUTIVO:")
        resumen = report['RESUMEN_EJECUTIVO']
        print(f"   ✓ Campos revisados: {resumen['total_campos_revisados']}")
        print(f"   ✓ Campos correctos: {resumen['campos_correctos']}")
        print(f"   ⚠ Campos con problemas: {resumen['campos_con_problemas']}")
        print(f"   📦 Estado tabla: {resumen['tabla_status']}")
        
        if resumen['anexos_que_requieren_atencion']:
            print(f"   🔍 Anexos que requieren atención: {resumen['anexos_que_requieren_atencion']}")
        else:
            print("   ✅ Todos los anexos están correctos")
        print()
        
        print("📝 COMPARACIÓN DETALLADA POR CAMPOS:")
        print("-" * 50)
        for field, field_data in report['CAMPOS_COMPARADOS'].items():
            print(f"\n🏷️  {field.upper()}:")
            print(f"   📄 Anexo 1: {field_data['anexo1_value']}")
            print(f"   📄 Anexo 2: {field_data['anexo2_value']}")
            print(f"   📄 Anexo 3: {field_data['anexo3_value']}")
            print(f"   📊 {field_data['status']} (Confianza: {field_data['confianza_general']:.0%})")
            
            if field_data['status'] == '⚠️ REVISAR':
                print(f"   🔍 Problema: {field_data['problema_detectado']}")
                print(f"   💡 Acción: {field_data['accion_requerida']}")
        
        print(f"\n📦 TABLA DE INSUMOS (ANEXO 1 vs ANEXO 2):")
        print("-" * 50)
        tabla_info = report['TABLA_INSUMOS']
        print(f"   📊 {tabla_info['status']}")
        print(f"   📈 Similitud total: {tabla_info['similitud_total']}")
        print(f"   🔢 Items que coinciden: {tabla_info['resumen_items']['items_que_coinciden']}")
        print(f"   📋 Items solo en anexo 1: {tabla_info['resumen_items']['items_solo_en_anexo1']}")
        print(f"   📋 Items solo en anexo 2: {tabla_info['resumen_items']['items_solo_en_anexo2']}")
        
        if args.verbose and tabla_info['items_coincidentes']:
            print(f"\n   🔍 DETALLE DE ITEMS QUE COINCIDEN:")
            for i, item in enumerate(tabla_info['items_coincidentes'][:5], 1):  # Mostrar máximo 5
                print(f"     {i}. {item['status']} ({item['similitud']})")
                print(f"        📄 Anexo 1: {item['anexo1']}")
                print(f"        📄 Anexo 2: {item['anexo2']}")
            
            if len(tabla_info['items_coincidentes']) > 5:
                print(f"     ... y {len(tabla_info['items_coincidentes']) - 5} items más")
        
        if tabla_info['items_anexo1_unicos']:
            print(f"\n   📦 ITEMS ÚNICOS EN ANEXO 1:")
            for item in tabla_info['items_anexo1_unicos'][:3]:
                print(f"     • Cant={item.get('cantidad', 'N/A')} - {item.get('descripcion', 'N/A')}")
            if len(tabla_info['items_anexo1_unicos']) > 3:
                print(f"     ... y {len(tabla_info['items_anexo1_unicos']) - 3} items más")
        
        if tabla_info['items_anexo2_unicos']:
            print(f"\n   📦 ITEMS ÚNICOS EN ANEXO 2:")
            for item in tabla_info['items_anexo2_unicos'][:3]:
                print(f"     • Cant={item.get('cantidad', 'N/A')} - {item.get('descripcion', 'N/A')}")
            if len(tabla_info['items_anexo2_unicos']) > 3:
                print(f"     ... y {len(tabla_info['items_anexo2_unicos']) - 3} items más")
        
        print(f"\n💡 NOTA IMPORTANTE:")
        print(f"   {resumen['nota_importante']}")
        
    else:
        # Mostrar resultados de comparación de 2 documentos de manera CLARA
        print("=" * 60)
        print("🔍 REPORTE DE COMPARACIÓN DE DOCUMENTOS (ANEXO 1 vs ANEXO 2)")
        print("=" * 60)
        estado_simple = "✅ DOCUMENTOS CORRECTOS" if report['overall_status'] == "Correcto" else "⚠️ REQUIERE REVISIÓN"
        print(f"📋 {estado_simple}")
        print(f"📈 Confianza general: {report['confidence_score']:.0%}")
        print()
        
        print("📝 COMPARACIÓN POR CAMPOS:")
        print("-" * 40)
        for field, data in report['field_comparisons'].items():
            status_icon = "✅" if data['status'] == 'Correcto' else "⚠️"
            print(f"\n🏷️  {field.upper()}: {status_icon}")
            print(f"   📄 Anexo 1: {data['anexo1_value']}")
            print(f"   📄 Anexo 2: {data['anexo2_value']}")
            print(f"   📊 Similitud: {data['similarity_score']:.0%}")
            if args.verbose and data['status'] != 'Correcto':
                print(f"   💡 Razón: {data['reason']}")
        
        print(f"\n📊 RESUMEN DE CAMPOS:")
        print(f"   ✅ Campos correctos: {report['summary']['correcto_count']}")
        print(f"   ⚠️ Campos que requieren revisión: {report['summary']['revision_count']}")
        print(f"   📋 Total de campos: {report['summary']['total_fields']}")
        
        # Mostrar información de tabla simplificada
        table_info = report['table_comparison']
        tabla_status_icon = "✅" if table_info['status'] == 'Correcto' else "⚠️"
        print(f"\n📦 TABLA DE INSUMOS: {tabla_status_icon}")
        print(f"   📈 Similitud total: {table_info['total_similarity']:.0%}")
        print(f"   🔢 Items coincidentes: {table_info['summary']['matches']}")
        print(f"   📋 Items solo en anexo1: {table_info['summary']['anexo1_only']}")
        print(f"   📋 Items solo en anexo2: {table_info['summary']['anexo2_only']}")
        
        if args.verbose and table_info['matched_items']:
            print(f"\n   🔍 DETALLE DE ITEMS COINCIDENTES:")
            for i, match in enumerate(table_info['matched_items'][:5], 1):
                match_icon = "✅" if match['status'] == 'Correcto' else "⚠️"
                print(f"     {i}. {match_icon} (Similitud: {match['overall_similarity']:.0%})")
                print(f"        📄 Anexo1: Cant={match['anexo1_item'].get('cantidad', 'N/A')} - {match['anexo1_item'].get('descripcion', 'N/A')}")
                print(f"        📄 Anexo2: Cant={match['anexo2_item'].get('cantidad', 'N/A')} - {match['anexo2_item'].get('descripcion', 'N/A')}")
            
            if len(table_info['matched_items']) > 5:
                print(f"     ... y {len(table_info['matched_items']) - 5} items más")
        
        if table_info['anexo1_only']:
            print(f"\n   📦 ITEMS ÚNICOS EN ANEXO1:")
            for item in table_info['anexo1_only'][:3]:
                print(f"     • Cant={item.get('cantidad', 'N/A')} - {item.get('descripcion', 'N/A')}")
            if len(table_info['anexo1_only']) > 3:
                print(f"     ... y {len(table_info['anexo1_only']) - 3} items más")
        
        if table_info['anexo2_only']:
            print(f"\n   📦 ITEMS ÚNICOS EN ANEXO2:")
            for item in table_info['anexo2_only'][:3]:
                print(f"     • Cant={item.get('cantidad', 'N/A')} - {item.get('descripcion', 'N/A')}")
            if len(table_info['anexo2_only']) > 3:
                print(f"     ... y {len(table_info['anexo2_only']) - 3} items más")
    
    # Guardar reporte si se especifica
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReporte guardado en: {args.output}")
    
    return 0

if __name__ == '__main__':
    exit(main())