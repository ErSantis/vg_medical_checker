"""
Matcher Module - Comparación de documentos médicos

Este módulo contiene la lógica para comparar documentos anexo1 y anexo2
usando fuzzy matching y sistema de sinónimos para productos médicos.
"""

from .document_matcher import (
    DocumentMatcher,
    DocumentComparison,
    MatchResult,
    TableItemMatch,
    TableComparison,
    load_synonyms,
    find_synonym_match,
    are_products_synonymous
)

__all__ = [
    'DocumentMatcher',
    'DocumentComparison', 
    'MatchResult',
    'TableItemMatch',
    'TableComparison',
    'load_synonyms',
    'find_synonym_match',
    'are_products_synonymous'
]