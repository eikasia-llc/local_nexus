"""
Query router for classifying user queries into structured, unstructured, or hybrid types.

This module determines the optimal retrieval strategy based on query characteristics:
- STRUCTURED: SQL queries over DuckDB tables (aggregations, counts, filters)
- UNSTRUCTURED: Semantic search over documents (policy questions, explanations)
- HYBRID: Both structured and unstructured (e.g., "customers who mentioned X with >$1K value")
"""

from enum import Enum
from typing import Optional
import re


class QueryType(Enum):
    """Classification of query types for routing."""
    STRUCTURED = "structured"      # SQL/tabular data queries
    UNSTRUCTURED = "unstructured"  # Semantic/document queries
    HYBRID = "hybrid"              # Requires both retrieval paths


class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies using keyword heuristics.

    The router uses a two-phase classification:
    1. Fast keyword-based heuristics (no LLM cost)
    2. Optional LLM fallback for ambiguous cases

    Design principles:
    - Default to UNSTRUCTURED for ambiguous queries (safer, more general)
    - Use STRUCTURED when clear quantitative/aggregation signals present
    - Use HYBRID when both semantic and computational signals detected
    """

    # Keywords strongly indicating structured/SQL queries
    STRUCTURED_KEYWORDS = {
        # Aggregation operations
        'count', 'sum', 'total', 'average', 'avg', 'mean', 'median',
        'maximum', 'max', 'minimum', 'min', 'percent', 'percentage',

        # Quantitative terms
        'how many', 'how much', 'number of', 'amount of', 'quantity',
        'total revenue', 'total sales', 'total cost', 'total value',

        # Time-based aggregations
        'per month', 'per year', 'per day', 'per week', 'per quarter',
        'quarterly', 'monthly', 'yearly', 'daily', 'weekly',
        'by month', 'by year', 'by quarter',

        # Comparison/ranking
        'top', 'bottom', 'highest', 'lowest', 'most', 'least', 'rank',
        'compare', 'comparison', 'versus', 'vs',

        # Numeric comparisons (without semantic context)
        'greater than', 'less than', 'more than', 'at least', 'at most',

        # Data operations
        'filter', 'group by', 'grouped', 'sorted', 'sort by', 'order by',
        'distinct', 'unique', 'duplicate',

        # Table references
        'table', 'column', 'row', 'rows', 'record', 'records',
        'field', 'fields', 'dataset',
    }

    # Keywords strongly indicating unstructured/semantic queries
    UNSTRUCTURED_KEYWORDS = {
        # Document/policy questions
        'policy', 'guideline', 'procedure', 'process', 'rule', 'regulation',
        'document', 'docs', 'documentation', 'manual', 'handbook',
        'report', 'annual report',

        # Explanation requests
        'explain', 'describe', 'what is', 'what are', 'what does',
        'how does', 'how do', 'why', 'definition', 'meaning',

        # Content search
        'find information', 'search for', 'look for', 'tell me about',
        'information about', 'details about', 'summary of', 'overview',

        # Qualitative terms
        'sentiment', 'opinion', 'feedback', 'review', 'comment',
        'complaint', 'praise', 'mention', 'said', 'wrote',
    }

    # Keywords indicating hybrid queries (both structured + unstructured)
    # These require BOTH semantic understanding AND data computation
    HYBRID_KEYWORDS = {
        # Semantic + aggregation combinations
        'who mentioned', 'that mentioned', 'who said', 'who complained',
        'who wrote', 'mentioned pricing', 'mentioned concerns',
        'customers who', 'users who', 'clients who',
        'complaints about', 'feedback about', 'reviews mentioning',

        # Semantic filter + data
        'with value', 'and have', 'and spent',

        # Analysis across data types
        'correlate', 'relationship between', 'compare sentiment',
        'trends in feedback', 'analyze comments',
    }

    def __init__(self, llm_fallback: bool = False, llm_func: Optional[callable] = None):
        """
        Initialize the query router.

        Args:
            llm_fallback: Whether to use LLM for ambiguous queries
            llm_func: Optional function for LLM classification (signature: str -> QueryType)
        """
        self.llm_fallback = llm_fallback
        self.llm_func = llm_func

    def _normalize_query(self, query: str) -> str:
        """Normalize query for keyword matching."""
        return query.lower().strip()

    def _calculate_keyword_scores(self, query: str) -> dict[str, float]:
        """
        Calculate confidence scores for each query type based on keyword matches.

        Returns:
            Dict with scores for 'structured', 'unstructured', 'hybrid'
        """
        normalized = self._normalize_query(query)

        scores = {
            'structured': 0.0,
            'unstructured': 0.0,
            'hybrid': 0.0,
        }

        # Check for hybrid keywords first (multi-word patterns)
        for keyword in self.HYBRID_KEYWORDS:
            if keyword in normalized:
                scores['hybrid'] += 2.0  # Higher weight for hybrid patterns

        # Check structured keywords
        for keyword in self.STRUCTURED_KEYWORDS:
            if keyword in normalized:
                scores['structured'] += 1.0

        # Check unstructured keywords
        for keyword in self.UNSTRUCTURED_KEYWORDS:
            if keyword in normalized:
                scores['unstructured'] += 1.0

        # Additional heuristics

        # Numbers with comparison operators suggest structured
        if re.search(r'[<>=]\s*\d+|>\s*\$?\d+|\d+\s*%', normalized):
            scores['structured'] += 1.5

        # Questions starting with "how many/much" are typically structured
        if re.match(r'^how\s+(many|much)', normalized):
            scores['structured'] += 2.0

        # "What is the total/sum/count" patterns are structured despite "what is"
        if re.search(r'what\s+(is|are)\s+(the\s+)?(total|sum|count|average|number)', normalized):
            scores['structured'] += 3.0  # Strong structured signal

        # Questions about specific entities with semantic context (hybrid)
        # More flexible pattern: "customers/users mentioned" or "customers who mentioned"
        if re.search(r'(customer|user|client)s?\s+(who\s+)?(mention|said|wrote|complain)', normalized):
            scores['hybrid'] += 2.0

        # "mentioned X and have/spent" pattern indicates hybrid
        if re.search(r'mention\w*\s+\w+.*\s+(and|with)\s+(have|spent|value|>|<)', normalized):
            scores['hybrid'] += 2.0

        # Pure "what is/are" questions WITHOUT aggregation terms are unstructured
        if re.match(r'^what\s+(is|are|does)', normalized):
            # Only add unstructured score if no strong structured keywords
            if not any(kw in normalized for kw in ['total', 'sum', 'count', 'average', 'number of']):
                scores['unstructured'] += 1.5

        # Questions about policies/procedures are unstructured
        if 'policy' in normalized or 'procedure' in normalized:
            scores['unstructured'] += 2.0

        return scores

    def _classify_from_scores(self, scores: dict[str, float], confidence_threshold: float = 1.5) -> tuple[QueryType, float]:
        """
        Determine query type from keyword scores.

        Args:
            scores: Dict of scores per query type
            confidence_threshold: Minimum score difference to be confident

        Returns:
            Tuple of (QueryType, confidence_score)
        """
        # Find the highest scoring type
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        # Calculate confidence (difference from second highest)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            confidence = sorted_scores[0] - sorted_scores[1]
        else:
            confidence = max_score

        # If hybrid has any significant score and structured also has score,
        # prefer hybrid (it's the more complete solution)
        if scores['hybrid'] >= 2.0 and scores['structured'] >= 1.0:
            return QueryType.HYBRID, scores['hybrid']

        # Map string to enum
        type_map = {
            'structured': QueryType.STRUCTURED,
            'unstructured': QueryType.UNSTRUCTURED,
            'hybrid': QueryType.HYBRID,
        }

        return type_map[max_type], confidence

    def classify(self, query: str) -> QueryType:
        """
        Classify a query into STRUCTURED, UNSTRUCTURED, or HYBRID.

        Args:
            query: User's natural language query

        Returns:
            QueryType enum value
        """
        if not query or not query.strip():
            return QueryType.UNSTRUCTURED  # Default for empty queries

        scores = self._calculate_keyword_scores(query)
        query_type, confidence = self._classify_from_scores(scores)

        # If confidence is low and LLM fallback is enabled, use LLM
        if confidence < 1.0 and self.llm_fallback and self.llm_func:
            try:
                return self.llm_func(query)
            except Exception:
                pass  # Fall through to heuristic result

        # Default to unstructured if no strong signals
        if max(scores.values()) == 0:
            return QueryType.UNSTRUCTURED

        return query_type

    def classify_with_confidence(self, query: str) -> tuple[QueryType, float, dict]:
        """
        Classify a query and return confidence details.

        Args:
            query: User's natural language query

        Returns:
            Tuple of (QueryType, confidence_score, detailed_scores)
        """
        if not query or not query.strip():
            return QueryType.UNSTRUCTURED, 0.0, {'structured': 0, 'unstructured': 0, 'hybrid': 0}

        scores = self._calculate_keyword_scores(query)
        query_type, confidence = self._classify_from_scores(scores)

        # Default to unstructured if no strong signals
        if max(scores.values()) == 0:
            return QueryType.UNSTRUCTURED, 0.0, scores

        return query_type, confidence, scores

    def get_routing_explanation(self, query: str) -> str:
        """
        Generate a human-readable explanation of the routing decision.

        Args:
            query: User's natural language query

        Returns:
            Explanation string
        """
        query_type, confidence, scores = self.classify_with_confidence(query)

        explanations = {
            QueryType.STRUCTURED: "This query appears to involve data aggregation, counting, or table operations.",
            QueryType.UNSTRUCTURED: "This query appears to be asking about document content, policies, or explanations.",
            QueryType.HYBRID: "This query requires both semantic search and data computation.",
        }

        return (
            f"Query Type: {query_type.value.upper()}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Scores: structured={scores['structured']:.1f}, "
            f"unstructured={scores['unstructured']:.1f}, "
            f"hybrid={scores['hybrid']:.1f}\n"
            f"Reason: {explanations[query_type]}"
        )


def create_llm_classifier(llm_func: callable) -> callable:
    """
    Create an LLM-based classifier function for ambiguous queries.

    Args:
        llm_func: Function that takes a prompt and returns LLM response

    Returns:
        Classifier function that returns QueryType
    """
    import re as regex

    def classifier(query: str) -> QueryType:
        prompt = f"""Classify this query into one category:
- STRUCTURED: Requires SQL/database queries (counts, sums, filters, aggregations)
- UNSTRUCTURED: Requires document/semantic search (policies, explanations, content)
- HYBRID: Requires both (semantic filters + data computation)

Query: "{query}"

Respond with only one word: STRUCTURED, UNSTRUCTURED, or HYBRID"""

        response = llm_func(prompt).strip().upper()

        # Check for exact word matches using word boundaries
        # Important: Check HYBRID and UNSTRUCTURED before STRUCTURED
        # because "UNSTRUCTURED" contains "STRUCTURED" as substring
        if regex.search(r'\bHYBRID\b', response):
            return QueryType.HYBRID
        elif regex.search(r'\bUNSTRUCTURED\b', response):
            return QueryType.UNSTRUCTURED
        elif regex.search(r'\bSTRUCTURED\b', response):
            return QueryType.STRUCTURED
        else:
            return QueryType.UNSTRUCTURED  # Default

    return classifier


if __name__ == "__main__":
    # Quick test
    router = QueryRouter()

    test_queries = [
        "How many customers do we have?",
        "What is our PTO policy?",
        "Which customers mentioned pricing concerns and have >$10K lifetime value?",
        "Show me total sales by month",
        "Explain the refund process",
        "Top 5 products by revenue",
        "What do the reviews say about shipping?",
    ]

    print("Query Classification Examples:")
    print("=" * 60)
    for query in test_queries:
        result = router.classify(query)
        print(f"\nQuery: {query}")
        print(f"Type: {result.value}")
