"""
Hybrid search implementation combining BM25 text search with vector similarity.

This module provides hybrid retrieval that leverages both lexical matching
(for exact phrases) and semantic similarity (for related concepts).

Also provides query-time entity priming: when entities are mentioned in the query,
memories linked to those entities receive a relevance boost.
"""
import logging
import math
from typing import List, Tuple, Dict, Any, Optional
from uuid import UUID
from collections import defaultdict

from rapidfuzz import fuzz

from lt_memory.entity_weights import (
    ENTITY_TYPE_WEIGHTS,
    ENTITY_BOOST_COEFFICIENT,
    MAX_ENTITY_BOOST,
    FUZZY_MATCH_THRESHOLD,
)

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Combines BM25 text search with vector similarity for optimal retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods,
    with intent-aware weighting to optimize for different query types.

    Also provides query-time entity priming: when entities are mentioned in
    the query, memories linked to those entities receive a relevance boost.
    """

    def __init__(self, db_access, entity_extractor=None):
        """
        Initialize hybrid searcher.

        Args:
            db_access: LTMemoryDB instance for database operations
            entity_extractor: Optional EntityExtractor instance for NER priming.
                              If not provided, will be lazily initialized on first use.
        """
        self.db = db_access
        self._entity_extractor = entity_extractor
        self._entity_extractor_initialized = entity_extractor is not None
        self._cached_user_entities = None
        self._cached_entities_user_id: Optional[str] = None

    @property
    def entity_extractor(self):
        """Lazy initialization of EntityExtractor for NER priming."""
        if not self._entity_extractor_initialized:
            try:
                from lt_memory.entity_extraction import EntityExtractor
                self._entity_extractor = EntityExtractor()
                self._entity_extractor_initialized = True
                logger.info("HybridSearcher: Lazily initialized EntityExtractor for NER priming")
            except Exception as e:
                logger.warning(f"Failed to initialize EntityExtractor: {e}. Entity priming disabled.")
                self._entity_extractor = None
                self._entity_extractor_initialized = True
        return self._entity_extractor

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        search_intent: str = "general",
        limit: int = 20,
        similarity_threshold: float = 0.5,
        min_importance: float = 0.1
    ) -> List[Any]:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Args:
            query_text: Text query for BM25 search
            query_embedding: Embedding for vector search
            search_intent: Intent type (recall/explore/exact/general)
            limit: Maximum results to return
            similarity_threshold: Minimum similarity for vector search
            min_importance: Minimum importance score

        Returns:
            List of Memory objects ranked by hybrid score
        """
        # Run searches in parallel (would be async in production)
        bm25_results = self._bm25_search(
            query_text,
            limit=limit * 2,  # Oversample for fusion
            min_importance=min_importance
        )

        vector_results = self._vector_search(
            query_embedding,
            limit=limit * 2,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

        # Apply intent-based weighting
        weights = {
            "recall": (0.6, 0.4),    # User trying to remember - favor exact matches
            "explore": (0.3, 0.7),   # User exploring concepts - favor semantic similarity
            "exact": (0.8, 0.2),     # User used specific phrases - strong BM25 preference
            "general": (0.4, 0.6)    # Balanced approach for ambient understanding
        }

        bm25_weight, vector_weight = weights.get(search_intent, weights["general"])

        # Combine using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight,
            limit
        )

        # Apply entity priming boost if entity extractor is available
        entity_boost_count = 0
        if self.entity_extractor and query_text:
            fused_results, entity_boost_count = self._apply_entity_priming(
                query_text, fused_results
            )

        logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"-> {len(fused_results)} fused results (intent: {search_intent}, "
            f"entity_boosts: {entity_boost_count})"
        )

        return fused_results

    def _bm25_search(
        self,
        query_text: str,
        limit: int,
        min_importance: float
    ) -> List[Tuple[Any, float]]:
        """
        Perform BM25 text search using PostgreSQL full-text search.

        Returns list of (Memory, score) tuples.
        """
        resolved_user_id = self.db._resolve_user_id()

        with self.db.session_manager.get_session(resolved_user_id) as session:
            # Use plainto_tsquery for user-friendly query parsing
            query = """
            SELECT m.*,
                   ts_rank(m.search_vector, plainto_tsquery('english', %(query)s)) as rank
            FROM memories m
            WHERE m.search_vector @@ plainto_tsquery('english', %(query)s)
              AND m.importance_score >= %(min_importance)s
              AND (m.expires_at IS NULL OR m.expires_at > NOW())
              AND m.is_archived = FALSE
            ORDER BY rank DESC
            LIMIT %(limit)s
            """

            results = session.execute_query(query, {
                'query': query_text,
                'limit': limit,
                'min_importance': min_importance
            })

            # Convert to Memory objects with scores
            from lt_memory.models import Memory
            return [(Memory(**row), row['rank']) for row in results]

    def _vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        min_importance: float
    ) -> List[Tuple[Any, float]]:
        """
        Perform vector similarity search.

        Returns list of (Memory, score) tuples.
        """
        # Reuse existing vector search
        memories = self.db.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

        # Use similarity scores calculated by database
        results = []
        for memory in memories:
            if memory.similarity_score is None:
                raise RuntimeError(
                    f"Memory {memory.id} missing similarity_score - "
                    f"this indicates db.search_similar() did not populate the transient field"
                )
            results.append((memory, memory.similarity_score))

        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Any, float]],
        vector_results: List[Tuple[Any, float]],
        bm25_weight: float,
        vector_weight: float,
        limit: int
    ) -> List[Any]:
        """
        Combine results using Reciprocal Rank Fusion (RRF) with sigmoid normalization.

        RRF formula: score(d) = Σ(1 / (k + rank(d)))
        where k is a constant (typically 60) that determines how quickly scores decay.

        Raw RRF scores are compressed into ~0.007-0.016 range which provides poor
        discrimination. We apply sigmoid transformation to spread scores into
        a useful 0-1 range for meaningful thresholding and interpretability.
        """
        k = 60  # RRF constant

        # Calculate raw RRF scores
        rrf_scores = defaultdict(float)
        memory_map = {}

        # Process BM25 results
        for rank, (memory, _) in enumerate(bm25_results, 1):
            memory_id = str(memory.id)
            rrf_scores[memory_id] += bm25_weight * (1.0 / (k + rank))
            memory_map[memory_id] = memory

        # Process vector results - preserve original cosine similarity
        for rank, (memory, cosine_sim) in enumerate(vector_results, 1):
            memory_id = str(memory.id)
            rrf_scores[memory_id] += vector_weight * (1.0 / (k + rank))
            memory._vector_similarity = cosine_sim  # Preserve for logging
            memory_map[memory_id] = memory

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply sigmoid transformation to spread scores into useful 0-1 range
        # Raw RRF scores cluster around 0.007-0.016; sigmoid with k=1000 and
        # midpoint=0.009 spreads this to ~0.1-0.85 for meaningful discrimination
        def sigmoid_normalize(raw_score: float) -> float:
            # Sigmoid: 1 / (1 + exp(-k * (x - midpoint)))
            # k=1000 provides good spread, midpoint=0.009 centers on typical RRF range
            return 1.0 / (1.0 + math.exp(-1000 * (raw_score - 0.009)))

        # Return top memories with normalized scores
        results = []
        for memory_id, raw_rrf_score in sorted_ids[:limit]:
            memory = memory_map[memory_id]
            # Store sigmoid-normalized score for interpretable thresholding
            memory.similarity_score = sigmoid_normalize(raw_rrf_score)
            memory._raw_rrf_score = raw_rrf_score  # Preserve raw for debugging
            results.append(memory)
        return results

    def _apply_entity_priming(
        self,
        query_text: str,
        memories: List[Any]
    ) -> Tuple[List[Any], int]:
        """
        Apply entity-based relevance boost to search results.

        When entities mentioned in the query match the user's known entities,
        memories linked to those entities receive a multiplicative boost.

        Args:
            query_text: User's search query
            memories: List of Memory objects from RRF fusion

        Returns:
            Tuple of (boosted memories sorted by score, count of boosted memories)
        """
        if not memories:
            return memories, 0

        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities_with_types(query_text)
        if not query_entities:
            return memories, 0

        # Match query entities to user's known entities
        matched_entities = self._match_entities_to_user(query_entities)
        if not matched_entities:
            return memories, 0

        # Apply boost to memories containing matched entities
        boosted_count = 0
        for memory in memories:
            entity_boost = self._calculate_entity_boost(memory, matched_entities)
            if entity_boost > 0:
                boost_factor = 1.0 + min(entity_boost, MAX_ENTITY_BOOST)
                if memory.similarity_score is not None:
                    memory.similarity_score *= boost_factor
                boosted_count += 1

        # Re-sort by boosted scores
        memories.sort(key=lambda m: m.similarity_score or 0, reverse=True)

        logger.debug(
            f"Entity priming: matched {len(matched_entities)} entities, "
            f"boosted {boosted_count} memories"
        )

        return memories, boosted_count

    def _match_entities_to_user(
        self,
        query_entities: List[Tuple[str, str]]
    ) -> Dict[UUID, Tuple[float, str]]:
        """
        Match extracted query entities to user's known entities.

        Uses targeted DB lookup for exact matches first, then limited fuzzy
        matching only for unmatched entities. Avoids fetching all user entities.

        Args:
            query_entities: List of (entity_name, entity_type) from query

        Returns:
            Dict mapping entity_id -> (match_confidence, entity_type)
        """
        if not query_entities:
            return {}

        matched = {}
        unmatched_queries = []

        # Step 1: Try exact match via targeted DB query
        exact_matches = self._find_exact_entity_matches(query_entities)
        for entity in exact_matches:
            matched[entity.id] = (1.0, entity.entity_type)

        # Track which query entities didn't get exact matches
        matched_names = {e.name.lower() for e in exact_matches}
        for query_name, query_type in query_entities:
            if query_name.lower() not in matched_names:
                unmatched_queries.append((query_name, query_type))

        # Step 2: For unmatched entities, try fuzzy matching against top entities
        if unmatched_queries:
            fuzzy_matches = self._find_fuzzy_entity_matches(unmatched_queries, matched)
            matched.update(fuzzy_matches)

        return matched

    def _find_exact_entity_matches(
        self,
        query_entities: List[Tuple[str, str]]
    ) -> List[Any]:
        """
        Find exact entity matches via targeted DB query.

        Args:
            query_entities: List of (entity_name, entity_type) from query

        Returns:
            List of matched Entity objects
        """
        if not query_entities:
            return []

        # Build query for exact matches on name (case-insensitive)
        entity_names = [name.lower() for name, _ in query_entities]

        resolved_user_id = self.db._resolve_user_id()

        with self.db.session_manager.get_session(resolved_user_id) as session:
            query = """
            SELECT * FROM entities
            WHERE LOWER(name) = ANY(%(names)s)
              AND is_archived = FALSE
            ORDER BY link_count DESC
            """
            results = session.execute_query(query, {'names': entity_names})

            from lt_memory.models import Entity
            return [Entity(**row) for row in results]

    def _find_fuzzy_entity_matches(
        self,
        unmatched_queries: List[Tuple[str, str]],
        already_matched: Dict[UUID, Tuple[float, str]]
    ) -> Dict[UUID, Tuple[float, str]]:
        """
        Find fuzzy entity matches for queries that didn't get exact matches.

        Only fetches top entities by link_count to limit the search space.

        Args:
            unmatched_queries: List of (entity_name, entity_type) that need fuzzy matching
            already_matched: Already matched entity IDs to skip

        Returns:
            Dict mapping entity_id -> (match_confidence, entity_type)
        """
        resolved_user_id = self.db._resolve_user_id()

        # Fetch limited set of top entities for fuzzy matching
        # Use cached entities if available for this user, otherwise fetch top 100
        if (
            self._cached_user_entities is None
            or self._cached_entities_user_id != resolved_user_id
        ):
            self._cached_user_entities = self.db.get_active_entities(
                limit=100,
                user_id=resolved_user_id,
            )
            self._cached_entities_user_id = resolved_user_id

        if not self._cached_user_entities:
            return {}

        matched = {}

        for query_name, query_type in unmatched_queries:
            query_name_lower = query_name.lower()
            best_match = None
            best_score = 0.0

            for entity in self._cached_user_entities:
                if entity.id in already_matched or entity.id in matched:
                    continue

                # Fuzzy match on name
                score = fuzz.ratio(query_name_lower, entity.name.lower()) / 100.0

                # Bonus for matching type
                if entity.entity_type == query_type:
                    score = min(1.0, score + 0.1)

                if score >= FUZZY_MATCH_THRESHOLD and score > best_score:
                    best_match = entity
                    best_score = score

            if best_match:
                matched[best_match.id] = (best_score, best_match.entity_type)

        return matched

    def _calculate_entity_boost(
        self,
        memory: Any,
        matched_entities: Dict[UUID, Tuple[float, str]]
    ) -> float:
        """
        Calculate entity boost for a memory based on matched entities.

        Args:
            memory: Memory object with entity_links
            matched_entities: Dict of entity_id -> (confidence, type)

        Returns:
            Total boost factor (before capping)
        """
        if not memory.entity_links:
            return 0.0

        total_boost = 0.0

        for entity_link in memory.entity_links:
            try:
                entity_id = UUID(entity_link.get('uuid', ''))
            except (ValueError, TypeError):
                continue

            if entity_id in matched_entities:
                confidence, entity_type = matched_entities[entity_id]
                type_weight = ENTITY_TYPE_WEIGHTS.get(entity_type, 0.5)
                total_boost += confidence * type_weight * ENTITY_BOOST_COEFFICIENT

        return total_boost

    def clear_entity_cache(self):
        """Clear cached entities and user-scoping metadata."""
        self._cached_user_entities = None
        self._cached_entities_user_id = None
