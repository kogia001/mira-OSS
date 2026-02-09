"""
Proactive memory surfacing for CNS integration.

Provides intelligent memory search using pre-computed fingerprint embeddings
and automatic inclusion of linked memories for context enrichment.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from config.config import ProactiveConfig
from lt_memory.db_access import LTMemoryDB
from lt_memory.linking import LinkingService
from lt_memory.vector_ops import VectorOps

logger = logging.getLogger(__name__)


class ProactiveService:
    """
    Service for proactive memory surfacing in conversations.

    Finds relevant memories using pre-computed fingerprint embeddings
    and automatically includes linked memories for richer context.
    """

    def __init__(
        self,
        config: ProactiveConfig,
        vector_ops: VectorOps,
        linking_service: LinkingService,
        db: LTMemoryDB
    ):
        """
        Initialize proactive service.

        Args:
            config: Proactive surfacing configuration
            vector_ops: Vector operations service
            linking_service: Memory linking service
            db: Database access layer
        """
        self.config = config
        self.vector_ops = vector_ops
        self.linking = linking_service
        self.db = db

        logger.debug("ProactiveService initialized")

    def search_with_embedding(
        self,
        embedding: np.ndarray,
        fingerprint: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using pre-computed fingerprint embedding.

        The fingerprint is a retrieval-optimized query expansion that replaces
        the original user message for better embedding similarity matching.

        Args:
            embedding: Pre-computed 768d fingerprint embedding
            fingerprint: Expanded query text (for BM25 and reranking)
            limit: Maximum number of memories to return

        Returns:
            List of relevant memory dictionaries with metadata

        Raises:
            Exception: If search operations fail
        """
        if limit is None:
            limit = self.config.max_memories

        # Hybrid search using fingerprint as query text
        search_results = self.vector_ops.hybrid_search(
            query_text=fingerprint,
            query_embedding=embedding,
            search_intent="general",  # Fingerprint is already optimized
            limit=limit * 2,  # Oversample for filtering
            similarity_threshold=self.config.similarity_threshold,
            min_importance=self.config.min_importance_score
        )

        primary_results = search_results[:limit]

        if not primary_results:
            logger.debug("No relevant memories found")
            return []

        # Include linked memories for context enrichment
        expanded_results = self._include_linked_memories(primary_results)

        # Rerank and filter linked memories by type, confidence, importance
        reranked_results = self._rerank_with_links(expanded_results)
        final_results = reranked_results[:limit]

        logger.info(
            f"Found {len(final_results)} relevant memories "
            f"({len(primary_results)} primary)"
        )

        # Track access for retrieved memories to update importance scores
        self._track_memory_access(final_results)

        return [self._memory_to_dict(m) for m in final_results]

    def _track_memory_access(self, memories: List[Any]) -> None:
        """
        Track access for retrieved memories to update importance scores.

        Updates access_count, last_accessed, and recalculates importance
        scores for each retrieved memory. Only tracks primary memories,
        not linked memories (which are secondary context).

        Args:
            memories: List of Memory objects that were retrieved
        """
        for memory in memories:
            try:
                self.db.update_access_stats(memory.id)
            except Exception as e:
                # Log but don't fail the search - access tracking is enhancement
                logger.warning(
                    f"Failed to update access stats for memory {memory.id}: {e}"
                )

    def _include_linked_memories(
        self,
        primary_memories: List[Any]
    ) -> List[Any]:
        """
        Include memories linked to primary search results.

        Traverses memory graph and attaches related memories as children
        of primary memories, preserving link metadata.

        Args:
            primary_memories: List of primary memory search results

        Returns:
            List of primary memories with linked_memories populated
        """
        if not primary_memories:
            return []

        for primary_memory in primary_memories:
            linked_with_metadata = self.linking.traverse_related(
                memory_id=primary_memory.id,
                depth=self.config.max_link_traversal_depth
            )

            primary_memory.linked_memories = []

            for linked_data in linked_with_metadata:
                linked_memory = linked_data["memory"]
                linked_memory.link_metadata = {
                    "link_type": linked_data["link_type"],
                    "confidence": linked_data["confidence"],
                    "reasoning": linked_data["reasoning"],
                    "depth": linked_data["depth"],
                    "linked_from_id": linked_data["linked_from_id"]
                }
                primary_memory.linked_memories.append(linked_memory)

            logger.debug(
                f"Attached {len(primary_memory.linked_memories)} linked memories "
                f"to primary memory {primary_memory.id}"
            )

        return primary_memories

    def _rerank_with_links(self, primary_memories: List[Any]) -> List[Any]:
        """
        Rerank and filter memories considering link types, confidence, importance.

        Ranking formula: type_weight × inherited_importance × confidence
        """
        LINK_TYPE_WEIGHTS = {
            "conflicts": 1.0,
            "invalidated_by": 1.0,
            "supersedes": 0.9,
            "causes": 0.8,
            "motivated_by": 0.8,
            "instance_of": 0.7,
            "shares_entity": 0.4,
        }
        MIN_CONFIDENCE = 0.6

        primary_ids = {str(m.id) for m in primary_memories}

        for primary_memory in primary_memories:
            if not hasattr(primary_memory, 'linked_memories'):
                continue

            linked_memories = primary_memory.linked_memories
            if not linked_memories:
                continue

            scored_linked = []

            for linked in linked_memories:
                # Deduplication
                if str(linked.id) in primary_ids:
                    continue

                link_meta = getattr(linked, 'link_metadata', {})
                link_type = link_meta.get('link_type', 'unknown')
                confidence = link_meta.get('confidence')

                # Confidence filtering
                if confidence is not None and confidence < MIN_CONFIDENCE:
                    continue

                # Type-based weighting
                type_weight = 0.5
                for known_type, weight in LINK_TYPE_WEIGHTS.items():
                    if link_type == known_type or link_type.startswith(f"{known_type}:"):
                        type_weight = weight
                        break

                # Importance inheritance
                linked_importance = getattr(linked, 'importance_score', 0.5)
                primary_importance = getattr(primary_memory, 'importance_score', 0.5)
                inherited_importance = (linked_importance * 0.7) + (primary_importance * 0.3)

                final_score = type_weight * inherited_importance * (confidence or 1.0)
                scored_linked.append((linked, final_score))

            scored_linked.sort(key=lambda x: x[1], reverse=True)
            primary_memory.linked_memories = [linked for linked, score in scored_linked]

        return primary_memories

    def _memory_to_dict(self, memory) -> Dict[str, Any]:
        """Convert Memory model to dictionary with hierarchical structure."""
        result = {
            "id": str(memory.id),
            "text": memory.text,
            "importance_score": memory.importance_score,
            "similarity_score": memory.similarity_score,  # Sigmoid-normalized RRF score (0-1)
            "vector_similarity": getattr(memory, '_vector_similarity', None),  # Raw cosine similarity
            "_raw_rrf_score": getattr(memory, '_raw_rrf_score', None),  # Raw RRF before sigmoid
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "access_count": memory.access_count,
            "happens_at": memory.happens_at.isoformat() if memory.happens_at else None,
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
            "inbound_links": memory.inbound_links if hasattr(memory, 'inbound_links') else [],
            "outbound_links": memory.outbound_links if hasattr(memory, 'outbound_links') else [],
        }

        if hasattr(memory, 'link_metadata'):
            result['link_metadata'] = memory.link_metadata

        if hasattr(memory, 'linked_memories') and memory.linked_memories:
            result['linked_memories'] = [
                self._memory_to_dict(linked)
                for linked in memory.linked_memories
            ]
        else:
            result['linked_memories'] = []

        return result
