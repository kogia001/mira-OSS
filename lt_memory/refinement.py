"""
Memory refinement service for LT_Memory system.

Handles two refinement operations:
1. Verbose trimming: Distilling long memories into concise core facts
2. Consolidation clustering: Identifying and merging redundant similar memories

Both operations preserve importance and critical details while improving
memory system quality over time.
"""
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID

from lt_memory.models import (
    Memory,
    ExtractedMemory,
    RefinementCandidate,
    ConsolidationCluster
)
from config.config import RefinementConfig
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB
from clients.llm_provider import LLMProvider
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class RefinementService:
    """
    Service for refining memories through consolidation and verbose trimming.

    Provides:
    - Verbose memory identification and refinement
    - Consolidation cluster discovery
    - LLM-driven memory distillation
    - Prompt building for batch consolidation
    """

    def __init__(
        self,
        config: RefinementConfig,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        llm_provider: Optional[LLMProvider] = None
    ):
        """
        Initialize refinement service.

        Args:
            config: Refinement configuration
            vector_ops: Vector operations for similarity search
            db: Database access layer
            llm_provider: LLM provider for sync refinement (optional)
        """
        self.config = config
        self.vector_ops = vector_ops
        self.db = db
        self.llm_provider = llm_provider
        self._load_prompts()

    def _load_prompts(self):
        """
        Load refinement and consolidation prompts.

        Raises:
            FileNotFoundError: If any required prompt file not found (prompts are required configuration)
        """
        prompts_dir = Path("config/prompts")

        # Verbose trimming system prompt
        refinement_system_path = prompts_dir / "memory_refinement_system.txt"
        if not refinement_system_path.exists():
            raise FileNotFoundError(
                f"Required prompt file not found: {refinement_system_path}. "
                f"Prompts are system configuration, not optional features."
            )
        with open(refinement_system_path, 'r', encoding='utf-8') as f:
            self.refinement_system_prompt = f.read().strip()

        # Verbose trimming user template
        refinement_user_path = prompts_dir / "memory_refinement_user.txt"
        if not refinement_user_path.exists():
            raise FileNotFoundError(
                f"Required prompt file not found: {refinement_user_path}. "
                f"Prompts are system configuration, not optional features."
            )
        with open(refinement_user_path, 'r', encoding='utf-8') as f:
            self.refinement_user_template = f.read().strip()

        # Consolidation prompt
        consolidation_system_path = prompts_dir / "memory_consolidation_system.txt"
        if not consolidation_system_path.exists():
            raise FileNotFoundError(
                f"Required prompt file not found: {consolidation_system_path}. "
                f"Prompts are system configuration, not optional features."
            )
        with open(consolidation_system_path, 'r', encoding='utf-8') as f:
            self.consolidation_system_prompt = f.read().strip()

        logger.info("Loaded refinement prompts")

    def identify_verbose_memories(
        self,
        limit: int = 10
    ) -> List[RefinementCandidate]:
        """
        Identify verbose memories exceeding character threshold.

        Looks for memories that:
        - Exceed configured character threshold
        - Have been accessed multiple times (stable/useful)
        - Haven't been recently refined
        - Are at least 7 days old

        Args:
            limit: Maximum candidates to return

        Returns:
            List of RefinementCandidate objects
        """
        # Get all active memories
        all_memories = self.db.get_all_memories(include_archived=False)

        candidates = []
        min_age_threshold = utc_now() - timedelta(days=self.config.min_age_for_refinement_days)

        for memory in all_memories:
            # Skip if too short
            if len(memory.text) < self.config.verbose_threshold_chars:
                continue

            # Skip if recently refined
            if memory.is_refined and memory.last_refined_at:
                cooldown = timedelta(days=self.config.refinement_cooldown_days)
                if memory.last_refined_at > utc_now() - cooldown:
                    continue

            # Skip if too new
            if memory.created_at > min_age_threshold:
                continue

            # Skip if not accessed enough (not stable/useful)
            if memory.access_count < self.config.min_access_count_for_refinement:
                continue

            # Skip if rejected max times (system has decided it's fine as-is)
            if memory.refinement_rejection_count >= self.config.max_rejection_count:
                continue

            candidate = RefinementCandidate(
                memory_id=memory.id,
                reason="verbose",
                current_text=memory.text,
                char_count=len(memory.text),
                target_memory_ids=[],
                similarity_scores=[]
            )

            candidates.append(candidate)

        # Sort by character count descending
        candidates.sort(key=lambda c: c.char_count, reverse=True)

        logger.info(f"Identified {len(candidates[:limit])} verbose memories for refinement")

        return candidates[:limit]

    def identify_consolidation_clusters(
        self,
        min_cluster_size: Optional[int] = None,
        max_cluster_size: Optional[int] = None
    ) -> List[ConsolidationCluster]:
        """
        Identify clusters of similar memories for consolidation using hub-based approach.

        Instead of checking all memories, finds high-value "hub" memories
        (high importance, frequent access) and clusters around them.
        More efficient than exhaustive search.

        Args:
            min_cluster_size: Minimum memories in cluster (default from config)
            max_cluster_size: Maximum memories in cluster (default from config)

        Returns:
            List of ConsolidationCluster objects
        """
        if min_cluster_size is None:
            min_cluster_size = self.config.min_cluster_size

        if max_cluster_size is None:
            max_cluster_size = self.config.max_cluster_size

        # Get all active memories sorted by importance and access
        all_memories = self.db.get_all_memories(include_archived=False)

        # Filter to hub candidates:
        # - High importance with frequent access, OR
        # - Well-connected (5+ inbound relationship links)
        hub_memories = [
            m for m in all_memories
            if (m.importance_score >= 0.3 and m.access_count >= 5) or
               len(m.inbound_links) >= 5
        ]

        # Sort by importance descending, take top 50
        hub_memories.sort(key=lambda m: m.importance_score, reverse=True)
        hub_memories = hub_memories[:50]

        logger.info(f"Found {len(hub_memories)} hub candidates for clustering")

        # Build clusters using similarity search
        processed_ids = set()
        clusters = []

        for memory in hub_memories:
            if memory.id in processed_ids:
                continue

            # Find similar memories
            similar_memories = self.vector_ops.find_similar_to_memory(
                memory_id=memory.id,
                limit=max_cluster_size,
                similarity_threshold=self.config.consolidation_similarity_threshold,
                min_importance=0.001  # Filter cold storage (0.0) memories
            )

            if len(similar_memories) < (min_cluster_size - 1):
                # Not enough similar memories to form cluster
                continue

            # Build cluster
            cluster_memories = [memory] + similar_memories[:max_cluster_size - 1]
            cluster_ids = [m.id for m in cluster_memories]
            cluster_texts = [m.text for m in cluster_memories]

            # Calculate similarity scores
            similarity_scores = [
                m.similarity_score if m.similarity_score is not None
                else self.config.consolidation_similarity_threshold
                for m in similar_memories[:max_cluster_size - 1]
            ]

            avg_similarity = (
                sum(similarity_scores) / len(similarity_scores)
                if similarity_scores else 0.0
            )

            # Create cluster
            cluster = ConsolidationCluster(
                cluster_id=f"cluster_{memory.id}",
                memory_ids=cluster_ids,
                memory_texts=cluster_texts,
                similarity_scores=similarity_scores,
                avg_similarity=avg_similarity,
                consolidation_confidence=avg_similarity  # Use avg similarity as confidence
            )

            # Only include if meets confidence threshold
            if cluster.consolidation_confidence >= self.config.consolidation_confidence_threshold:
                clusters.append(cluster)

                # Mark all cluster members as processed
                processed_ids.update(cluster_ids)

        logger.info(f"Identified {len(clusters)} consolidation clusters")

        return clusters

    def build_consolidation_payload(
        self,
        cluster: ConsolidationCluster
    ) -> Dict[str, Any]:
        """
        Build consolidation analysis payload for batch API.

        Args:
            cluster: ConsolidationCluster to analyze

        Returns:
            Dictionary with prompt and parameters
        """
        # User prompt contains ONLY the memories to analyze
        memories_text = "\n\n".join([
            f"Memory {i+1} (ID: {memory_id}):\n{text}"
            for i, (memory_id, text) in enumerate(zip(cluster.memory_ids, cluster.memory_texts))
        ])

        user_prompt = f"""Analyze these similar memories and determine if consolidation would provide clear improvement:

{memories_text}

Respond with JSON:
{{
    "should_consolidate": true/false,
    "consolidated_text": "Combined memory text if consolidating (or empty string)",
    "reason": "Brief explanation of decision"
}}"""

        return {
            "cluster_id": cluster.cluster_id,
            "memory_ids": [str(mid) for mid in cluster.memory_ids],
            "system_prompt": self.consolidation_system_prompt,  # From file
            "user_prompt": user_prompt  # Data only
        }

    def refine_verbose_memory_sync(
        self,
        memory: Memory
    ) -> Dict[str, Any]:
        """
        Synchronously refine a verbose memory.

        Uses LLM to trim, split, or mark as do_nothing.
        Use sparingly - prefer batch processing for efficiency.

        Args:
            memory: Memory to refine

        Returns:
            Dict with:
                - action: "trim"|"split"|"do_nothing"
                - refined_memories: List[ExtractedMemory]
                - rejection_count_increment: int

        Raises:
            RuntimeError: If LLM provider not configured or LLM call fails
            ValueError: If LLM response cannot be parsed
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider required for sync refinement")

        user_prompt = self.refinement_user_template.format(
            memory_text=memory.text
        )

        response = self.llm_provider.generate_response(
            messages=[
                {"role": "system", "content": self.refinement_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=self.config.refinement_max_tokens,
            top_p=0.2
        )

        response_text = self.llm_provider.extract_text_content(response)

        # Parse refinement response
        try:
            refinement = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse refinement response: {e}") from e

        action = refinement.get("action", "do_nothing")

        if action == "trim":
            refined_text = refinement.get("refined_text", "").strip()
            if not refined_text:
                logger.warning("Trim action produced empty text, treating as do_nothing")
                return {
                    "action": "do_nothing",
                    "refined_memories": [],
                    "rejection_count_increment": 1
                }

            return {
                "action": "trim",
                "refined_memories": [ExtractedMemory(
                    text=refined_text,
                    importance_score=memory.importance_score,
                    expires_at=memory.expires_at,
                    happens_at=memory.happens_at,
                    confidence=refinement.get("confidence", 0.95),
                    consolidates_memory_ids=[str(memory.id)]
                )],
                "rejection_count_increment": 0
            }

        elif action == "split":
            split_texts = refinement.get("split_memories", [])
            if not split_texts or len(split_texts) < 2:
                logger.warning("Split action with <2 memories, treating as do_nothing")
                return {
                    "action": "do_nothing",
                    "refined_memories": [],
                    "rejection_count_increment": 1
                }

            return {
                "action": "split",
                "refined_memories": [
                    ExtractedMemory(
                        text=text.strip(),
                        importance_score=memory.importance_score,
                        expires_at=memory.expires_at,
                        happens_at=memory.happens_at,
                        confidence=refinement.get("confidence", 0.90),
                        consolidates_memory_ids=[str(memory.id)]
                    )
                    for text in split_texts if text.strip()
                ],
                "rejection_count_increment": 0
            }

        else:  # do_nothing
            logger.info(f"Memory {memory.id} marked do_nothing: {refinement.get('reason')}")
            return {
                "action": "do_nothing",
                "refined_memories": [],
                "rejection_count_increment": 1
            }

    def run_full_refinement(
        self
    ) -> Dict[str, int]:
        """
        Run complete refinement pass for a user.

        Identifies both verbose memories and consolidation clusters.
        Returns statistics for APScheduler job logging.

        Returns:
            Statistics dictionary with counts
        """
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()

        stats = {
            "verbose_candidates": 0,
            "consolidation_clusters": 0
        }

        # Identify verbose memories
        verbose_candidates = self.identify_verbose_memories(
            limit=self.config.verbose_candidates_limit
        )
        stats["verbose_candidates"] = len(verbose_candidates)

        # Identify consolidation clusters
        consolidation_clusters = self.identify_consolidation_clusters()
        stats["consolidation_clusters"] = len(consolidation_clusters)

        logger.info(
            f"Refinement analysis complete for user {user_id}: "
            f"{stats['verbose_candidates']} verbose memories, "
            f"{stats['consolidation_clusters']} consolidation clusters"
        )

        return stats

    def cleanup(self):
        """
        Clean up resources.

        No-op: Dependencies managed by factory lifecycle.
        Nulling references breaks in-flight scheduler jobs.
        """
        logger.debug("RefinementService cleanup completed (no-op)")
