"""
Pydantic models for LT_Memory system.

All data structures for memories, links, batches, and processing chunks.
"""
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID


class Memory(BaseModel):
    """
    Represents a stored memory from database.

    Returned by db_access methods for type-safe memory operations.
    """
    id: UUID
    user_id: UUID
    text: str
    embedding: Optional[List[float]] = None  # mdbr-leaf-ir-asym (768d)
    importance_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    access_count: int = 0
    mention_count: int = 0  # Explicit LLM references (strongest importance signal)
    last_accessed: Optional[datetime] = None
    happens_at: Optional[datetime] = None
    inbound_links: List[Dict[str, Any]] = Field(default_factory=list)
    outbound_links: List[Dict[str, Any]] = Field(default_factory=list)
    entity_links: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    is_refined: bool = False
    last_refined_at: Optional[datetime] = None
    refinement_rejection_count: int = 0

    # Activity day snapshots for vacation-proof scoring
    activity_days_at_creation: Optional[int] = None
    activity_days_at_last_access: Optional[int] = None

    # Transient field populated by similarity search queries
    similarity_score: Optional[float] = None

    # Transient fields populated by proactive service during link traversal
    linked_memories: Optional[List[Any]] = Field(default=None, exclude=True)
    link_metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)



class ExtractedMemory(BaseModel):
    """
    Memory extracted from continuum chunk.

    Used during extraction pipeline before persistence.
    """
    text: str
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    expires_at: Optional[datetime] = None
    happens_at: Optional[datetime] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    relationship_type: Optional[str] = None
    related_memory_ids: List[str] = Field(default_factory=list)
    consolidates_memory_ids: List[str] = Field(default_factory=list)
    linking_hints: List[int] = Field(
        default_factory=list,
        description="Indices of other memories in this batch to consider for relationship classification"
    )

    @field_validator('importance_score', 'confidence')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v


class ExtractionResult(BaseModel):
    """
    Result of memory extraction containing memories and linking hints.

    Used to pass both extracted memories and intra-batch linking hints
    through the extraction pipeline.
    """
    memories: List['ExtractedMemory']
    linking_pairs: List[tuple[int, int]] = Field(
        default_factory=list,
        description="Pairs of memory indices that should be evaluated for relationships"
    )


class MemoryLink(BaseModel):
    """
    Relationship link between memories.

    Stored bidirectionally in memory inbound_links/outbound_links JSONB arrays.
    """
    source_id: UUID
    target_id: UUID
    # Supports current and legacy relationship labels.
    # Also allows entity-style link prefixes (shares_entity:TYPE).
    link_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    created_at: datetime

    @field_validator('link_type')
    @classmethod
    def validate_link_type(cls, v: str) -> str:
        """Validate link type matches relationship classification types."""
        allowed = {
            "conflicts",
            "supports",
            "supersedes",
            "related",
            "causes",
            "instance_of",
            "invalidated_by",
            "motivated_by",
            "shares_entity",
        }
        if v not in allowed and not v.startswith("shares_entity:"):
            raise ValueError(
                "link_type must be one of "
                f"{allowed} or start with 'shares_entity:', got '{v}'"
            )
        return v


class Entity(BaseModel):
    """
    Persistent knowledge anchor (entity) that memories link to.

    Entities represent named entities (people, organizations, products, events, etc.)
    extracted from memory text. They serve as knowledge graph nodes enabling
    entity-based memory retrieval and relationship discovery.
    """
    id: UUID
    user_id: UUID
    name: str = Field(description="Canonical normalized entity name")
    entity_type: str = Field(description="spaCy NER type: PERSON, ORG, GPE, PRODUCT, EVENT, etc.")
    embedding: Optional[List[float]] = Field(
        default=None,
        description="spaCy word vector (300d from en_core_web_lg) for semantic similarity"
    )
    link_count: int = Field(default=0, description="Number of memories linking to this entity")
    last_linked_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of most recent memory link (for dormancy detection)"
    )
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_archived: bool = False
    archived_at: Optional[datetime] = None

    # Transient field populated by similarity search queries
    similarity_score: Optional[float] = None


class ProcessingChunk(BaseModel):
    """
    Ephemeral continuum chunk for batch extraction processing.

    Temporary container that holds messages and metadata during batch
    submission orchestration. Discarded after batch request is built.
    Holds Message objects directly (no conversion to dict).
    """
    messages: List[Any]  # Message objects from cns.core.message
    temporal_start: datetime
    temporal_end: datetime
    chunk_index: int
    memory_context_snapshot: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator('messages')
    @classmethod
    def validate_messages_not_empty(cls, v: List[Any]) -> List[Any]:
        """Ensure chunk has at least one message."""
        if not v:
            raise ValueError("ProcessingChunk must contain at least one message")
        return v

    @classmethod
    def from_conversation_messages(
        cls,
        messages: List[Any],  # Message objects
        chunk_index: int
    ) -> 'ProcessingChunk':
        """
        Create ProcessingChunk from continuum Message objects.

        Holds Message objects directly without conversion to preserve
        all attributes and methods during batch payload construction.

        Args:
            messages: List of Message objects from continuum
            chunk_index: Index of this chunk in sequence

        Returns:
            ProcessingChunk instance

        Raises:
            ValueError: If messages list is empty
        """
        if not messages:
            raise ValueError("Cannot create chunk from empty message list")

        return cls(
            messages=messages,
            temporal_start=messages[0].created_at,
            temporal_end=messages[-1].created_at,
            chunk_index=chunk_index,
            memory_context_snapshot={}
        )


class ExtractionBatch(BaseModel):
    """
    Batch extraction tracking.

    Represents a row in extraction_batches table.
    """
    id: Optional[UUID] = None  # Generated by database
    batch_id: str  # Anthropic batch ID
    custom_id: str
    user_id: UUID
    chunk_index: int
    request_payload: Dict[str, Any]
    chunk_metadata: Optional[Dict[str, Any]] = None
    memory_context: Optional[Dict[str, Any]] = None
    status: str  # submitted, processing, completed, failed, expired, cancelled
    created_at: datetime
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    result_url: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None
    extracted_memories: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None

    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of allowed values."""
        allowed = {'submitted', 'processing', 'completed', 'failed', 'expired', 'cancelled'}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v



class PostProcessingBatch(BaseModel):
    """
    Post-processing batch tracking for relationship classification.

    Represents a row in post_processing_batches table.
    """
    id: Optional[UUID] = None  # Generated by database
    batch_id: str  # Anthropic batch ID
    batch_type: str  # relationship_classification or consolidation
    user_id: UUID
    request_payload: Dict[str, Any]
    input_data: Dict[str, Any]
    status: str
    created_at: datetime
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    result_payload: Optional[Dict[str, Any]] = None
    items_submitted: int
    items_completed: int = 0
    items_failed: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    links_created: int = 0
    conflicts_flagged: int = 0
    memories_consolidated: int = 0

    @field_validator('batch_type')
    @classmethod
    def validate_batch_type(cls, v: str) -> str:
        """Validate batch type is one of allowed values."""
        allowed = {'relationship_classification', 'consolidation', 'consolidation_review'}
        if v not in allowed:
            raise ValueError(f"batch_type must be one of {allowed}, got '{v}'")
        return v



class RefinementCandidate(BaseModel):
    """
    Memory identified for refinement.

    Used during refinement analysis to track memories needing consolidation or trimming.
    """
    memory_id: UUID
    reason: str  # 'verbose', 'consolidatable', 'stale'
    current_text: str
    char_count: int
    target_memory_ids: List[UUID] = Field(default_factory=list)  # For consolidation
    similarity_scores: List[float] = Field(default_factory=list)  # Similarity to targets

    @field_validator('reason')
    @classmethod
    def validate_reason(cls, v: str) -> str:
        """Validate reason is one of allowed values."""
        allowed = {'verbose', 'consolidatable', 'stale'}
        if v not in allowed:
            raise ValueError(f"reason must be one of {allowed}, got '{v}'")
        return v


class ConsolidationCluster(BaseModel):
    """
    Cluster of similar memories for consolidation.

    Used during refinement to group memories that should be merged.
    """
    cluster_id: str
    memory_ids: List[UUID]
    memory_texts: List[str]
    similarity_scores: List[float]
    avg_similarity: float
    consolidation_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator('memory_ids')
    @classmethod
    def validate_min_cluster_size(cls, v: List[UUID]) -> List[UUID]:
        """Ensure cluster has at least 2 memories."""
        if len(v) < 2:
            raise ValueError("ConsolidationCluster must contain at least 2 memories")
        return v
