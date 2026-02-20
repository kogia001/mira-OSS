"""
Chat API endpoint - simple request/response over HTTP.

Provides a non-streaming JSON API to send a user message and receive
the assistant's response plus structured metadata. Authenticated via
Bearer token (header) or session cookie.
"""
import base64
import logging
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from auth.api import get_current_user
from clients.files_manager import FilesManager
from utils.distributed_lock import UserRequestLock
from utils.document_processing import process_document, ProcessedDocument, SUPPORTED_DOCUMENT_FORMATS, MAX_DOCUMENT_SIZE_MB
from utils.image_compression import compress_image, CompressedImage
from utils.text_sanitizer import sanitize_message_content
from utils.timezone_utils import utc_now, format_utc_iso
from .base import BaseHandler, APIResponse, ValidationError, create_success_response
from cns.services.orchestrator import get_orchestrator
from cns.infrastructure.continuum_pool import get_continuum_pool


logger = logging.getLogger(__name__)

router = APIRouter()


# Image validation constants (keep consistent with websocket implementation)
SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_MB = 5

# Distributed per-user request lock (coordinates across workers)
_user_request_lock = UserRequestLock(ttl=60)


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., description="User message text")
    image: Optional[str] = Field(None, description="Optional image as base64 string (no data: prefix)")
    image_type: Optional[str] = Field(None, description="MIME type for image (e.g., image/jpeg)")
    document: Optional[str] = Field(None, description="Optional document as base64 string")
    document_type: Optional[str] = Field(None, description="MIME type for document (e.g., application/pdf)")


class ChatEndpoint(BaseHandler):
    """Handler for HTTP chat requests (non-streaming)."""

    def process_request(
        self,
        *,
        user_id: str,
        message: str,
        image: Optional[str],
        image_type: Optional[str],
        document: Optional[str],
        document_type: Optional[str],
    ) -> APIResponse:
        start_time = utc_now()

        # Set user context for RLS and utility functions
        from utils.user_context import set_current_user_id
        set_current_user_id(user_id)

        # Basic validation
        msg = (message or "").strip()
        if not msg:
            raise ValidationError("Message cannot be empty")

        # Sanitize text
        msg = sanitize_message_content(msg)

        # Validate and compress image if provided
        compressed: Optional[CompressedImage] = None
        if image:
            if not image_type:
                raise ValidationError("image_type is required when image is provided")
            if image_type not in SUPPORTED_IMAGE_FORMATS:
                raise ValidationError(
                    f"Unsupported image format. Supported: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
                )
            try:
                decoded = base64.b64decode(image, validate=True)
                if len(decoded) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                    raise ValidationError(f"Image exceeds maximum size of {MAX_IMAGE_SIZE_MB}MB")

                # Compress to both tiers: inference (1200px) and storage (512px WebP)
                compressed = compress_image(decoded, image_type)

            except ValidationError:
                raise
            except ValueError as e:
                # compress_image raises ValueError on failure
                raise ValidationError(f"Image compression failed: {e}")
            except Exception as e:
                raise ValidationError(f"Invalid base64 image: {str(e)}")

        # Validate document if provided (decode and process after getting orchestrator)
        document_bytes: Optional[bytes] = None
        if document:
            if not document_type:
                raise ValidationError("document_type is required when document is provided")
            if document_type not in SUPPORTED_DOCUMENT_FORMATS:
                raise ValidationError(
                    f"Unsupported document format. Supported: PDF, DOCX, XLSX, TXT, CSV, JSON"
                )
            try:
                document_bytes = base64.b64decode(document, validate=True)
                if len(document_bytes) > MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                    raise ValidationError(f"Document exceeds maximum size of {MAX_DOCUMENT_SIZE_MB}MB")
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Invalid base64 document: {str(e)}")

        # Concurrency control: one active request per user
        lock_token = _user_request_lock.acquire(user_id)
        if not lock_token:
            # Use a validation error to preserve consistent error envelope
            raise ValidationError("Another chat request is already in progress for this user")

        files_manager: Optional[FilesManager] = None
        try:
            # Resolve dependencies
            orchestrator = get_orchestrator()
            continuum_pool = get_continuum_pool()

            # Get the user's continuum
            continuum = continuum_pool.get_or_create()

            # Increment segment turn counter at API boundary (before any internal processing)
            # This ensures only real user messages increment the counter, not synthetic messages
            segment_turn_number = continuum_pool.repository.increment_segment_turn(
                continuum.id, user_id
            )

            # Get segment ID for file lifecycle tracking
            active_sentinel = continuum_pool.repository.find_active_segment(continuum.id, user_id)
            if not active_sentinel:
                raise ValidationError("No active segment found")
            segment_id = active_sentinel.metadata.get('segment_id')
            if not segment_id:
                raise ValidationError("Active segment missing segment_id")

            # Process document with Files API support
            processed_doc: Optional[ProcessedDocument] = None
            if document_bytes:
                # Initialize FilesManager with Anthropic client
                files_manager = FilesManager(orchestrator.llm_provider.anthropic_client)

                try:
                    processed_doc = process_document(
                        document_bytes,
                        document_type,
                        files_manager=files_manager,
                        filename=f"document.{document_type.split('/')[-1]}",  # Extract extension from MIME
                        segment_id=segment_id
                    )
                except ValueError as e:
                    raise ValidationError(f"Document processing failed: {e}")
                except Exception as e:
                    raise ValidationError(f"Document upload failed: {e}")

            # Build content arrays (inference tier for LLM, storage tier for persistence)
            inference_content: Union[str, List[Dict[str, Any]]]
            storage_content: Optional[Union[str, List[Dict[str, Any]]]] = None

            if compressed:
                # Image: Inference tier (1200px) for current LLM call
                inference_content = [
                    {"type": "text", "text": msg},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": compressed.inference_media_type,
                            "data": compressed.inference_base64,
                        }
                    }
                ]
                # Storage tier (512px WebP) for persistence and multi-turn context
                storage_content = [
                    {"type": "text", "text": msg},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": compressed.storage_media_type,
                            "data": compressed.storage_base64,
                        }
                    }
                ]
            elif processed_doc:
                # Document handling based on content_type
                if processed_doc.content_type == "container_upload":
                    # Structured data: Files API with file_id (CSV, XLSX, JSON for code execution)
                    doc_block: Dict[str, Any] = {
                        "type": "container_upload",
                        "file_id": processed_doc.data  # file_id from Files API (no source wrapper!)
                    }
                elif processed_doc.content_type == "document":
                    # PDF: Base64 document block
                    doc_block = {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": processed_doc.media_type,
                            "data": processed_doc.data,
                        }
                    }
                else:
                    # DOCX/plain text: Extracted text
                    doc_block = {
                        "type": "text",
                        "text": f"[Document: {processed_doc.media_type}]\n{processed_doc.data}",
                    }

                inference_content = [{"type": "text", "text": msg}, doc_block]
                # Storage: Use same block as inference (file_id persists until segment collapse)
                storage_content = [
                    {"type": "text", "text": msg},
                    doc_block  # Reuse same block (file_id or base64)
                ]
            else:
                inference_content = msg

            # Create a Unit of Work and process via orchestrator
            uow = continuum_pool.begin_work(continuum)

            from config.config_manager import config as app_config
            continuum, response_text, metadata = orchestrator.process_message(
                continuum,
                inference_content,
                app_config.system_prompt,
                stream=True,           # orchestrator currently streams internally
                stream_callback=None,   # no external streaming for HTTP endpoint
                unit_of_work=uow,
                storage_content=storage_content,  # 512px WebP for persistence
                segment_turn_number=segment_turn_number,  # Turn count within segment
            )

            # Commit batched changes
            uow.commit()

            processing_time_ms = int((utc_now() - start_time).total_seconds() * 1000)

            # Build response
            data: Dict[str, Any] = {
                "continuum_id": str(continuum.id),
                "response": response_text,
                "metadata": {
                    "tools_used": metadata.get("tools_used", []),
                    "referenced_memories": metadata.get("referenced_memories", []),
                    "surfaced_memories": metadata.get("surfaced_memories", []),
                    "code_execution_file_ids": metadata.get("code_execution_file_ids", []),
                    "verified_output_paths": metadata.get("verified_output_paths", []),
                    "output_claim_detected": metadata.get("output_claim_detected", False),
                    "output_claim_blocked": metadata.get("output_claim_blocked", False),
                    "processing_time_ms": processing_time_ms,
                },
            }

            return create_success_response(
                data=data,
                meta={
                    "timestamp": format_utc_iso(utc_now()),
                },
            )

        finally:
            # Note: File cleanup happens on segment collapse, not per-request
            _user_request_lock.release(user_id, lock_token)


@router.post("/chat")
async def chat_endpoint(request: ChatRequest, current_user=Depends(get_current_user)):
    """Send a message and receive assistant response as JSON."""
    try:
        handler = ChatEndpoint()
        response = handler.handle_request(
            user_id=current_user["user_id"],
            message=request.message,
            image=request.image,
            image_type=request.image_type,
            document=request.document,
            document_type=request.document_type,
        )
        return response.to_dict()

    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": e.message
                }
            }
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Chat processing failed"
                }
            }
        )
