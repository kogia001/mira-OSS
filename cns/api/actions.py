"""
Actions API endpoint - domain-routed state mutations.

Executes state-changing operations through domain-specific handlers that
call tools and services directly, just as MIRA does during continuums.
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator

from utils.user_context import get_current_user_id, set_current_user_id
from auth.api import get_current_user
from .base import BaseHandler, ValidationError, NotFoundError
from utils.timezone_utils import utc_now, format_utc_iso
from clients.valkey_client import get_valkey_client
from working_memory.trinkets.base import TRINKET_KEY_PREFIX

logger = logging.getLogger(__name__)

router = APIRouter()



class DomainType(str, Enum):
    """Supported action domains."""
    REMINDER = "reminder"
    MEMORY = "memory"
    USER = "user"
    CONTACTS = "contacts"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CONTINUUM = "continuum"


class ActionRequest(BaseModel):
    """Action request schema."""
    domain: DomainType = Field(..., description="Domain for the action")
    action: str = Field(..., description="Action to perform")
    data: Dict[str, Any] = Field(default_factory=dict, description="Action-specific data")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        if not v.strip():
            raise ValueError("Action cannot be empty")
        return v.strip()


class BaseDomainHandler(BaseHandler):
    """Base handler for domain-specific actions."""
    
    # Define available actions and their required/optional fields
    ACTIONS: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self):
        super().__init__()  # Initialize BaseHandler (logger, thread pool)
        from utils.user_context import get_current_user_id
        self.user_id = get_current_user_id()
    
    def validate_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action and its data against schema."""
        if action not in self.ACTIONS:
            available_actions = list(self.ACTIONS.keys())
            raise ValidationError(
                f"Unknown action '{action}' for {self.__class__.__name__}. "
                f"Available actions: {', '.join(available_actions)}"
            )
        
        # Get schema for this action
        schema = self.ACTIONS[action]
        required_fields = schema.get('required', [])
        optional_fields = schema.get('optional', [])
        all_fields = required_fields + optional_fields
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(
                f"Missing required fields for action '{action}': {', '.join(missing_fields)}"
            )
        
        # Check for unknown fields
        unknown_fields = [field for field in data.keys() if field not in all_fields]
        if unknown_fields:
            raise ValidationError(
                f"Unknown fields for action '{action}': {', '.join(unknown_fields)}. "
                f"Valid fields: {', '.join(all_fields)}"
            )
        
        # Validate field types
        for field, value in data.items():
            if field in schema.get('types', {}):
                expected_type = schema['types'][field]
                
                # Special handling for UUID type
                if expected_type == 'uuid':
                    from uuid import UUID
                    try:
                        UUID(value)
                    except (ValueError, TypeError):
                        raise ValidationError(f"Field '{field}' must be a valid UUID")
                elif not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Field '{field}' has invalid type, got {type(value).__name__}"
                    )
        
        return data
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action. Override in subclasses."""
        raise NotImplementedError(f"Action '{action}' not implemented")


class ReminderDomainHandler(BaseDomainHandler):
    """Handler for reminder domain actions."""
    
    ACTIONS = {
        "complete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": str}
        },
        "bulk_complete": {
            "required": ["ids"],
            "optional": [],
            "types": {"ids": list}
        },
        "create": {
            "required": ["title", "date"],
            "optional": ["description", "contact_name", "additional_notes"],
            "types": {
                "title": str,
                "date": str,
                "description": str,
                "contact_name": str,
                "additional_notes": str
            }
        },
        "update": {
            "required": ["id"],
            "optional": ["title", "date", "description", "contact_name", "additional_notes"],
            "types": {
                "id": str,
                "title": str,
                "date": str,
                "description": str,
                "contact_name": str,
                "additional_notes": str
            }
        },
        "delete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": str}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reminder actions using ReminderTool."""
        from tools.implementations.reminder_tool import ReminderTool
        reminder_tool = ReminderTool()
        
        try:
            if action == "complete":
                # Mark single reminder as completed
                result = reminder_tool.run(
                    operation="mark_completed",
                    reminder_id=data["id"]
                )

                return {
                    "completed": True,
                    "reminder": result.get("reminder"),
                    "message": result.get("message", "Reminder marked as completed")
                }
            
            elif action == "bulk_complete":
                # Mark multiple reminders as completed
                reminder_ids = data["ids"]
                
                # Validate that ids is a non-empty list of strings
                if not reminder_ids:
                    raise ValidationError("At least one reminder ID is required")
                
                if not all(isinstance(id, str) for id in reminder_ids):
                    raise ValidationError("All reminder IDs must be strings")
                
                completed = []
                failed = []
                
                for reminder_id in reminder_ids:
                    try:
                        result = reminder_tool.run(
                            operation="mark_completed",
                            reminder_id=reminder_id
                        )
                        completed.append({
                            "id": reminder_id,
                            "title": result.get("reminder", {}).get("title", "Unknown")
                        })
                    except Exception as e:
                        failed.append({
                            "id": reminder_id,
                            "error": str(e)
                        })
                
                return {
                    "completed_count": len(completed),
                    "failed_count": len(failed),
                    "completed": completed,
                    "failed": failed,
                    "message": f"Completed {len(completed)} of {len(reminder_ids)} reminders"
                }

            elif action == "create":
                # Create new reminder with all provided fields
                result = reminder_tool.run(
                    operation="add_reminder",
                    **data  # Pass all validated data
                )
                return {
                    "created": True,
                    "reminder": result.get("reminder"),
                    "contact_found": result.get("contact_found", False),
                    "contact_info": result.get("contact_info"),
                    "message": result.get("message", "Reminder created")
                }
            
            elif action == "update":
                # Update existing reminder
                reminder_id = data.pop("id")  # Remove id from update fields
                
                # Only pass fields that were actually provided
                update_fields = {k: v for k, v in data.items() if v is not None}
                
                if not update_fields:
                    raise ValidationError("At least one field to update must be provided")
                
                result = reminder_tool.run(
                    operation="update_reminder",
                    reminder_id=reminder_id,
                    **update_fields
                )
                return {
                    "updated": True,
                    "reminder": result.get("reminder"),
                    "updated_fields": result.get("updated_fields", []),
                    "message": result.get("message", "Reminder updated")
                }
            
            elif action == "delete":
                # Delete reminder
                result = reminder_tool.run(
                    operation="delete_reminder",
                    reminder_id=data["id"]
                )
                return {
                    "deleted": True,
                    "id": data["id"],
                    "message": result.get("message", "Reminder deleted")
                }
            
            else:
                raise ValidationError(f"Unknown action: {action}")

        except ValueError as e:
            # Tool raises ValueError for business logic errors
            raise ValidationError(str(e))


class MemoryDomainHandler(BaseDomainHandler):
    """Handler for memory domain actions."""
    
    ACTIONS = {
        "create": {
            "required": ["content"],
            "optional": ["importance"],
            "types": {
                "content": str,
                "importance": (int, float)
            }
        },
        "delete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": "uuid"}
        },
        "bulk_delete": {
            "required": ["ids"],
            "optional": [],
            "types": {"ids": list}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory actions using LTMemoryDB."""
        from lt_memory.db_access import LTMemoryDB
        from utils.database_session_manager import get_shared_session_manager

        # Set user context for LTMemoryDB operations
        set_current_user_id(self.user_id)

        # Use shared session manager to prevent connection pool exhaustion
        session_manager = get_shared_session_manager()
        lt_db = LTMemoryDB(session_manager)
        
        if action == "create":
            # Manual memory creation
            content = data["content"]
            importance = data.get("importance", 0.5)
            
            # Validate importance score
            if not 0 <= importance <= 1:
                raise ValidationError("Importance score must be between 0 and 1")
            
            # Generate embedding for the memory
            from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
            from lt_memory.models import ExtractedMemory
            embeddings_provider = get_hybrid_embeddings_provider()  # Use singleton
            # Use deep embeddings for memory storage (768d document encoding)
            embedding = embeddings_provider.encode_deep(content)

            # Convert ndarray to list for database storage (serialization boundary)
            embedding_list = embedding.tolist()

            # Create ExtractedMemory object (embedding passed separately to store_memories)
            memory = ExtractedMemory(
                text=content,
                importance_score=importance
            )

            memory_ids = lt_db.store_memories([memory], embeddings=[embedding_list])
            
            if not memory_ids:
                raise ValidationError("Failed to create memory")
            
            # Get the created memory to return
            created_memory = lt_db.get_memory(memory_ids[0])

            return jsonable_encoder({
                "created": True,
                "memory": {
                    "id": created_memory.id,
                    "text": created_memory.text,
                    "importance_score": created_memory.importance_score,
                    "created_at": created_memory.created_at
                },
                "message": "Memory created successfully"
            })
        
        elif action == "delete":
            # Delete single memory
            memory_id = data["id"]
            
            # Verify memory exists
            memory = lt_db.get_memory(memory_id)
            if not memory:
                raise NotFoundError("memory", memory_id)

            # Delete the memory
            deleted_count = lt_db.delete_memories([memory_id])
            
            return {
                "deleted": deleted_count > 0,
                "id": memory_id,
                "message": f"Memory deleted successfully" if deleted_count > 0 else "Failed to delete memory"
            }
        
        elif action == "bulk_delete":
            # Delete multiple memories
            memory_ids = data["ids"]
            
            # Validate that ids is a non-empty list of strings
            if not memory_ids:
                raise ValidationError("At least one memory ID is required")
            
            if not all(isinstance(id, str) for id in memory_ids):
                raise ValidationError("All memory IDs must be strings")
            
            # Delete memories
            deleted_count = lt_db.delete_memories(memory_ids)
            
            return {
                "deleted_count": deleted_count,
                "requested_count": len(memory_ids),
                "ids": memory_ids,
                "message": f"Deleted {deleted_count} of {len(memory_ids)} memories"
            }
        
        else:
            raise ValidationError(f"Unknown action: {action}")


class ContactsDomainHandler(BaseDomainHandler):
    """Handler for contacts domain actions."""
    
    ACTIONS = {
        "create": {
            "required": ["name"],
            "optional": ["email", "phone"],
            "types": {
                "name": str,
                "email": str,
                "phone": str
            }
        },
        "get": {
            "required": ["identifier"],
            "optional": [],
            "types": {"identifier": str}
        },
        "list": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "update": {
            "required": ["identifier"],
            "optional": ["name", "email", "phone"],
            "types": {
                "identifier": str,
                "name": str,
                "email": str,
                "phone": str
            }
        },
        "delete": {
            "required": ["identifier"],
            "optional": [],
            "types": {"identifier": str}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute contacts actions using ContactsTool."""
        from tools.implementations.contacts_tool import ContactsTool
        contacts_tool = ContactsTool()
        
        try:
            if action == "create":
                # Create new contact
                result = contacts_tool.run(
                    operation="add_contact",
                    **data  # Pass all validated data
                )
                return {
                    "created": True,
                    "contact": result.get("contact"),
                    "message": result.get("message", "Contact created")
                }
            
            elif action == "get":
                # Get contact by UUID or name
                result = contacts_tool.run(
                    operation="get_contact",
                    identifier=data["identifier"]
                )
                return {
                    "found": True,
                    "contact": result.get("contact"),
                    "message": result.get("message", "Contact found")
                }
            
            elif action == "list":
                # List all contacts
                result = contacts_tool.run(
                    operation="list_contacts"
                )
                return {
                    "contacts": result.get("contacts", []),
                    "count": len(result.get("contacts", [])),
                    "message": result.get("message", "Contacts retrieved")
                }
            
            elif action == "update":
                # Update existing contact
                identifier = data.pop("identifier")  # Remove identifier from update fields
                
                # Only pass fields that were actually provided
                update_fields = {k: v for k, v in data.items() if v is not None}
                
                if not update_fields:
                    raise ValidationError("At least one field to update must be provided")
                
                result = contacts_tool.run(
                    operation="update_contact",
                    identifier=identifier,
                    **update_fields
                )
                return {
                    "updated": True,
                    "contact": result.get("contact"),
                    "message": result.get("message", "Contact updated")
                }
            
            elif action == "delete":
                # Delete contact
                result = contacts_tool.run(
                    operation="delete_contact",
                    identifier=data["identifier"]
                )
                return {
                    "deleted": True,
                    "deleted_contact": result.get("deleted_contact"),
                    "message": result.get("message", "Contact deleted")
                }
            
            else:
                raise ValidationError(f"Unknown action: {action}")
                
        except ValueError as e:
            # Tool raises ValueError for business logic errors
            raise ValidationError(str(e))


class UserDomainHandler(BaseDomainHandler):
    """Handler for user preference/settings actions."""
    
    ACTIONS = {
        "update_preferences": {
            "required": [],
            "optional": ["theme", "timezone", "language", "calendar_url"],
            "types": {
                "theme": str,
                "timezone": str,
                "language": str,
                "calendar_url": str
            }
        },
        "store_calendar_config": {
            "required": ["calendar_url"],
            "optional": [],
            "types": {
                "calendar_url": str
            }
        },
        "get_calendar_config": {
            "required": [],
            "optional": [],
            "types": {}
        }
    }
    
    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute user preference actions."""
        if action == "update_preferences":
            # Validate theme if provided
            if "theme" in data:
                valid_themes = ["light", "dark", "auto"]
                if data["theme"] not in valid_themes:
                    raise ValidationError(
                        f"Invalid theme '{data['theme']}'. "
                        f"Valid themes: {', '.join(valid_themes)}"
                    )
            
            # Validate timezone if provided
            if "timezone" in data:
                from utils.timezone_utils import validate_timezone
                try:
                    validate_timezone(data["timezone"])
                except Exception:
                    raise ValidationError(f"Invalid timezone: {data['timezone']}")
            
            # Validate language if provided
            if "language" in data:
                valid_languages = ["en", "es", "fr", "de", "ja", "zh"]
                if data["language"] not in valid_languages:
                    raise ValidationError(
                        f"Invalid language '{data['language']}'. "
                        f"Valid languages: {', '.join(valid_languages)}"
                    )
            
            # Validate calendar URL if provided
            if "calendar_url" in data:
                calendar_url = data["calendar_url"]
                if not calendar_url or not isinstance(calendar_url, str):
                    raise ValidationError("Calendar URL must be a non-empty string")
                # Basic URL validation
                if not (calendar_url.startswith("http://") or calendar_url.startswith("https://")):
                    raise ValidationError("Calendar URL must start with http:// or https://")
            
            # For now, return dummy success response
            # In the future, this would update a user_preferences table
            updated_prefs = {
                "theme": data.get("theme", "light"),
                "timezone": data.get("timezone", "UTC"),
                "language": data.get("language", "en"),
                "calendar_url": data.get("calendar_url")
            }
            
            return {
                "updated": True,
                "preferences": updated_prefs,
                "message": "Preferences updated successfully (placeholder implementation)"
            }

        elif action == "store_calendar_config":
            from utils.user_credentials import UserCredentialService

            calendar_url = data["calendar_url"]

            # Store calendar URL in user credentials
            credential_service = UserCredentialService()
            credential_service.store_credential(
                user_id=self.user_id,
                credential_type="calendar_url",
                service_name="calendar",
                credential_value=calendar_url
            )

            return {
                "success": True,
                "message": "Calendar URL stored successfully"
            }
        
        elif action == "get_calendar_config":
            from utils.user_credentials import UserCredentialService

            credential_service = UserCredentialService()
            calendar_url = credential_service.get_credential(
                user_id=self.user_id,
                credential_type="calendar_url",
                service_name="calendar"
            )

            return {
                "success": True,
                "calendar_url": calendar_url if calendar_url else None,
                "message": "Calendar configuration retrieved" if calendar_url else "No calendar URL configured"
            }
        
        else:
            raise ValidationError(f"Unknown action: {action}")


class DomainKnowledgeDomainHandler(BaseDomainHandler):
    """Handler for domaindoc actions with SQLite-based section storage."""

    ACTIONS = {
        "create": {
            "required": ["label", "description"],
            "optional": [],
            "types": {"label": str, "description": str}
        },
        "enable": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "disable": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "delete": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "list": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "get": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "modify_metadata": {
            "required": ["label"],
            "optional": ["new_label", "description"],
            "types": {"label": str, "new_label": str, "description": str}
        },
        "list_sections": {
            "required": ["label"],
            "optional": ["parent"],
            "types": {"label": str, "parent": str}
        },
        "get_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "update_section": {
            "required": ["label", "section", "content"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "content": str, "parent": str}
        },
        "create_section": {
            "required": ["label", "section", "content"],
            "optional": ["after", "parent"],
            "types": {"label": str, "section": str, "content": str, "after": str, "parent": str}
        },
        "rename_section": {
            "required": ["label", "section", "new_name"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "new_name": str, "parent": str}
        },
        "delete_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "reorder_sections": {
            "required": ["label", "order"],
            "optional": ["parent"],
            "types": {"label": str, "order": list, "parent": str}
        },
        "expand_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "collapse_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "get_section_history": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "rollback_section": {
            "required": ["label", "section", "version_num"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "version_num": int, "parent": str}
        }
    }

    def _get_db(self):
        """Get UserDataManager for current user."""
        from utils.userdata_manager import get_user_data_manager
        return get_user_data_manager(self.user_id)

    def _validate_label(self, label: str) -> None:
        """Validate domaindoc label."""
        if not label:
            raise ValidationError("Label cannot be empty")
        if not label.replace("_", "").isalnum():
            raise ValidationError(f"Invalid label '{label}'. Use only letters, numbers, and underscores.")

    def _get_domaindoc(self, db, label: str) -> Dict[str, Any]:
        """Get domaindoc by label, raising ValidationError if not found."""
        results = db.select("domaindocs", "label = :label", {"label": label})
        if not results:
            raise ValidationError(f"Domaindoc '{label}' not found")
        return results[0]

    def _get_section(self, db, domaindoc_id: int, header: str, parent_header: str = None) -> Dict[str, Any]:
        """Get section by header, optionally under a parent."""
        if parent_header:
            parent = self._get_section(db, domaindoc_id, parent_header)
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header AND parent_section_id = :parent_id",
                {"doc_id": domaindoc_id, "header": header, "parent_id": parent["id"]}
            )
            if not results:
                raise ValidationError(f"Subsection '{header}' not found under '{parent_header}'")
        else:
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header AND parent_section_id IS NULL",
                {"doc_id": domaindoc_id, "header": header}
            )
            if not results:
                raise ValidationError(f"Section '{header}' not found")
        return db._decrypt_dict(results[0])

    def _get_all_sections(self, db, domaindoc_id: int, parent_id: int = None) -> list:
        """Get sections ordered by sort_order, optionally filtered by parent."""
        if parent_id is not None:
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND parent_section_id = :parent_id ORDER BY sort_order",
                {"doc_id": domaindoc_id, "parent_id": parent_id}
            )
        else:
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND parent_section_id IS NULL ORDER BY sort_order",
                {"doc_id": domaindoc_id}
            )
        return [db._decrypt_dict(row) for row in results]

    def _get_subsections(self, db, parent_id: int) -> list:
        """Get all subsections of a parent section."""
        results = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE parent_section_id = :parent_id ORDER BY sort_order",
            {"parent_id": parent_id}
        )
        return [db._decrypt_dict(row) for row in results]

    def _check_duplicate_section(self, db, domaindoc_id: int, header: str, parent_section_id: int | None, exclude_section_id: int | None = None) -> None:
        """Raise ValidationError if a section with this header already exists at the same level."""
        query = """
            SELECT id FROM domaindoc_sections
            WHERE domaindoc_id = :doc_id
            AND header = :header
            AND parent_section_id IS NOT DISTINCT FROM :parent_id
        """
        params = {"doc_id": domaindoc_id, "header": header, "parent_id": parent_section_id}

        if exclude_section_id:
            query += " AND id != :exclude_id"
            params["exclude_id"] = exclude_section_id

        existing = db.fetchone(query, params)
        if existing:
            level = "subsection" if parent_section_id else "section"
            raise ValidationError(f"A {level} named '{header}' already exists at this level")

    def _record_version(self, db, domaindoc_id: int, operation: str, diff_data: dict, section_id: int = None) -> int:
        """Record a version entry for an operation."""
        import json
        result = db.fetchone(
            "SELECT MAX(version_num) as max_ver FROM domaindoc_versions WHERE domaindoc_id = :doc_id",
            {"doc_id": domaindoc_id}
        )
        version_num = (result.get("max_ver") or 0) + 1
        now = format_utc_iso(utc_now())

        db.insert("domaindoc_versions", {
            "domaindoc_id": domaindoc_id,
            "section_id": section_id,
            "version_num": version_num,
            "operation": operation,
            "encrypted__diff_data": json.dumps(diff_data),
            "created_at": now
        })
        return version_num

    def _invalidate_trinket_cache(self) -> None:
        """Invalidate domaindoc trinket cache after state changes."""
        user_id = get_current_user_id()
        hash_key = f"{TRINKET_KEY_PREFIX}:{user_id}"
        valkey = get_valkey_client()
        valkey.hdel_with_retry(hash_key, "domaindoc")

    # Read-only actions that don't require cache invalidation
    _READ_ONLY_ACTIONS = {"list", "get", "list_sections", "get_section", "get_section_history"}

    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domaindoc actions using SQLite storage."""
        import json

        try:
            db = self._get_db()

            if action == "create":
                result = self._action_create(db, data)
            elif action == "enable":
                result = self._action_enable(db, data)
            elif action == "disable":
                result = self._action_disable(db, data)
            elif action == "delete":
                result = self._action_delete(db, data)
            elif action == "list":
                result = self._action_list(db)
            elif action == "get":
                result = self._action_get(db, data)
            elif action == "modify_metadata":
                result = self._action_modify_metadata(db, data)
            elif action == "list_sections":
                result = self._action_list_sections(db, data)
            elif action == "get_section":
                result = self._action_get_section(db, data)
            elif action == "update_section":
                result = self._action_update_section(db, data)
            elif action == "create_section":
                result = self._action_create_section(db, data)
            elif action == "rename_section":
                result = self._action_rename_section(db, data)
            elif action == "delete_section":
                result = self._action_delete_section(db, data)
            elif action == "reorder_sections":
                result = self._action_reorder_sections(db, data)
            elif action == "expand_section":
                result = self._action_expand_section(db, data)
            elif action == "collapse_section":
                result = self._action_collapse_section(db, data)
            elif action == "get_section_history":
                result = self._action_get_section_history(db, data)
            elif action == "rollback_section":
                result = self._action_rollback_section(db, data)
            else:
                raise ValidationError(f"Unknown action: {action}")

            # Invalidate trinket cache after write operations
            if action not in self._READ_ONLY_ACTIONS:
                self._invalidate_trinket_cache()

            return result

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Domaindoc action '{action}' failed: {e}", exc_info=True)
            raise ValidationError(f"Action failed: {str(e)}")

    def _action_create(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new domaindoc."""
        label = data["label"]
        description = data["description"]

        self._validate_label(label)
        if not description or len(description) > 1000:
            raise ValidationError("Description must be 1-1000 characters")

        existing = db.select("domaindocs", "label = :label", {"label": label})
        if existing:
            raise ValidationError(f"Domaindoc '{label}' already exists")

        expanded_description = self._expand_description(label, description)
        now = format_utc_iso(utc_now())

        domaindoc_id = db.insert("domaindocs", {
            "label": label,
            "encrypted__description": expanded_description,
            "enabled": False,
            "created_at": now,
            "updated_at": now
        })

        db.insert("domaindoc_sections", {
            "domaindoc_id": int(domaindoc_id),
            "header": "OVERVIEW",
            "encrypted__content": "",
            "sort_order": 0,
            "collapsed": False,
            "created_at": now,
            "updated_at": now
        })

        return {
            "created": True,
            "label": label,
            "description": expanded_description,
            "message": f"Domaindoc '{label}' created. Use enable action to activate it."
        }

    def _action_enable(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enable a domaindoc. Collapses all sections except first."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindocs SET enabled = TRUE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )
        db.execute(
            "UPDATE domaindoc_sections SET collapsed = TRUE, updated_at = :now WHERE domaindoc_id = :doc_id AND sort_order > 0",
            {"now": now, "doc_id": doc["id"]}
        )

        return {"enabled": True, "label": label, "message": f"Domaindoc '{label}' enabled"}

    def _action_disable(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Disable a domaindoc."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindocs SET enabled = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )

        return {"disabled": True, "label": label, "message": f"Domaindoc '{label}' disabled"}

    def _action_delete(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a domaindoc and all its sections."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        db.execute("DELETE FROM domaindocs WHERE id = :id", {"id": doc["id"]})

        return {"deleted": True, "label": label, "message": f"Domaindoc '{label}' deleted"}

    def _action_list(self, db) -> Dict[str, Any]:
        """List all domaindocs."""
        results = db.fetchall("SELECT * FROM domaindocs ORDER BY label")
        domaindocs = [db._decrypt_dict(row) for row in results]

        return {
            "domaindocs": [
                {
                    "label": d["label"],
                    "description": d.get("encrypted__description", ""),
                    "enabled": d.get("enabled", False),
                    "created_at": d.get("created_at"),
                    "updated_at": d.get("updated_at")
                }
                for d in domaindocs
            ],
            "count": len(domaindocs),
            "message": f"Found {len(domaindocs)} domaindocs"
        }

    def _action_get(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a domaindoc with all its sections (including subsections)."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        # Get ALL sections for this domaindoc (both top-level and subsections)
        all_results = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id ORDER BY parent_section_id NULLS FIRST, sort_order",
            {"doc_id": doc["id"]}
        )
        all_sections = [db._decrypt_dict(row) for row in all_results]

        return {
            "label": label,
            "description": doc.get("encrypted__description", ""),
            "enabled": doc.get("enabled", False),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "sections": [
                {
                    "id": s["id"],
                    "header": s["header"],
                    "content": s.get("encrypted__content", ""),
                    "collapsed": s.get("collapsed", False),
                    "sort_order": s.get("sort_order", 0),
                    "parent_section_id": s.get("parent_section_id"),
                    "updated_at": s.get("updated_at"),
                    "created_at": s.get("created_at")
                }
                for s in all_sections
            ],
            "message": f"Retrieved domaindoc '{label}'"
        }

    def _action_modify_metadata(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify domaindoc label or description."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        new_label = data.get("new_label")
        new_description = data.get("description")

        if not new_label and not new_description:
            raise ValidationError("At least one of 'new_label' or 'description' must be provided")

        updates = {"updated_at": format_utc_iso(utc_now())}

        if new_label:
            self._validate_label(new_label)
            # Check for collision
            existing = db.select("domaindocs", "label = :label AND id != :id", {"label": new_label, "id": doc["id"]})
            if existing:
                raise ValidationError(f"Domaindoc '{new_label}' already exists")
            updates["label"] = new_label

        if new_description:
            if len(new_description) > 2000:
                raise ValidationError("Description must be under 2000 characters")
            updates["encrypted__description"] = new_description

        db.update("domaindocs", updates, "id = :id", {"id": doc["id"]})

        return {
            "updated": True,
            "label": new_label or label,
            "description": new_description or doc.get("encrypted__description"),
            "message": f"Domaindoc metadata updated"
        }

    def _action_list_sections(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """List sections for a domaindoc, optionally under a parent."""
        label = data["label"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        if parent_header:
            parent = self._get_section(db, doc["id"], parent_header)
            sections = self._get_all_sections(db, doc["id"], parent_id=parent["id"])
        else:
            sections = self._get_all_sections(db, doc["id"])

        return {
            "label": label,
            "parent": parent_header,
            "sections": [
                {
                    "header": s["header"],
                    "collapsed": s.get("collapsed", False),
                    "sort_order": s.get("sort_order", 0),
                    "char_count": len(s.get("encrypted__content", "")),
                    "has_children": len(self._get_subsections(db, s["id"])) > 0,
                    "child_count": len(self._get_subsections(db, s["id"]))
                }
                for s in sections
            ]
        }

    def _action_get_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a single section's content."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)
        subsections = self._get_subsections(db, section["id"])

        return {
            "label": label,
            "section": header,
            "parent": parent_header,
            "content": section.get("encrypted__content", ""),
            "collapsed": section.get("collapsed", False),
            "sort_order": section.get("sort_order", 0),
            "has_children": len(subsections) > 0,
            "child_count": len(subsections)
        }

    def _action_update_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a section's content (full replacement via UI)."""
        label = data["label"]
        header = data["section"]
        content = data["content"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)
        previous_content = section.get("encrypted__content", "")
        now = format_utc_iso(utc_now())

        db.update(
            "domaindoc_sections",
            {"encrypted__content": content, "updated_at": now},
            "id = :id",
            {"id": section["id"]}
        )

        # Record version for UI edits (destructive - needs previous content)
        self._record_version(db, doc["id"], "ui_replace", {
            "section": header,
            "previous_content": previous_content,
            "new_length": len(content),
            "parent": parent_header
        }, section["id"])

        return {"updated": True, "label": label, "section": header, "parent": parent_header, "char_count": len(content)}

    def _action_create_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new section or subsection."""
        label = data["label"]
        header = data["section"]
        content = data["content"]
        after = data.get("after")
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        now = format_utc_iso(utc_now())

        parent_id = None
        if parent_header:
            # Creating a subsection - validate depth
            parent = self._get_section(db, doc["id"], parent_header)
            if parent.get("parent_section_id") is not None:
                raise ValidationError("Maximum nesting depth is 1. Cannot add subsection to a subsection.")
            if parent["sort_order"] == 0:
                raise ValidationError("Cannot add subsections to the overview section")
            parent_id = parent["id"]

        # Get sections at the target level (top-level or within parent)
        sections = self._get_all_sections(db, doc["id"], parent_id)

        # Check for duplicate section name at this level
        self._check_duplicate_section(db, doc["id"], header, parent_id)

        if after:
            after_sec = self._get_section(db, doc["id"], after, parent_header)
            new_order = after_sec["sort_order"] + 1
            for sec in sections:
                if sec["sort_order"] >= new_order:
                    db.execute(
                        "UPDATE domaindoc_sections SET sort_order = sort_order + 1 WHERE id = :id",
                        {"id": sec["id"]}
                    )
        else:
            new_order = max((s["sort_order"] for s in sections), default=-1) + 1

        section_id = db.insert("domaindoc_sections", {
            "domaindoc_id": doc["id"],
            "parent_section_id": parent_id,
            "header": header,
            "encrypted__content": content,
            "sort_order": new_order,
            "collapsed": True,  # New sections start collapsed
            "created_at": now,
            "updated_at": now
        })

        # Record version
        self._record_version(db, doc["id"], "ui_create_section", {
            "header": header,
            "content_length": len(content),
            "after": after,
            "sort_order": new_order,
            "parent": parent_header
        }, int(section_id))

        return {"created": True, "label": label, "section": header, "parent": parent_header, "sort_order": new_order}

    def _action_rename_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rename a section or subsection header."""
        label = data["label"]
        header = data["section"]
        new_name = data["new_name"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Check for duplicate (exclude current section to allow case changes)
        parent_id = section.get("parent_section_id")
        self._check_duplicate_section(db, doc["id"], new_name, parent_id, exclude_section_id=section["id"])

        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindoc_sections SET header = :new_name, updated_at = :now WHERE id = :id",
            {"new_name": new_name, "now": now, "id": section["id"]}
        )

        # Record version
        self._record_version(db, doc["id"], "ui_rename_section", {
            "old_name": header,
            "new_name": new_name,
            "parent": parent_header
        }, section["id"])

        return {"renamed": True, "label": label, "from": header, "to": new_name, "parent": parent_header}

    def _action_delete_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a section or subsection."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Can't delete overview (first top-level section)
        if section["sort_order"] == 0 and section.get("parent_section_id") is None:
            raise ValidationError("Cannot delete the first section (overview)")

        # For top-level sections, check if section AND all subsections are expanded
        deleted_children = []
        if section.get("parent_section_id") is None:
            subsections = self._get_subsections(db, section["id"])
            if subsections:
                # Check section itself is expanded
                if section.get("collapsed"):
                    raise ValidationError(f"Please expand section '{header}' before deleting to review its contents")
                # Check all subsections are expanded
                collapsed_subs = [s["header"] for s in subsections if s.get("collapsed")]
                if collapsed_subs:
                    raise ValidationError(
                        f"Please expand all subsections of '{header}' before deleting to confirm you've reviewed their contents: {collapsed_subs}"
                    )
                deleted_children = [s["header"] for s in subsections]

        # Record version BEFORE delete (destructive - store content for undo)
        self._record_version(db, doc["id"], "ui_delete_section", {
            "header": header,
            "deleted_content": section.get("encrypted__content", ""),
            "sort_order": section["sort_order"],
            "parent": parent_header,
            "deleted_children": deleted_children
        }, section["id"])

        # Delete cascades to children via FK ON DELETE CASCADE
        db.execute("DELETE FROM domaindoc_sections WHERE id = :id", {"id": section["id"]})

        # Renumber remaining sections at the same level
        parent_id = section.get("parent_section_id")
        sections = self._get_all_sections(db, doc["id"], parent_id)
        for i, s in enumerate(sections):
            if s["sort_order"] != i:
                db.execute(
                    "UPDATE domaindoc_sections SET sort_order = :order WHERE id = :id",
                    {"order": i, "id": s["id"]}
                )

        result = {"deleted": True, "label": label, "section": header, "parent": parent_header}
        if deleted_children:
            result["deleted_children"] = deleted_children
        return result

    def _action_reorder_sections(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Reorder sections at a level (top-level or within a parent)."""
        label = data["label"]
        order = data["order"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        parent_id = None
        if parent_header:
            parent = self._get_section(db, doc["id"], parent_header)
            parent_id = parent["id"]

        sections = self._get_all_sections(db, doc["id"], parent_id)

        existing = {s["header"] for s in sections}
        provided = set(order)

        if existing != provided:
            level_desc = f"subsections of '{parent_header}'" if parent_header else "top-level sections"
            raise ValidationError(f"Order must contain exactly all {level_desc}: {list(existing)}")

        now = format_utc_iso(utc_now())
        for new_order, header in enumerate(order):
            sec = next(s for s in sections if s["header"] == header)
            db.execute(
                "UPDATE domaindoc_sections SET sort_order = :order, updated_at = :now WHERE id = :id",
                {"order": new_order, "now": now, "id": sec["id"]}
            )

        return {"reordered": True, "label": label, "order": order, "parent": parent_header}

    def _action_expand_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Expand a section or subsection."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)
        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindoc_sections SET collapsed = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": section["id"]}
        )

        return {"expanded": True, "label": label, "section": header, "parent": parent_header}

    def _action_collapse_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collapse a section or subsection."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Can't collapse first top-level section (overview)
        if section["sort_order"] == 0 and section.get("parent_section_id") is None:
            return {"collapsed": False, "label": label, "section": header, "parent": parent_header, "note": "First section cannot be collapsed"}

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindoc_sections SET collapsed = TRUE, updated_at = :now WHERE id = :id",
            {"now": now, "id": section["id"]}
        )

        return {"collapsed": True, "label": label, "section": header, "parent": parent_header}

    def _action_get_section_history(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get version history for a section or subsection."""
        import json
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        results = db.fetchall(
            "SELECT * FROM domaindoc_versions WHERE domaindoc_id = :doc_id AND section_id = :sec_id ORDER BY version_num DESC",
            {"doc_id": doc["id"], "sec_id": section["id"]}
        )
        versions = [db._decrypt_dict(row) for row in results]

        return {
            "label": label,
            "section": header,
            "parent": parent_header,
            "versions": [
                {
                    "version_num": v["version_num"],
                    "operation": v["operation"],
                    "diff_data": json.loads(v.get("encrypted__diff_data", "{}")) if v.get("encrypted__diff_data") else {},
                    "created_at": v.get("created_at")
                }
                for v in versions
            ]
        }

    def _action_rollback_section(self, db, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback a section or subsection to a previous version using stored previous_content."""
        import json
        label = data["label"]
        header = data["section"]
        version_num = data["version_num"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Find the target version
        result = db.fetchone(
            "SELECT * FROM domaindoc_versions WHERE domaindoc_id = :doc_id AND section_id = :sec_id AND version_num = :ver",
            {"doc_id": doc["id"], "sec_id": section["id"], "ver": version_num}
        )
        if not result:
            raise ValidationError(f"Version {version_num} not found for section '{header}'")

        version = db._decrypt_dict(result)
        diff_data = json.loads(version.get("encrypted__diff_data", "{}"))

        if "previous_content" not in diff_data:
            raise ValidationError(f"Version {version_num} does not contain restorable content")

        # Restore the content
        restored_content = diff_data["previous_content"]
        now = format_utc_iso(utc_now())

        db.update(
            "domaindoc_sections",
            {"encrypted__content": restored_content, "updated_at": now},
            "id = :id",
            {"id": section["id"]}
        )

        # Record the rollback as a new version
        next_ver_result = db.fetchone(
            "SELECT MAX(version_num) as max_ver FROM domaindoc_versions WHERE domaindoc_id = :doc_id",
            {"doc_id": doc["id"]}
        )
        next_ver = (next_ver_result.get("max_ver") or 0) + 1

        db.insert("domaindoc_versions", {
            "domaindoc_id": doc["id"],
            "section_id": section["id"],
            "version_num": next_ver,
            "operation": "rollback",
            "encrypted__diff_data": json.dumps({
                "rolled_back_to": version_num,
                "restored_length": len(restored_content),
                "parent": parent_header
            }),
            "created_at": now
        })

        return {
            "rolled_back": True,
            "label": label,
            "section": header,
            "parent": parent_header,
            "to_version": version_num,
            "new_version": next_ver,
            "restored_chars": len(restored_content)
        }

    def _expand_description(self, label: str, description: str) -> str:
        """Use LLM to expand a brief description into comprehensive guidance."""
        from clients.llm_provider import LLMProvider

        try:
            llm = LLMProvider()

            prompt = f"""You are helping expand a brief description into comprehensive guidance for a knowledge document.

The user wants to create a domain knowledge document called "{label}" with this description:
"{description}"

Expand this into a well-formed descriptor (1-3 sentences) that clearly explains:
1. What topics this document covers
2. What specific information should be recorded
3. The level of detail expected

Write ONLY the expanded description, nothing else. Be specific and actionable.
Example input: "plants, bugs, where I buy stuff"
Example output: "Backyard garden management: current plantings with locations and planting dates, pest and disease observations with treatments applied, soil amendments and fertilizer schedules, preferred suppliers and product recommendations, and seasonal lessons learned."
"""

            response = llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                thinking_enabled=False
            )

            expanded = llm.extract_text_content(response).strip()

            if expanded and len(expanded) > 20:
                return expanded
            else:
                logger.warning(f"LLM expansion too short, using original: {expanded}")
                return description

        except Exception as e:
            logger.warning(f"Failed to expand description via LLM: {e}")
            return description


class ContinuumDomainHandler(BaseDomainHandler):
    """Handler for continuum-level configuration actions (LLM tier, segment collapse)."""

    GROQ_CHAT_COMPLETIONS_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    MODEL_ALIASES = {
        "gpt-openai/gpt-oss-120b": "openai/gpt-oss-120b",
    }

    ACTIONS = {
        "get_llm_tier": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "set_llm_tier": {
            "required": ["tier"],
            "optional": [],
            "types": {
                "tier": str
            }
        },
        "get_tier_provider_status": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "set_provider_key": {
            "required": ["selection_id", "api_key"],
            "optional": [],
            "types": {
                "selection_id": int,
                "api_key": str
            }
        },
        "set_tier_provider_selection": {
            "required": ["tier", "selection_id"],
            "optional": [],
            "types": {
                "tier": str,
                "selection_id": int
            }
        },
        "set_tier_model": {
            "required": ["tier", "model"],
            "optional": [],
            "types": {
                "tier": str,
                "model": str
            }
        },
        "set_tier_thinking_budget": {
            "required": ["tier", "thinking_budget"],
            "optional": [],
            "types": {
                "tier": str,
                "thinking_budget": int
            }
        },
        "reload_tier_cache": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "collapse_segment": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "postpone_collapse": {
            "required": ["minutes"],
            "optional": [],
            "types": {
                "minutes": int
            }
        },
        "get_segment_status": {
            "required": [],
            "optional": [],
            "types": {}
        }
    }

    def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM tier actions."""
        if action == "get_llm_tier":
            from utils.user_context import get_user_preferences, get_account_tiers, get_accessible_tiers

            prefs = get_user_preferences()
            all_tiers = get_account_tiers()
            accessible = get_accessible_tiers(prefs.max_tier)
            accessible_names = {t.name for t in accessible}

            # Return tiers with access info: accessible ones + locked ones with show_locked=true
            tier_list = []
            for tier in sorted(all_tiers.values(), key=lambda t: t.display_order):
                is_accessible = tier.name in accessible_names
                # Show tier if: accessible OR (not accessible but show_locked=true)
                if is_accessible or tier.show_locked:
                    tier_list.append({
                        "name": tier.name,
                        "model": tier.model,
                        "description": tier.description,
                        "accessible": is_accessible,
                        "locked_message": tier.locked_message if not is_accessible else None
                    })

            return {
                "success": True,
                "tier": prefs.llm_tier,
                "max_tier": prefs.max_tier,
                "available_tiers": tier_list
            }

        elif action == "set_llm_tier":
            from utils.user_context import (
                update_user_preference, get_account_tiers,
                can_access_tier, get_user_preferences
            )

            tier = data.get("tier")
            tiers = get_account_tiers()
            if tier not in tiers:
                raise ValidationError(
                    f"Invalid tier. Must be one of: {list(tiers.keys())}"
                )

            prefs = get_user_preferences()
            if not can_access_tier(tier, prefs.max_tier):
                raise ValidationError(
                    f"Tier '{tier}' not available. Your account has access up to '{prefs.max_tier}'."
                )

            update_user_preference('llm_tier', tier)
            return {
                "success": True,
                "tier": tier,
                "message": f"LLM tier set to {tier}"
            }

        elif action == "get_tier_provider_status":
            from clients.postgres_client import PostgresClient
            from clients.vault_client import has_api_key

            db = PostgresClient('mira_service')
            rows = db.execute_query(
                "SELECT name, provider, endpoint_url, api_key_name, model, thinking_budget "
                "FROM account_tiers WHERE name IN (%s, %s, %s) ORDER BY display_order",
                ("fast", "balanced", "oss")
            )

            tiers = []
            for row in rows:
                api_key_name = row.get("api_key_name")
                tiers.append({
                    "name": row["name"],
                    "provider": row["provider"],
                    "endpoint_url": row.get("endpoint_url"),
                    "api_key_name": api_key_name,
                    "model": row["model"],
                    "thinking_budget": row["thinking_budget"],
                    "has_api_key": has_api_key(api_key_name) if api_key_name else None
                })

            return {
                "success": True,
                "tiers": tiers
            }

        elif action == "set_provider_key":
            from clients.vault_client import set_api_key

            selection_id = data.get("selection_id")
            api_key = (data.get("api_key") or "").strip()

            if selection_id < 1:
                raise ValidationError("selection_id must be >= 1")
            if not api_key:
                raise ValidationError("api_key must be non-empty")

            key_name = f"provider_key_{selection_id}"
            try:
                set_api_key(key_name, api_key)
            except Exception as e:
                raise ValidationError(f"Vault write failed: {e.__class__.__name__}")

            return {
                "success": True,
                "key_name": key_name
            }

        elif action == "set_tier_provider_selection":
            from clients.postgres_client import PostgresClient
            import utils.user_context as user_context

            tier = (data.get("tier") or "").strip().lower()
            selection_id = data.get("selection_id")

            if tier not in {"fast", "balanced", "oss"}:
                raise ValidationError("tier must be one of: fast, balanced, oss")
            if selection_id < 1:
                raise ValidationError("selection_id must be >= 1")

            key_name = f"provider_key_{selection_id}"
            db = PostgresClient('mira_service')
            rows_updated = db.execute_update(
                "UPDATE account_tiers SET provider=%s, endpoint_url=%s, api_key_name=%s WHERE name=%s",
                ("generic", self.GROQ_CHAT_COMPLETIONS_ENDPOINT, key_name, tier)
            )
            if rows_updated != 1:
                raise ValidationError(f"Failed to update tier provider selection for '{tier}'")

            user_context._tiers_cache = None

            return {
                "success": True,
                "tier": tier,
                "endpoint_url": self.GROQ_CHAT_COMPLETIONS_ENDPOINT,
                "api_key_name": key_name
            }

        elif action == "set_tier_model":
            from clients.postgres_client import PostgresClient
            import utils.user_context as user_context

            tier = (data.get("tier") or "").strip().lower()
            model = (data.get("model") or "").strip()
            model = self.MODEL_ALIASES.get(model, model)

            if tier not in {"fast", "balanced", "oss"}:
                raise ValidationError("tier must be one of: fast, balanced, oss")
            if not model:
                raise ValidationError("model must be non-empty")

            db = PostgresClient('mira_service')
            rows_updated = db.execute_update(
                "UPDATE account_tiers SET model=%s WHERE name=%s",
                (model, tier)
            )
            if rows_updated != 1:
                raise ValidationError(f"Failed to update model for tier '{tier}'")

            user_context._tiers_cache = None

            return {
                "success": True,
                "tier": tier,
                "model": model
            }

        elif action == "set_tier_thinking_budget":
            from clients.postgres_client import PostgresClient
            import utils.user_context as user_context

            tier = (data.get("tier") or "").strip().lower()
            thinking_budget = data.get("thinking_budget")

            if tier not in {"fast", "balanced", "oss"}:
                raise ValidationError("tier must be one of: fast, balanced, oss")
            if thinking_budget is None or thinking_budget < 0:
                raise ValidationError("thinking_budget must be an integer >= 0")

            db = PostgresClient('mira_service')
            rows_updated = db.execute_update(
                "UPDATE account_tiers SET thinking_budget=%s WHERE name=%s",
                (thinking_budget, tier)
            )
            if rows_updated != 1:
                raise ValidationError(f"Failed to update thinking budget for tier '{tier}'")

            user_context._tiers_cache = None

            return {
                "success": True,
                "tier": tier,
                "thinking_budget": thinking_budget
            }

        elif action == "reload_tier_cache":
            import utils.user_context as user_context
            user_context._tiers_cache = None
            return {"success": True}

        elif action == "collapse_segment":
            from cns.infrastructure.continuum_pool import get_continuum_pool
            from cns.infrastructure.continuum_repository import get_continuum_repository
            from cns.core.events import SegmentTimeoutEvent
            from cns.services.segment_collapse_handler import get_segment_collapse_handler

            # Get user's continuum and active segment
            continuum_pool = get_continuum_pool()
            continuum = continuum_pool.get_or_create()

            continuum_repo = get_continuum_repository()
            sentinel = continuum_repo.find_active_segment(continuum.id, self.user_id)

            if not sentinel:
                raise NotFoundError("segment", "active")

            segment_id = sentinel.metadata.get("segment_id")

            # Create event and invoke the collapse handler
            event = SegmentTimeoutEvent.create(
                continuum_id=str(continuum.id),
                user_id=self.user_id,
                segment_id=segment_id,
                inactive_duration_minutes=0,  # Manual trigger
                local_hour=utc_now().hour
            )

            handler = get_segment_collapse_handler()
            collapsed_sentinel = handler.handle_timeout(event)

            if not collapsed_sentinel:
                raise ValidationError("Segment collapse failed - check server logs")

            return {
                "collapsed": True,
                "segment_id": segment_id,
                "summary": collapsed_sentinel.metadata.get("segment_summary"),
                "title": collapsed_sentinel.metadata.get("display_title"),
                "message": "Segment collapsed successfully"
            }

        elif action == "postpone_collapse":
            from cns.infrastructure.continuum_pool import get_continuum_pool
            from cns.infrastructure.continuum_repository import get_continuum_repository
            from datetime import timedelta

            minutes = data.get("minutes")
            if not (1 <= minutes <= 1440):
                raise ValidationError("minutes must be between 1 and 1440 (24 hours)")

            # Get user's continuum and active segment
            continuum_pool = get_continuum_pool()
            continuum = continuum_pool.get_or_create()

            continuum_repo = get_continuum_repository()
            sentinel = continuum_repo.find_active_segment(continuum.id, self.user_id)

            if not sentinel:
                raise NotFoundError("segment", "active")

            segment_id = sentinel.metadata.get("segment_id")

            # Stack extensions: add to existing virtual time if it's in the future
            existing_virtual_str = sentinel.metadata.get("virtual_last_message_time")
            now = utc_now()

            if existing_virtual_str:
                from datetime import datetime
                existing_virtual = datetime.fromisoformat(existing_virtual_str)
                if existing_virtual > now:
                    # Add to existing future time
                    virtual_time = existing_virtual + timedelta(minutes=minutes)
                else:
                    # Existing time is in the past, start fresh
                    virtual_time = now + timedelta(minutes=minutes)
            else:
                virtual_time = now + timedelta(minutes=minutes)
            success = continuum_repo.set_segment_virtual_last_message_time(
                continuum.id,
                self.user_id,
                virtual_time
            )

            if not success:
                raise ValidationError("Failed to set postpone - no active segment found")

            return {
                "postponed": True,
                "segment_id": segment_id,
                "virtual_last_message_time": format_utc_iso(virtual_time),
                "minutes": minutes,
                "message": f"Segment collapse postponed for {minutes} minutes"
            }

        elif action == "get_segment_status":
            from cns.infrastructure.continuum_pool import get_continuum_pool
            from cns.infrastructure.continuum_repository import get_continuum_repository
            from datetime import timedelta
            from config import config

            # Get user's continuum and active segment
            continuum_pool = get_continuum_pool()
            continuum = continuum_pool.get_or_create()

            continuum_repo = get_continuum_repository()
            sentinel = continuum_repo.find_active_segment(continuum.id, self.user_id)

            if not sentinel:
                return {
                    "has_active_segment": False,
                    "segment_id": None,
                    "collapse_at": None,
                    "is_postponed": False
                }

            segment_id = sentinel.metadata.get("segment_id")
            virtual_time_str = sentinel.metadata.get("virtual_last_message_time")

            # Get last real message time from segment messages
            segment_messages = continuum_repo.load_segment_messages(
                continuum.id, self.user_id, sentinel.created_at
            )
            if segment_messages:
                last_activity = segment_messages[-1].created_at
            else:
                last_activity = sentinel.created_at

            # Check if virtual time extends beyond actual activity
            is_postponed = False
            if virtual_time_str:
                from datetime import datetime
                virtual_time = datetime.fromisoformat(virtual_time_str)
                if virtual_time > last_activity:
                    last_activity = virtual_time
                    is_postponed = True

            # Calculate collapse time
            timeout_minutes = config.system.segment_timeout
            collapse_at = last_activity + timedelta(minutes=timeout_minutes)

            return {
                "has_active_segment": True,
                "segment_id": segment_id,
                "last_activity": format_utc_iso(last_activity),
                "collapse_at": format_utc_iso(collapse_at),
                "timeout_minutes": timeout_minutes,
                "is_postponed": is_postponed
            }

        else:
            raise ValidationError(f"Unknown action: {action}")


class ActionsEndpoint(BaseHandler):
    """Main actions endpoint handler with domain-based routing."""

    def __init__(self):
        super().__init__()
        self.domain_handlers = {
            DomainType.REMINDER: ReminderDomainHandler,
            DomainType.MEMORY: MemoryDomainHandler,
            DomainType.USER: UserDomainHandler,
            DomainType.CONTACTS: ContactsDomainHandler,
            DomainType.DOMAIN_KNOWLEDGE: DomainKnowledgeDomainHandler,
            DomainType.CONTINUUM: ContinuumDomainHandler
        }
    
    def process_request(self, **params) -> Dict[str, Any]:
        """Route request to appropriate domain handler."""
        current_user = params['current_user']
        user_id = current_user['user_id']
        request_data = params['request_data']
        
        # Set user context for any functions that need it
        from utils.user_context import set_current_user_id
        set_current_user_id(user_id)
        
        domain = request_data.domain
        action = request_data.action
        data = request_data.data
        
        # Get domain handler
        handler_class = self.domain_handlers.get(domain)
        if not handler_class:
            raise ValidationError(f"Unknown domain: {domain}")

        # Create handler instance
        handler = handler_class()
        
        # Validate action and data
        validated_data = handler.validate_action(action, data)
        
        # Execute action
        result = handler.execute_action(action, validated_data)
        
        # Add metadata
        result["meta"] = {
            "domain": domain.value,
            "action": action,
            "timestamp": format_utc_iso(utc_now())
        }
        
        return result


def get_actions_handler() -> ActionsEndpoint:
    """Get actions endpoint handler instance."""
    return ActionsEndpoint()


@router.post("/actions")
async def actions_endpoint(
    request_data: ActionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute state-changing operations through domain-routed actions."""
    try:
        handler = get_actions_handler()
        response = handler.handle_request(request_data=request_data, current_user=current_user)
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
    except NotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": e.message
                }
            }
        )
    except NotImplementedError as e:
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "error": {
                    "code": "NOT_IMPLEMENTED",
                    "message": str(e)
                }
            }
        )
    except Exception as e:
        logger.error(f"Actions endpoint error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Action execution failed"
                }
            }
        )


# =============================================================================
# TOOL QUERY ENDPOINT
# =============================================================================

# Whitelist of tools that can be queried directly via API
QUERYABLE_TOOLS = {"reminder_tool", "contacts_tool"}


def _get_tool_instance(tool_name: str):
    """Import and instantiate a tool by name."""
    if tool_name == "reminder_tool":
        from tools.implementations.reminder_tool import ReminderTool
        return ReminderTool()
    elif tool_name == "contacts_tool":
        from tools.implementations.contacts_tool import ContactsTool
        return ContactsTool()
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


@router.get("/tools/{tool_name}/query")
async def query_tool(
    tool_name: str,
    operation: str = Query(..., description="Tool operation to execute"),
    date_type: Optional[str] = Query(None, description="Date filter type (for reminder_tool)"),
    category: Optional[str] = Query(None, description="Category filter (for reminder_tool)"),
    current_user: dict = Depends(get_current_user)
):
    """
    Query a tool directly for read-only operations.

    This endpoint allows the UI to query tool data without going through
    the LLM or trinket system. Useful for polling current state.

    Only whitelisted tools can be queried.
    """
    # Set user context
    set_current_user_id(current_user["user_id"])

    if tool_name not in QUERYABLE_TOOLS:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": {
                    "code": "FORBIDDEN",
                    "message": f"Tool '{tool_name}' is not queryable via API"
                }
            }
        )

    try:
        tool = _get_tool_instance(tool_name)

        # Build kwargs from query params (only include non-None values)
        kwargs = {"operation": operation}
        if date_type is not None:
            kwargs["date_type"] = date_type
        if category is not None:
            kwargs["category"] = category

        result = tool.run(**kwargs)

        return {
            "success": True,
            "data": result,
            "meta": {
                "tool": tool_name,
                "operation": operation,
                "timestamp": format_utc_iso(utc_now())
            }
        }

    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(e)
                }
            }
        )
    except Exception as e:
        logger.error(f"Tool query error for {tool_name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }
        )
