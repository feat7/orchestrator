"""Google Drive service with mock and real implementations."""

from typing import Optional
import uuid
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.db.models import GdriveCache
from app.config import settings


class DriveService:
    """Service for Google Drive operations with mock and real implementations."""

    def __init__(self, db: AsyncSession, credentials: Optional[Credentials] = None):
        """Initialize the Drive service.

        Args:
            db: Database session
            credentials: Google OAuth credentials (None for mock mode)
        """
        self.db = db
        self.credentials = credentials
        self._service = None

    @property
    def service(self):
        """Lazy-load Drive API service."""
        if self._service is None and self.credentials:
            self._service = build("drive", "v3", credentials=self.credentials)
        return self._service

    async def search_files(
        self,
        user_id: str,
        embedding: list[float],
        filters: Optional[dict] = None,
        limit: int = 10,
        similarity_threshold: float = 0.25,
    ) -> list[dict]:
        """Search files using local cache with vector similarity.

        Always searches the local pgvector cache first (fast semantic search).
        This uses embeddings generated during sync for semantic similarity ranking.

        Args:
            user_id: The user's ID
            embedding: Query embedding vector
            filters: Optional filters (mime_type, modified_date, name)
            limit: Max results to return
            similarity_threshold: Minimum similarity score (0-1) to include results

        Returns:
            List of matching files with similarity scores
        """
        # Calculate cosine distance and convert to similarity (1 - distance)
        distance_expr = GdriveCache.embedding.cosine_distance(embedding)

        # Always use local pgvector search for speed and semantic matching
        query = select(
            GdriveCache,
            (1 - distance_expr).label("similarity")
        ).where(
            GdriveCache.user_id == uuid.UUID(user_id)
        )

        # Apply metadata filters
        if filters:
            if filters.get("mime_type"):
                query = query.where(
                    GdriveCache.mime_type.ilike(f"%{filters['mime_type']}%")
                )
            if filters.get("name"):
                query = query.where(
                    GdriveCache.name.ilike(f"%{filters['name']}%")
                )
            if filters.get("modified_after"):
                query = query.where(
                    GdriveCache.modified_at >= filters["modified_after"]
                )
            if filters.get("modified_before"):
                query = query.where(
                    GdriveCache.modified_at <= filters["modified_before"]
                )

        # Order by similarity (descending) and limit
        query = query.order_by(distance_expr).limit(limit * 2)

        result = await self.db.execute(query)
        rows = result.all()

        # Filter by similarity threshold and format results
        files = []
        for row in rows:
            file = row[0]
            similarity = float(row[1]) if row[1] is not None else 0

            # Skip results below threshold
            if similarity < similarity_threshold:
                continue

            files.append({
                "id": str(file.file_id),
                "name": file.name,
                "mime_type": file.mime_type,
                "content_preview": file.content_preview,
                "web_link": file.web_link,
                "modified_at": file.modified_at.isoformat() if file.modified_at else None,
                "owners": file.owners,
                "similarity": round(similarity, 3),
            })

            if len(files) >= limit:
                break

        return files

    async def search_files_bm25(
        self,
        user_id: str,
        query: str,
        filters: Optional[dict] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search files using BM25/full-text search (keyword matching).

        Uses PostgreSQL's tsvector and ts_rank for fast keyword-based search.

        Args:
            user_id: The user's ID
            query: Search query string
            filters: Optional filters (mime_type, name, modified_after, modified_before)
            limit: Max results to return

        Returns:
            List of matching files with BM25 rank scores
        """
        words = query.lower().split()
        if not words:
            return []

        tsquery = func.plainto_tsquery("english", query)

        query_stmt = (
            select(
                GdriveCache,
                func.ts_rank(GdriveCache.search_vector, tsquery).label("rank")
            )
            .where(GdriveCache.user_id == uuid.UUID(user_id))
            .where(GdriveCache.search_vector.op("@@")(tsquery))
        )

        # Apply metadata filters (same as vector search)
        if filters:
            if filters.get("mime_type"):
                query_stmt = query_stmt.where(
                    GdriveCache.mime_type.ilike(f"%{filters['mime_type']}%")
                )
            if filters.get("name"):
                query_stmt = query_stmt.where(
                    GdriveCache.name.ilike(f"%{filters['name']}%")
                )
            if filters.get("modified_after"):
                query_stmt = query_stmt.where(
                    GdriveCache.modified_at >= filters["modified_after"]
                )
            if filters.get("modified_before"):
                query_stmt = query_stmt.where(
                    GdriveCache.modified_at <= filters["modified_before"]
                )

        query_stmt = (
            query_stmt
            .order_by(func.ts_rank(GdriveCache.search_vector, tsquery).desc())
            .limit(limit)
        )

        result = await self.db.execute(query_stmt)
        rows = result.all()

        files = []
        for row in rows:
            file = row[0]
            rank = float(row[1]) if row[1] is not None else 0

            files.append({
                "id": str(file.file_id),
                "name": file.name,
                "mime_type": file.mime_type,
                "content_preview": file.content_preview,
                "web_link": file.web_link,
                "modified_at": file.modified_at.isoformat() if file.modified_at else None,
                "owners": file.owners,
                "bm25_rank": round(rank, 4),
            })

        return files

    async def get_file(self, user_id: str, file_id: str) -> Optional[dict]:
        """Get full file metadata and content preview.

        Args:
            user_id: The user's ID
            file_id: The file ID

        Returns:
            File data dictionary or None
        """
        if not settings.use_mock_google and self.service:
            try:
                file = self.service.files().get(
                    fileId=file_id,
                    fields="id, name, mimeType, webViewLink, modifiedTime, createdTime, owners, parents, description, permissions",
                ).execute()

                return {
                    "id": file["id"],
                    "name": file.get("name", "Untitled"),
                    "mime_type": file.get("mimeType", ""),
                    "content_preview": file.get("description", ""),
                    "parent_folder": file.get("parents", [None])[0],
                    "web_link": file.get("webViewLink", ""),
                    "owners": [o.get("emailAddress", "") for o in file.get("owners", [])],
                    "shared_with": [
                        p.get("emailAddress", "")
                        for p in file.get("permissions", [])
                        if p.get("emailAddress")
                    ],
                    "created_at": file.get("createdTime"),
                    "modified_at": file.get("modifiedTime"),
                }
            except Exception as e:
                print(f"Drive API error: {e}")
                return None

        # Mock mode: use local cache
        result = await self.db.execute(
            select(GdriveCache).where(
                GdriveCache.user_id == uuid.UUID(user_id),
                GdriveCache.file_id == file_id,
            )
        )
        file = result.scalar_one_or_none()

        if file:
            return {
                "id": file.file_id,
                "name": file.name,
                "mime_type": file.mime_type,
                "content_preview": file.content_preview,
                "parent_folder": file.parent_folder,
                "web_link": file.web_link,
                "owners": file.owners,
                "shared_with": file.shared_with,
                "created_at": (
                    file.created_at.isoformat() if file.created_at else None
                ),
                "modified_at": (
                    file.modified_at.isoformat() if file.modified_at else None
                ),
            }
        return None

    async def share_file(
        self, user_id: str, file_id: str, email: str, role: str = "reader"
    ) -> dict:
        """Share a file with another user.

        Args:
            user_id: The owner's user ID
            file_id: The file ID to share
            email: Email to share with
            role: Permission role (reader, writer, commenter)

        Returns:
            Share result data
        """
        if not settings.use_mock_google and self.service:
            try:
                permission = {
                    "type": "user",
                    "role": role,
                    "emailAddress": email,
                }

                result = self.service.permissions().create(
                    fileId=file_id,
                    body=permission,
                    sendNotificationEmail=True,
                ).execute()

                return {
                    "file_id": file_id,
                    "permission_id": result.get("id"),
                    "shared_with": email,
                    "role": role,
                    "shared_at": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                print(f"Drive API error sharing file: {e}")
                raise

        # Mock mode
        return {
            "file_id": file_id,
            "shared_with": email,
            "role": role,
            "shared_at": datetime.utcnow().isoformat(),
        }

    async def create_folder(self, user_id: str, name: str, parent_id: str = None) -> dict:
        """Create a new folder.

        Args:
            user_id: The user's ID
            name: Folder name
            parent_id: Optional parent folder ID

        Returns:
            Created folder data
        """
        if not settings.use_mock_google and self.service:
            try:
                file_metadata = {
                    "name": name,
                    "mimeType": "application/vnd.google-apps.folder",
                }

                if parent_id:
                    file_metadata["parents"] = [parent_id]

                folder = self.service.files().create(
                    body=file_metadata,
                    fields="id, name, mimeType, webViewLink",
                ).execute()

                return {
                    "id": folder["id"],
                    "name": folder.get("name"),
                    "mime_type": folder.get("mimeType"),
                    "web_link": folder.get("webViewLink"),
                    "parent_id": parent_id,
                }
            except Exception as e:
                print(f"Drive API error creating folder: {e}")
                raise

        # Mock mode
        return {
            "id": f"folder_{uuid.uuid4().hex[:8]}",
            "name": name,
            "mime_type": "application/vnd.google-apps.folder",
            "parent_id": parent_id,
        }

    async def move_file(
        self, user_id: str, file_id: str, new_parent_id: str
    ) -> dict:
        """Move a file to a different folder.

        Args:
            user_id: The user's ID
            file_id: The file ID to move
            new_parent_id: The destination folder ID

        Returns:
            Move result data
        """
        if not settings.use_mock_google and self.service:
            try:
                # Get current parents
                file = self.service.files().get(
                    fileId=file_id,
                    fields="parents",
                ).execute()

                previous_parents = ",".join(file.get("parents", []))

                # Move file
                result = self.service.files().update(
                    fileId=file_id,
                    addParents=new_parent_id,
                    removeParents=previous_parents,
                    fields="id, parents",
                ).execute()

                return {
                    "file_id": file_id,
                    "new_parent_id": new_parent_id,
                    "moved": True,
                    "previous_parents": previous_parents.split(","),
                }
            except Exception as e:
                print(f"Drive API error moving file: {e}")
                raise

        # Mock mode
        return {
            "file_id": file_id,
            "new_parent_id": new_parent_id,
            "moved": True,
        }
