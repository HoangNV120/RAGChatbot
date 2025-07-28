import os
import io
import json
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import asyncio
import logging

# Scopes required for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveSync:
    def __init__(self,
                 service_account_file=None,
                 data_dir="app/data",
                 max_retries=3,
                 retry_delay=1):
        """
        Initialize Google Drive sync client using Service Account

        Args:
            service_account_file (str): Path to service account JSON file
            data_dir (str): Directory to download files to
            max_retries (int): Maximum number of retry attempts for API calls
            retry_delay (int): Delay between retry attempts in seconds
        """
        self.service_account_file = service_account_file or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service-account.json")
        self.data_dir = data_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.service = None
        self.logger = logging.getLogger(__name__)

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # File to store sync metadata
        self.sync_metadata_file = os.path.join(self.data_dir, "sync_metadata.json")
        self.sync_metadata = self._load_sync_metadata()

        self.logger.info(f"🔧 Google Drive authentication: Service Account ({self.service_account_file})")

    def _load_sync_metadata(self) -> Dict:
        """Load sync metadata from file"""
        try:
            if os.path.exists(self.sync_metadata_file):
                with open(self.sync_metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading sync metadata: {e}")
        return {}

    def _save_sync_metadata(self):
        """Save sync metadata to file"""
        try:
            with open(self.sync_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.sync_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving sync metadata: {e}")

    def _normalize_timestamp_to_utc(self, timestamp_str: str) -> str:
        """
        Normalize timestamp to UTC format for consistent comparison

        Args:
            timestamp_str (str): Timestamp string in various formats

        Returns:
            str: Normalized UTC timestamp
        """
        try:
            # Try parsing different timestamp formats
            if timestamp_str.endswith('Z'):
                # Already UTC format
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            elif '+' in timestamp_str or timestamp_str.count(':') > 2:
                # Has timezone info
                dt = datetime.fromisoformat(timestamp_str)
            else:
                # Assume local time (UTC+7 for Vietnam)
                dt = datetime.fromisoformat(timestamp_str)
                # If no timezone info, assume it's local time (UTC+7)
                if dt.tzinfo is None:
                    # Use replace() to set timezone properly
                    dt = dt.replace(tzinfo=timezone(timedelta(hours=7)))

            # Convert to UTC using replace() for consistent timezone handling
            if dt.tzinfo is None:
                # If still no timezone, assume UTC
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC
                dt = dt.astimezone(timezone.utc)

            # Return normalized UTC timestamp
            return dt.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')

        except Exception as e:
            self.logger.error(f"Error normalizing timestamp {timestamp_str}: {e}")
            return timestamp_str

    async def authenticate(self):
        """Authenticate using Service Account"""
        try:
            if not self.service_account_file:
                raise FileNotFoundError("Service account file not specified")

            if not os.path.exists(self.service_account_file):
                raise FileNotFoundError(f"Service account file not found: {self.service_account_file}")

            self.logger.info(f"🔑 Authenticating with service account: {self.service_account_file}")

            # Load service account credentials
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_file,
                scopes=SCOPES
            )

            # Build service
            self.service = build('drive', 'v3', credentials=creds)
            self.logger.info("✅ Service Account authentication successful")
            return True

        except Exception as e:
            self.logger.error(f"❌ Service Account authentication failed: {e}")
            return False

    async def _retry_api_call(self, func, *args, **kwargs):
        """
        Retry API calls with exponential backoff

        Args:
            func: The function to retry
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function call
        """
        for attempt in range(self.max_retries):
            try:
                return await asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs)
            except (HttpError, ConnectionError, Exception) as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed, raise the exception
                    raise e

                # Calculate delay with exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"⚠️ API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                self.logger.info(f"🔄 Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        raise Exception("Max retries exceeded")

    async def list_files_in_folder(self, folder_id: str, file_types: List[str] = None) -> List[Dict]:
        """
        List files in a Google Drive folder with retry logic

        Args:
            folder_id (str): Google Drive folder ID
            file_types (list): List of file extensions to filter (e.g., ['json', 'pdf', 'xlsx'])

        Returns:
            List of file information dictionaries, or raises exception on failure
        """
        if not self.service:
            raise Exception("Not authenticated with Google Drive")

        # Build query for files in folder
        query = f"'{folder_id}' in parents and trashed=false"

        # Add file type filter if specified
        if file_types:
            mime_conditions = []
            for file_type in file_types:
                if file_type.lower() == 'json':
                    mime_conditions.append("mimeType='application/json'")
                elif file_type.lower() == 'pdf':
                    mime_conditions.append("mimeType='application/pdf'")
                elif file_type.lower() in ['xlsx', 'xls']:
                    mime_conditions.append("mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'")
                elif file_type.lower() == 'txt':
                    mime_conditions.append("mimeType='text/plain'")

            if mime_conditions:
                query += f" and ({' or '.join(mime_conditions)})"

        # Execute query with retry logic
        results = await self._retry_api_call(
            lambda: self.service.files().list(
                q=query,
                fields="files(id,name,modifiedTime,size,mimeType)"
            ).execute()
        )

        files = results.get('files', [])
        self.logger.info(f"📁 Found {len(files)} files in Google Drive folder")
        return files

    async def download_file(self, file_id: str, file_name: str) -> bool:
        """
        Download a file from Google Drive with retry logic

        Args:
            file_id (str): Google Drive file ID
            file_name (str): Name to save the file as

        Returns:
            bool: True if download successful
        """
        if not self.service:
            raise Exception("Not authenticated with Google Drive")

        try:
            # Get file metadata with retry logic
            request = await self._retry_api_call(
                lambda: self.service.files().get_media(fileId=file_id)
            )

            # Download file
            file_path = os.path.join(self.data_dir, file_name)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while done is False:
                status, done = downloader.next_chunk()

            # Save to file
            with open(file_path, 'wb') as f:
                f.write(fh.getvalue())

            self.logger.info(f"✅ Downloaded: {file_name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error downloading {file_name}: {e}")
            return False

    async def sync_folder(self, folder_id: str, file_types: List[str] = None) -> Dict:
        """
        Sync files from Google Drive folder to local data directory

        Args:
            folder_id (str): Google Drive folder ID
            file_types (list): List of file extensions to sync

        Returns:
            Dict with sync results
        """
        if not await self.authenticate():
            return {"success": False, "error": "Authentication failed"}

        try:
            # Get files from Google Drive with retry logic
            drive_files = await self.list_files_in_folder(folder_id, file_types)

            sync_results = {
                "success": True,
                "total_files": len(drive_files),
                "downloaded": 0,
                "updated": 0,
                "skipped": 0,
                "deleted": 0,
                "errors": []
            }

            # Create a set of current Drive file names for deletion detection
            current_drive_files = set()

            for file_info in drive_files:
                file_id = file_info['id']
                file_name = file_info['name']
                modified_time = file_info['modifiedTime']

                # Add to current files set
                current_drive_files.add(file_name)

                # Check if file needs to be downloaded/updated
                should_download = False

                # Check if file exists locally
                local_file_path = os.path.join(self.data_dir, file_name)
                if file_name not in self.sync_metadata:
                    should_download = True
                    action = "download"
                else:
                    # Check if file was modified since last sync
                    file_metadata = self.sync_metadata.get(file_name, {})
                    last_modified_in_metadata = file_metadata.get('last_modified')

                    if last_modified_in_metadata is None:
                        # File exists locally but no metadata - need to sync
                        should_download = True
                        action = "download"
                    else:
                        # Normalize both timestamps to UTC for proper comparison
                        drive_modified_utc = self._normalize_timestamp_to_utc(modified_time)
                        metadata_modified_utc = self._normalize_timestamp_to_utc(last_modified_in_metadata)
                        print(f"Comparing timestamps - Drive: {drive_modified_utc}, Metadata: {metadata_modified_utc}")

                        if drive_modified_utc > metadata_modified_utc:
                            # File was modified on Drive after last sync
                            should_download = True
                            action = "update"
                            self.logger.info(f"🔄 File modified - Drive: {drive_modified_utc}, Last sync: {metadata_modified_utc}")
                        else:
                            # File hasn't changed since last sync - skip
                            action = "skip"

                if should_download:
                    success = await self.download_file(file_id, file_name)
                    if success:
                        # Update metadata
                        self.sync_metadata[file_name] = {
                            'file_id': file_id,
                            'last_modified': modified_time,
                            'last_sync': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                        }

                        if action == "download":
                            sync_results["downloaded"] += 1
                        else:
                            sync_results["updated"] += 1
                    else:
                        sync_results["errors"].append(f"Failed to download {file_name}")
                else:
                    sync_results["skipped"] += 1
                    self.logger.info(f"⏭️ Skipped (no changes): {file_name}")

            # Only handle file deletions if we successfully got the file list
            # This prevents data loss when API calls fail
            if len(drive_files) > 0 or len(self.sync_metadata) == 0:
                # Handle file deletions - check for files that were previously synced but no longer exist in Drive
                deleted_files = await self._handle_deleted_files(current_drive_files, file_types)
                sync_results["deleted"] = len(deleted_files)

                if deleted_files:
                    sync_results["deleted_files"] = deleted_files
            else:
                self.logger.warning("⚠️ Skipping file deletion check - no files returned from Drive (potential API issue)")

            # Save sync metadata
            self._save_sync_metadata()

            self.logger.info(f"🔄 Sync completed - Downloaded: {sync_results['downloaded']}, Updated: {sync_results['updated']}, Skipped: {sync_results['skipped']}, Deleted: {sync_results['deleted']}")
            return sync_results

        except Exception as e:
            self.logger.error(f"❌ Sync failed: {e}")
            return {"success": False, "error": str(e), "api_connection_failed": True}

    async def _handle_deleted_files(self, current_drive_files: set, file_types: List[str] = None) -> List[str]:
        """
        Handle files that were deleted from Google Drive

        Args:
            current_drive_files (set): Set of current file names in Google Drive
            file_types (list): List of file extensions to check

        Returns:
            List of deleted file names
        """
        deleted_files = []

        try:
            # Get list of files that were previously synced
            previously_synced_files = set(self.sync_metadata.keys())

            # Filter by file types if specified
            if file_types:
                filtered_synced_files = set()
                for file_name in previously_synced_files:
                    file_extension = file_name.split('.')[-1].lower()
                    if file_extension in [ft.lower() for ft in file_types]:
                        filtered_synced_files.add(file_name)
                previously_synced_files = filtered_synced_files

            # Find files that were synced before but are no longer in Drive
            files_to_delete = previously_synced_files - current_drive_files

            for file_name in files_to_delete:
                try:
                    # Remove from local directory if exists
                    local_file_path = os.path.join(self.data_dir, file_name)
                    if os.path.exists(local_file_path):
                        os.remove(local_file_path)
                        self.logger.info(f"🗑️ Deleted local file: {file_name}")

                    # Remove from sync metadata
                    if file_name in self.sync_metadata:
                        del self.sync_metadata[file_name]

                    deleted_files.append(file_name)
                    self.logger.info(f"✅ Cleaned up deleted file: {file_name}")

                except Exception as e:
                    self.logger.error(f"❌ Error cleaning up {file_name}: {e}")

            return deleted_files

        except Exception as e:
            self.logger.error(f"❌ Error handling deleted files: {e}")
            return []

    async def get_folder_id_by_name(self, folder_name: str, parent_folder_id: str = None) -> Optional[str]:
        """
        Get folder ID by name

        Args:
            folder_name (str): Name of the folder
            parent_folder_id (str): Parent folder ID (optional)

        Returns:
            str: Folder ID if found, None otherwise
        """
        if not self.service:
            raise Exception("Not authenticated with Google Drive")

        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent_folder_id:
                query += f" and '{parent_folder_id}' in parents"

            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.service.files().list(
                    q=query,
                    fields="files(id,name)"
                ).execute()
            )

            files = results.get('files', [])
            if files:
                return files[0]['id']
            return None

        except Exception as e:
            self.logger.error(f"❌ Error finding folder: {e}")
            return None

    def get_sync_status(self) -> Dict:
        """Get current sync status"""
        return {
            "last_sync_files": len(self.sync_metadata),
            "sync_metadata": self.sync_metadata,
            "data_directory": self.data_dir
        }
