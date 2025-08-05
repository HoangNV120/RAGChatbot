import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from app.google_drive_sync import GoogleDriveSync
from app.document_processor import DocumentProcessor
from app.vector_store import VectorStore
from app.syllabus_converter import SyllabusConverter
import threading
import json
import os
import re

class SmartDocumentScheduler:
    def __init__(self,
                 google_drive_folder_id: str,
                 data_dir: str = "app/data",
                 service_account_file: str = None,
                 schedule_interval_hours: int = 1):
        """
        Initialize Smart Document Scheduler

        Args:
            google_drive_folder_id (str): Google Drive folder ID to sync from
            data_dir (str): Local data directory
            service_account_file (str): Service account JSON file path
            schedule_interval_hours (int): How often to run sync (in hours)
        """
        self.google_drive_folder_id = google_drive_folder_id
        self.data_dir = data_dir
        self.schedule_interval_hours = schedule_interval_hours

        # Initialize components
        self.drive_sync = GoogleDriveSync(
            service_account_file=service_account_file,
            data_dir=data_dir
        )
        self.document_processor = DocumentProcessor(data_dir=data_dir)
        self.vector_store = VectorStore()
        self.syllabus_converter = SyllabusConverter()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Track processing status
        self.processing_log_file = os.path.join(data_dir, "processing_log.json")
        self.processing_log = self._load_processing_log()

        # Control flags
        self.is_running = False
        self.scheduler_thread = None

    def _load_processing_log(self) -> Dict:
        """Load processing log from file"""
        try:
            if os.path.exists(self.processing_log_file):
                with open(self.processing_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading processing log: {e}")
        return {}

    def _save_processing_log(self):
        """Save processing log to file"""
        try:
            with open(self.processing_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving processing log: {e}")

    async def sync_and_process(self) -> Dict:
        """
        Main function to sync from Google Drive and process documents

        Returns:
            Dict with processing results
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting scheduled sync and processing...")

        results = {
            "start_time": start_time.isoformat(),
            "sync_results": {},
            "processing_results": {},
            "deletion_results": {},
            "success": False,
            "errors": []
        }

        try:
            # Step 1: Sync from Google Drive
            self.logger.info("📥 Step 1: Syncing from Google Drive...")
            sync_results = await self.drive_sync.sync_folder(
                folder_id=self.google_drive_folder_id,
                file_types=['json', 'pdf', 'xlsx', 'xls']
            )
            results["sync_results"] = sync_results

            if not sync_results.get("success", False):
                results["errors"].append(f"Google Drive sync failed: {sync_results.get('error', 'Unknown error')}")
                return results

            # Step 2: Handle deleted files - remove from vector store
            deleted_files = sync_results.get("deleted_files", [])
            if deleted_files:
                self.logger.info(f"🗑️ Step 2: Processing {len(deleted_files)} deleted files...")
                deletion_results = await self._handle_deleted_documents(deleted_files)
                results["deletion_results"] = deletion_results

            # Step 3: Check if any files were downloaded/updated
            files_changed = sync_results.get("downloaded", 0) + sync_results.get("updated", 0)

            if files_changed == 0 and not deleted_files:
                self.logger.info("📄 No changes detected - skipping processing")
                results["success"] = True
                results["processing_results"] = {"message": "No files to process"}
                return results

            # Step 4: Smart document processing for new/updated files
            if files_changed > 0:
                self.logger.info(f"⚙️ Step 3: Processing {files_changed} changed files...")
                processing_results = await self._smart_process_documents()
                results["processing_results"] = processing_results

            # Step 5: Update processing log
            self._update_processing_log(sync_results, results.get("processing_results", {}))

            results["success"] = True
            results["end_time"] = datetime.now().isoformat()
            results["duration"] = str(datetime.now() - start_time)

            summary_msg = f"✅ Sync completed in {results['duration']}"
            if files_changed > 0:
                summary_msg += f" - Processed: {files_changed} files"
            if deleted_files:
                summary_msg += f" - Deleted: {len(deleted_files)} files"

            self.logger.info(summary_msg)

        except Exception as e:
            error_msg = f"Error in sync_and_process: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["end_time"] = datetime.now().isoformat()

        return results

    async def _handle_deleted_documents(self, deleted_files: List[str]) -> Dict:
        """
        Handle documents that need to be deleted from both main vector store and routing vector store

        Args:
            deleted_files (List[str]): List of deleted file names

        Returns:
            Dict with deletion results
        """
        deletion_results = {
            "total_files": len(deleted_files),
            "documents_deleted": 0,
            "routing_questions_deleted": 0,
            "errors": []
        }

        try:
            for file_name in deleted_files:
                try:
                    # Extract document name (remove file extension)
                    doc_name = self._extract_document_name(file_name)

                    if doc_name:
                        # Delete from main vector store
                        self.logger.info(f"🗑️ Deleting document from main vector store: {doc_name}")
                        main_success = await self.vector_store.delete_documents_by_name(doc_name)

                        if main_success:
                            deletion_results["documents_deleted"] += 1
                            self.logger.info(f"✅ Successfully deleted document: {doc_name}")
                        else:
                            error_msg = f"Failed to delete document: {doc_name}"
                            deletion_results["errors"].append(error_msg)
                            self.logger.warning(f"⚠️ {error_msg}")

                        # For Excel files, also delete from routing vector store
                        if file_name.lower().endswith(('.xlsx', '.xls')):
                            self.logger.info(f"🗑️ Deleting routing questions for Excel file: {file_name}")
                            routing_success = await self.document_processor.smart_router.routing_vector_store.delete_questions_by_source(file_name)

                            if routing_success:
                                deletion_results["routing_questions_deleted"] += 1
                                self.logger.info(f"✅ Successfully deleted routing questions for: {file_name}")
                            else:
                                error_msg = f"Failed to delete routing questions for: {file_name}"
                                deletion_results["errors"].append(error_msg)
                                self.logger.warning(f"⚠️ {error_msg}")

                    else:
                        error_msg = f"Could not extract document name from: {file_name}"
                        deletion_results["errors"].append(error_msg)
                        self.logger.warning(f"⚠️ {error_msg}")

                except Exception as e:
                    error_msg = f"Error deleting document for file {file_name}: {str(e)}"
                    deletion_results["errors"].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")

            return deletion_results

        except Exception as e:
            error_msg = f"Error in _handle_deleted_documents: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            deletion_results["errors"].append(error_msg)
            return deletion_results

    def _extract_document_name(self, file_name: str) -> str:
        """
        Extract document name from file name for vector store operations

        Args:
            file_name (str): Full file name with extension

        Returns:
            str: Document name for vector store
        """
        try:
            # Remove file extension
            name_without_ext = file_name.rsplit('.', 1)[0]

            # Handle different file types
            if file_name.lower().endswith('.json'):
                return name_without_ext
            elif file_name.lower().endswith('.pdf'):
                return name_without_ext
            elif file_name.lower().endswith(('.xlsx', '.xls')):
                # Excel files typically use "FQA" as document name
                return "FQA"
            else:
                return name_without_ext

        except Exception as e:
            self.logger.error(f"Error extracting document name from {file_name}: {e}")
            return None

    def _is_syllabus_file(self, filename: str) -> bool:
        """
        Determine if a file is a syllabus file based on its name
        
        Args:
            filename (str): Filename to check
            
        Returns:
            bool: True if the file is a syllabus, False otherwise
        """
        # Remove extension
        name_without_ext = filename.replace('.json', '')
        
        # Pattern for syllabus: 3 uppercase letters (may include Đ) + 3 digits + optional lowercase letter
        syllabus_pattern = r'^[A-ZĐ]{3}\d{3}[a-z]?$'
        
        # Check against pattern
        if re.match(syllabus_pattern, name_without_ext):
            return True
            
        # Check if name contains "LUK"
        if "LUK" in name_without_ext.upper():
            return True
            
        return False

    async def _smart_process_documents(self) -> Dict:
        """
        Smart document processing with file type detection and existing document handling

        Returns:
            Dict with processing results
        """
        processing_results = {
            "json_files": {"processed": 0, "updated": 0, "errors": []},
            "pdf_files": {"processed": 0, "updated": 0, "errors": []},
            "excel_files": {"processed": 0, "updated": 0, "errors": []},
            "total_documents": 0
        }

        try:
            # Get list of all document names in vector store
            existing_doc_names = await self.vector_store.list_all_document_names()
            self.logger.info(f"📋 Found {len(existing_doc_names)} existing documents in vector store")

            # Process JSON files
            json_results = await self._process_json_files_smart(existing_doc_names)
            processing_results["json_files"] = json_results

            # Process PDF files
            pdf_results = await self._process_pdf_files_smart(existing_doc_names)
            processing_results["pdf_files"] = pdf_results

            # Process Excel files
            excel_results = await self._process_excel_files_smart(existing_doc_names)
            processing_results["excel_files"] = excel_results

            # Calculate totals
            processing_results["total_documents"] = (
                json_results["processed"] +
                pdf_results["processed"] +
                excel_results["processed"]
            )

            return processing_results

        except Exception as e:
            error_msg = f"Error in smart processing: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            processing_results["errors"] = [error_msg]
            return processing_results

    async def _process_json_files_smart(self, existing_doc_names: List[str]) -> Dict:
        """Process JSON files with smart update detection"""
        results = {"processed": 0, "updated": 0, "errors": []}

        try:
            # Find JSON files in data directory, excluding system files
            json_files = []
            for f in os.listdir(self.data_dir):
                if f.endswith('.json'):
                    # Exclude system files
                    if f not in ['processing_log.json', 'sync_metadata.json']:
                        json_files.append(f)

            for json_file in json_files:
                try:
                    # Extract document name (filename without extension)
                    doc_name = json_file.replace('.json', '')
                    file_path = os.path.join(self.data_dir, json_file)

                    # Check if document exists in vector store
                    if doc_name in existing_doc_names:
                        # Update existing document
                        self.logger.info(f"🔄 Updating existing JSON document: {doc_name}")

                        new_documents = await self._process_single_syllabus_file(file_path, doc_name)

                        if new_documents:
                            # Update in vector store
                            success = await self.vector_store.update_documents_by_name(doc_name, new_documents)
                            if success:
                                results["updated"] += 1
                                # Delete processed file
                                os.remove(file_path)
                                self.logger.info(f"✅ Updated and deleted: {json_file}")
                            else:
                                results["errors"].append(f"Failed to update {doc_name} in vector store")
                        else:
                            results["errors"].append(f"Failed to process syllabus file {json_file}")
                    else:
                        # New document - process using syllabus converter
                        self.logger.info(f"📄 Processing new JSON document with intelligent chunking: {doc_name}")

                        new_documents = await self._process_single_syllabus_file(file_path, doc_name)

                        if new_documents:
                            # Add to vector store
                            success = await self.vector_store.add_documents(new_documents)
                            if success:
                                results["processed"] += 1
                                # Delete processed file
                                os.remove(file_path)
                                self.logger.info(f"✅ Processed new syllabus and deleted: {json_file}")
                            else:
                                results["errors"].append(f"Failed to add {doc_name} to vector store")
                        else:
                            results["errors"].append(f"Failed to process new syllabus file {json_file}")

                except Exception as e:
                    error_msg = f"Error processing JSON file {json_file}: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)

        except Exception as e:
            results["errors"].append(f"Error in JSON processing: {str(e)}")

        return results
        
    async def _process_single_syllabus_file(self, file_path: str, doc_name: str) -> List:
        """Process a single syllabus JSON file and return documents using the specialized syllabus converter with intelligent chunking"""
        try:
            # Use syllabus converter's improved process_syllabus_to_chunks method
            syllabus_chunks = self.syllabus_converter.process_syllabus_to_chunks(file_path)

            if not syllabus_chunks:
                self.logger.error(f"Failed to create intelligent chunks for syllabus file: {file_path}")
                return []

            # Add source file path to metadata of each chunk
            for chunk in syllabus_chunks:
                chunk.metadata.update({
                    "name": doc_name,
                    "type": "Syllabus",
                    "source": file_path
                })

            self.logger.info(f"Processed syllabus file {file_path} into {len(syllabus_chunks)} intelligent chunks")
            return syllabus_chunks

        except Exception as e:
            self.logger.error(f"Error processing syllabus file {file_path}: {e}")
            return []

    async def _process_single_json_file(self, file_path: str, doc_name: str) -> List:
        """Process a single JSON file and return documents"""
        try:
            from langchain_community.document_loaders import JSONLoader
            from langchain.schema import Document
            import json

            # Load the JSON file
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )

            documents = await asyncio.get_event_loop().run_in_executor(None, loader.load)

            # Convert JSON content to clean plain text
            cleaned_documents = []
            for doc in documents:
                try:
                    # Parse the JSON content
                    json_data = json.loads(doc.page_content)

                    # Convert JSON to clean plain text using the document processor method
                    plain_text = self.document_processor._convert_json_to_plain_text(json_data)

                    # Determine document type
                    doc_type = self.document_processor._determine_document_type(os.path.basename(file_path))

                    # Create new document with clean text
                    cleaned_doc = Document(
                        page_content=plain_text,
                        metadata={
                            "name": doc_name,
                            "type": doc_type,
                            "source": file_path
                        }
                    )
                    cleaned_documents.append(cleaned_doc)

                except Exception as parse_error:
                    self.logger.error(f"Error parsing JSON content: {parse_error}")
                    # Fallback to original content if parsing fails
                    doc.metadata.update({
                        "name": doc_name,
                        "type": self.document_processor._determine_document_type(os.path.basename(file_path))
                    })
                    cleaned_documents.append(doc)

            # Split documents
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.document_processor.text_splitter.split_documents(cleaned_documents)
            )



            return splits

        except Exception as e:
            self.logger.error(f"Error processing single JSON file {file_path}: {e}")
            return []

    async def _process_pdf_files_smart(self, existing_doc_names: List[str]) -> Dict:
        """Process PDF files with smart update detection"""
        results = {"processed": 0, "updated": 0, "errors": []}

        try:
            # Find PDF files in data directory
            pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]

            for pdf_file in pdf_files:
                try:
                    # Extract document name (filename without extension)
                    doc_name = pdf_file.replace('.pdf', '')

                    # Check if document exists in vector store
                    if doc_name in existing_doc_names:
                        # Update existing document
                        self.logger.info(f"🔄 Updating existing PDF document: {doc_name}")

                        # Process the single file
                        file_path = os.path.join(self.data_dir, pdf_file)
                        new_documents = await self._process_single_pdf_file(file_path, doc_name)

                        if new_documents:
                            # Update in vector store
                            success = await self.vector_store.update_documents_by_name(doc_name, new_documents)
                            if success:
                                results["updated"] += 1
                                # Delete processed file
                                os.remove(file_path)
                                self.logger.info(f"✅ Updated and deleted: {pdf_file}")
                            else:
                                results["errors"].append(f"Failed to update {doc_name} in vector store")
                    else:
                        # New document - will be processed normally
                        self.logger.info(f"📄 New PDF document detected: {doc_name}")
                        results["processed"] += 1

                except Exception as e:
                    error_msg = f"Error processing PDF file {pdf_file}: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)

            # Process any remaining new PDF files normally
            if results["processed"] > 0:
                await self.document_processor.load_and_process_pdf(delete_after_load=True)

        except Exception as e:
            results["errors"].append(f"Error in PDF processing: {str(e)}")

        return results

    async def _process_excel_files_smart(self, existing_doc_names: List[str]) -> Dict:
        """Process Excel files with smart update detection and routing sync"""
        results = {"processed": 0, "updated": 0, "routing_updated": 0, "errors": []}

        try:
            # Find Excel files in data directory
            excel_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.xlsx', '.xls'))]

            for excel_file in excel_files:
                try:
                    # For Excel files, use "FQA" as the document name
                    doc_name = "FQA"

                    # Check if FQA documents exist in vector store
                    if doc_name in existing_doc_names:
                        # Update existing FQA documents
                        self.logger.info(f"🔄 Updating existing Excel document: {doc_name}")

                        # Process the Excel file for both main vector store and routing
                        file_path = os.path.join(self.data_dir, excel_file)
                        new_documents, routing_questions = await self._process_single_excel_file_with_routing(file_path, excel_file)

                        if new_documents:
                            # Update main vector store
                            success = await self.vector_store.update_documents_by_name(doc_name, new_documents)
                            if success:
                                results["updated"] += 1
                                self.logger.info(f"✅ Updated main vector store for: {doc_name}")
                            else:
                                results["errors"].append(f"Failed to update {doc_name} in main vector store")

                        if routing_questions:
                            # Update routing vector store
                            routing_success = await self.document_processor.smart_router.routing_vector_store.update_questions_by_source(
                                excel_file, routing_questions
                            )
                            if routing_success:
                                results["routing_updated"] += 1
                                self.logger.info(f"✅ Updated routing questions for: {excel_file}")
                            else:
                                results["errors"].append(f"Failed to update routing questions for {excel_file}")

                        if new_documents or routing_questions:
                            # Delete processed file only if at least one update was successful
                            os.remove(file_path)
                            self.logger.info(f"✅ Updated and deleted: {excel_file}")
                    else:
                        # New document - will be processed normally
                        self.logger.info(f"📄 New Excel document detected: {doc_name}")
                        results["processed"] += 1

                except Exception as e:
                    error_msg = f"Error processing Excel file {excel_file}: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)

            # Process any remaining new Excel files normally (with routing sync)
            if results["processed"] > 0:
                await self.document_processor.load_and_process_excel(delete_after_load=True)
                self.logger.info("✅ Processed new Excel files with routing sync")

        except Exception as e:
            results["errors"].append(f"Error in Excel processing: {str(e)}")

        return results

    async def _process_single_json_file(self, file_path: str, doc_name: str) -> List:
        """Process a single JSON file and return documents"""
        try:
            from langchain_community.document_loaders import JSONLoader
            from langchain.schema import Document
            import json

            # Load the JSON file
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )

            documents = await asyncio.get_event_loop().run_in_executor(None, loader.load)

            # Convert JSON content to clean plain text
            cleaned_documents = []
            for doc in documents:
                try:
                    # Parse the JSON content
                    json_data = json.loads(doc.page_content)

                    # Convert JSON to clean plain text using the document processor method
                    plain_text = self.document_processor._convert_json_to_plain_text(json_data)

                    # Determine document type
                    doc_type = self.document_processor._determine_document_type(os.path.basename(file_path))

                    # Create new document with clean text
                    cleaned_doc = Document(
                        page_content=plain_text,
                        metadata={
                            "name": doc_name,
                            "type": doc_type,
                            "source": file_path
                        }
                    )
                    cleaned_documents.append(cleaned_doc)

                except Exception as parse_error:
                    self.logger.error(f"Error parsing JSON content: {parse_error}")
                    # Fallback to original content if parsing fails
                    doc.metadata.update({
                        "name": doc_name,
                        "type": self.document_processor._determine_document_type(os.path.basename(file_path))
                    })
                    cleaned_documents.append(doc)

            # Split documents
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.document_processor.text_splitter.split_documents(cleaned_documents)
            )


            return splits

        except Exception as e:
            self.logger.error(f"Error processing single JSON file {file_path}: {e}")
            return []

    async def _process_single_pdf_file(self, file_path: str, doc_name: str) -> List:
        """Process a single PDF file and return documents"""
        try:
            import pdfplumber
            import fitz
            from PIL import Image
            import io
            import pytesseract
            from langchain.schema import Document

            # Extract text from PDF (similar to document_processor logic)
            full_text = ""

            # Try pdfplumber first
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
            except Exception as e:
                self.logger.warning(f"pdfplumber failed for {file_path}: {e}")

            # Try OCR if no text found
            if not full_text.strip():
                try:
                    pdf_document = fitz.open(file_path)
                    for page_num in range(len(pdf_document)):
                        page = pdf_document.load_page(page_num)
                        mat = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        page_text = pytesseract.image_to_string(img, lang='vie+eng')
                        if page_text.strip():
                            full_text += page_text + "\n"
                    pdf_document.close()
                except Exception as e:
                    self.logger.warning(f"OCR failed for {file_path}: {e}")

            if not full_text.strip():
                return []

            # Create document
            doc = Document(
                page_content=full_text.strip(),
                metadata={
                    "name": doc_name,
                    "type": "Decision"
                }
            )

            # Split document
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.document_processor.text_splitter.split_documents([doc])
            )

            # Enhance with filename prefix
            enhanced_splits = []
            for split in splits:
                enhanced_content = f"{doc_name} : {split.page_content}"
                enhanced_split = Document(
                    page_content=enhanced_content,
                    metadata=split.metadata
                )
                enhanced_splits.append(enhanced_split)

            return enhanced_splits

        except Exception as e:
            self.logger.error(f"Error processing single PDF file {file_path}: {e}")
            return []

    async def _process_single_excel_file(self, file_path: str) -> List:
        """Process a single Excel file and return documents"""
        try:
            import pandas as pd
            from langchain.schema import Document

            # Load Excel file
            df = await asyncio.get_event_loop().run_in_executor(None, lambda: pd.read_excel(file_path))

            if 'question' not in df.columns or 'answer' not in df.columns:
                return []

            # Convert to documents
            documents = []
            for _, row in df.iterrows():
                questions = str(row['question']).split('|')
                answer = str(row['answer'])

                for q in questions:
                    q = q.strip()
                    if q:
                        content = f"Question: {q}\nAnswer: {answer}"
                        documents.append(Document(
                            page_content=content,
                            metadata={
                                "name": "FQA",
                                "type": "FQA"
                            }
                        ))

            # Split documents
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.document_processor.text_splitter.split_documents(documents)
            )

            return splits

        except Exception as e:
            self.logger.error(f"Error processing single Excel file {file_path}: {e}")
            return []

    async def _process_single_excel_file_with_routing(self, file_path: str, excel_file: str) -> tuple:
        """
        Process a single Excel file and return documents for both main vector store and routing

        Args:
            file_path (str): Path to the Excel file
            excel_file (str): Name of the Excel file (for routing source)

        Returns:
            tuple: (main_documents, routing_questions)
        """
        try:
            import pandas as pd
            from langchain.schema import Document

            # Load Excel file
            df = await asyncio.get_event_loop().run_in_executor(None, lambda: pd.read_excel(file_path))

            if 'question' not in df.columns or 'answer' not in df.columns:
                return [], []

            # Convert to documents for main vector store
            main_documents = []
            routing_questions = []

            for _, row in df.iterrows():
                questions = str(row['question']).split('|')
                answer = str(row['answer'])

                for q in questions:
                    q = q.strip()
                    if q:
                        # Create document for main vector store (RAG format)
                        content = f"Question: {q}\nAnswer: {answer}"
                        main_doc = Document(
                            page_content=content,
                            metadata={
                                "name": "FQA",
                                "type": "FQA",
                                "source": excel_file
                            }
                        )
                        main_documents.append(main_doc)

                        # Create routing question
                        routing_doc = Document(
                            page_content=q,  # Only question for embedding
                            metadata={
                                "answer": answer,  # Answer stored in metadata
                                "category": "FQA",
                                "source": excel_file
                            }
                        )
                        routing_questions.append(routing_doc)

            # Split main documents
            splits = []
            if main_documents:
                splits = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.document_processor.text_splitter.split_documents(main_documents)
                )

            self.logger.info(f"Processed {len(splits)} main documents and {len(routing_questions)} routing questions from {excel_file}")
            return splits, routing_questions

        except Exception as e:
            self.logger.error(f"Error processing single Excel file {file_path}: {e}")
            return [], []

    def _update_processing_log(self, sync_results: Dict, processing_results: Dict):
        """Update processing log with latest results"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sync_results": sync_results,
            "processing_results": processing_results
        }

        # Keep only last 50 entries
        if "history" not in self.processing_log:
            self.processing_log["history"] = []

        self.processing_log["history"].append(log_entry)
        self.processing_log["history"] = self.processing_log["history"][-50:]
        self.processing_log["last_run"] = log_entry

        self._save_processing_log()

    def start_scheduler(self):
        """Start the scheduler in a separate thread"""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return

        self.logger.info(f"🕐 Starting scheduler - will run every {self.schedule_interval_hours} hour(s)")

        # Schedule the regular job - temporarily changed to 1 minute for testing
        # schedule.every(self.schedule_interval_hours).hours.do(self._run_sync_job)
        schedule.every(1).minutes.do(self._run_sync_job)  # Test: run every 1 minute

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        # Run initial sync immediately in a separate thread
        initial_sync_thread = threading.Thread(target=self._run_initial_sync_immediately, daemon=True)
        initial_sync_thread.start()

        self.logger.info("✅ Scheduler started successfully - Running every 1 minute for testing")

    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        schedule.clear()
        self.logger.info("🛑 Scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _run_sync_job(self):
        """Run sync job - wrapper for asyncio"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.sync_and_process())
        finally:
            loop.close()

    def _run_initial_sync_immediately(self):
        """Run initial sync immediately on startup"""
        self.logger.info("🚀 Running initial sync immediately on startup...")
        self._run_sync_job()

    def _run_initial_sync(self):
        """Run initial sync on startup - DEPRECATED"""
        self.logger.info("🚀 Running initial sync on startup...")
        self._run_sync_job()
        # Remove the initial sync job after running once
        schedule.clear('initial')

    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            "is_running": self.is_running,
            "schedule_interval_hours": self.schedule_interval_hours,
            "data_directory": self.data_dir,
            "google_drive_folder_id": self.google_drive_folder_id,
            "last_run": self.processing_log.get("last_run"),
            "total_runs": len(self.processing_log.get("history", []))
        }

    async def manual_sync(self) -> Dict:
        """Manually trigger sync and processing"""
        self.logger.info("🔄 Manual sync triggered")
        return await self.sync_and_process()
