from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.vector_store import VectorStore
from app.syllabus_converter import SyllabusConverter
import os
import asyncio
import pandas as pd
import re
import json
from langchain.schema import Document
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import io
import pytesseract

class DocumentProcessor:
    def __init__(self, data_dir="data"):
        # Convert relative path to absolute path if needed
        if not os.path.isabs(data_dir):
            # Get the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(current_dir, data_dir)
        else:
            self.data_dir = data_dir

        print(f"Data directory path: {self.data_dir}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = VectorStore()
        
        # Initialize Syllabus Converter for course data
        self.syllabus_converter = SyllabusConverter()

    def _determine_document_type(self, filename: str) -> str:
        """
        Xác định loại tài liệu dựa trên tên file:
        - Syllabus: nếu có pattern [3 chữ in hoa có thể có Đ][3 số][có thể có 1 chữ thường] hoặc chứa "LUK"
        - Curriculum: các file JSON khác
        """
        # Bỏ extension .json
        name_without_ext = filename.replace('.json', '')

        # Pattern cho Syllabus: 3 chữ in hoa (có thể có Đ) + 3 số + có thể c 1 chữ thường
        syllabus_pattern = r'^[A-ZĐ]{3}\d{3}[a-z]?$'

        # Kiểm tra pattern cho Syllabus
        if re.match(syllabus_pattern, name_without_ext):
            return "Syllabus"

        # Kiểm tra chứa "LUK"
        if "LUK" in name_without_ext.upper():
            return "Syllabus"

        # Mặc định là Curriculum
        return "Curriculum"

    async def load_and_process_documents(self, delete_after_load=False):
        """
        Load JSON documents from the data directory, split them into chunks,
        and store them in the vector database with proper metadata

        Args:
            delete_after_load (bool): If True, delete files after successfully loading them
        """
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory at {self.data_dir}")
            return []

        # List files in directory to verify
        print(f"Files in directory: {os.listdir(self.data_dir)}")

        documents = []

        try:
            # Tìm tất cả file JSON trong thư mục, loại trừ file system
            json_files = []
            for f in os.listdir(self.data_dir):
                if f.endswith('.json'):
                    # Loại trừ các file system
                    if f not in ['processing_log.json', 'sync_metadata.json']:
                        json_files.append(f)

            if not json_files:
                print("No JSON files found in the data directory")
                return []

            print(f"Found {len(json_files)} JSON files")

            for json_file in json_files:
                file_path = os.path.join(self.data_dir, json_file)

                try:
                    # Xác định metadata
                    name = json_file.replace('.json', '')  # Tên file bỏ .json
                    doc_type = self._determine_document_type(json_file)

                    print(f"Processing {json_file} - Name: {name}, Type: {doc_type}")
                    
                    # Check if this is a syllabus file that should use the specialized converter
                    if doc_type == "Syllabus":
                        # Use syllabus converter for intelligent chunking
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)

                            # Convert to intelligent chunks
                            syllabus_chunks = self.syllabus_converter.convert_syllabus_to_intelligent_chunks(json_data)

                            if syllabus_chunks:
                                # Add source file path to metadata of each chunk
                                for chunk in syllabus_chunks:
                                    chunk.metadata.update({
                                        "name": name,
                                        "type": doc_type,
                                        "source": file_path
                                    })

                                documents.extend(syllabus_chunks)
                                print(f"Successfully converted syllabus {json_file} to {len(syllabus_chunks)} intelligent chunks")
                                continue
                            else:
                                print(f"No chunks created for syllabus {json_file}, falling back to standard method")
                        except Exception as syllabus_error:
                            print(f"Error in intelligent chunking for {json_file}: {syllabus_error}")
                            print("Falling back to standard method")

                    # For non-syllabus files or if syllabus conversion failed, use standard method
                    # Sử dụng JSONLoader để load file
                    loader = JSONLoader(
                        file_path=file_path,
                        jq_schema='.',  # Load toàn bộ JSON content
                        text_content=False  # Change to False to get structured data
                    )

                    # Load documents từ JSON file
                    file_documents = await asyncio.get_event_loop().run_in_executor(
                        None, loader.load
                    )

                    # Convert JSON content to clean plain text
                    cleaned_documents = []
                    for doc in file_documents:
                        # Get the raw JSON data
                        try:
                            import json
                            json_data = json.loads(doc.page_content)

                            # Convert JSON to clean plain text
                            plain_text = self._convert_json_to_plain_text(json_data)

                            # Create new document with clean text
                            cleaned_doc = Document(
                                page_content=plain_text,
                                metadata={
                                    "name": name,
                                    "type": doc_type,
                                    "source": file_path
                                }
                            )
                            cleaned_documents.append(cleaned_doc)

                        except Exception as parse_error:
                            print(f"Error parsing JSON content for {json_file}: {parse_error}")
                            # Fallback to original content if parsing fails
                            doc.metadata.update({
                                "name": name,
                                "type": doc_type
                            })
                            cleaned_documents.append(doc)

                    documents.extend(cleaned_documents)

                    print(f"Successfully loaded {json_file} with {len(file_documents)} documents")

                except Exception as file_error:
                    print(f"Error loading file {json_file}: {str(file_error)}")
                    continue

            if not documents:
                print("No documents were successfully loaded")
                return []

            print(f"Total loaded {len(documents)} documents from {len(json_files)} JSON files")

            # Split documents into chunks
            # Don't split syllabus documents as they are already intelligently chunked
            syllabus_docs = [doc for doc in documents if doc.metadata.get('type') == 'Syllabus']
            non_syllabus_docs = [doc for doc in documents if doc.metadata.get('type') != 'Syllabus']

            # Only split non-syllabus documents
            if non_syllabus_docs:
                splits = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.text_splitter.split_documents(non_syllabus_docs)
                )
                print(f"Split {len(non_syllabus_docs)} non-syllabus documents into {len(splits)} chunks")
            else:
                splits = []

            # Combine syllabus documents (already chunked) with split non-syllabus documents
            all_processed_docs = syllabus_docs + splits

            # Add documents to vector store
            await self.vector_store.add_documents(all_processed_docs)

            print(f"Processed {len(documents)} documents, created {len(all_processed_docs)} chunks with file names prefixed")

            # Delete files after successful processing if requested
            if delete_after_load:
                await self._delete_processed_files(json_files, "JSON")

            return all_processed_docs

        except Exception as e:
            print(f"Error processing JSON documents: {str(e)}")
            return []

    async def load_and_process_excel(self, delete_after_load=False):
        """
        Load data from all Excel files containing FQA,
        convert to documents, and add to both vector stores

        Args:
            delete_after_load (bool): If True, delete files after successfully loading them
        """
        try:
            # Check if directory exists
            if not os.path.exists(self.data_dir):
                print(f"Data directory does not exist: {self.data_dir}")
                return []

            # Find all Excel files in the data directory
            excel_files = []
            for f in os.listdir(self.data_dir):
                if f.endswith('.xlsx') or f.endswith('.xls'):
                    excel_files.append(f)

            if not excel_files:
                print("No Excel files found in the data directory")
                return []

            print(f"Found {len(excel_files)} Excel files")

            all_documents = []
            processed_files = []

            for excel_file in excel_files:
                excel_path = os.path.join(self.data_dir, excel_file)

                try:
                    print(f"Loading Excel file: {excel_file}")

                    # Load Excel file asynchronously
                    df = await asyncio.get_event_loop().run_in_executor(None, lambda: pd.read_excel(excel_path))
                    print(f"Excel file loaded with {len(df)} rows")

                    # Check if this file contains FQA data (must have 'question' and 'answer' columns)
                    if 'question' not in df.columns or 'answer' not in df.columns:
                        print(f"Skipping {excel_file} - does not contain 'question' and 'answer' columns")
                        continue

                    print(f"Processing FQA data from {excel_file}")

                    # Convert Excel data to documents
                    documents = []

                    # Process each row asynchronously
                    async def process_row(idx, row):
                        questions = str(row['question']).split('|')
                        answer = str(row['answer'])

                        row_docs = []
                        for q in questions:
                            q = q.strip()
                            if q:
                                # Create a document with both question and answer
                                content = f"Question: {q}\nAnswer: {answer}"
                                row_docs.append(Document(
                                    page_content=content,
                                    metadata={
                                        "name": "FQA",
                                        "type": "FQA",
                                        "source": excel_file
                                    }
                                ))
                        return row_docs

                    # Create tasks for all rows
                    tasks = [process_row(idx, row) for idx, row in df.iterrows()]

                    # Execute all tasks and gather results
                    results = await asyncio.gather(*tasks)

                    # Flatten list of lists
                    for row_docs in results:
                        documents.extend(row_docs)

                    print(f"Created {len(documents)} documents from {excel_file}")

                    if documents:
                        all_documents.extend(documents)
                        processed_files.append(excel_file)

                except Exception as file_error:
                    print(f"Error processing {excel_file}: {str(file_error)}")
                    continue

            if not all_documents:
                print("No valid FQA data found in any Excel files")
                return []

            print(f"Total created {len(all_documents)} documents from {len(processed_files)} Excel files")

            # Split documents into chunks (run in executor to avoid blocking)
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.text_splitter.split_documents(all_documents)
            )

            # Add documents to both vector stores
            # 1. Add to main vector store (RAG format with full content)
            await self.vector_store.add_documents(splits)
            print(f"Added {len(splits)} chunks to main vector store")

            # 2. Add to routing vector store (only questions for embedding)
            routing_documents = []
            for doc in all_documents:
                # Extract question from the RAG format content
                content = doc.page_content
                if content.startswith("Question: "):
                    # Extract just the question part
                    question_part = content.split("\nAnswer: ")[0].replace("Question: ", "")
                    # Extract answer part
                    answer_part = content.split("\nAnswer: ")[1] if "\nAnswer: " in content else ""

                    # Create routing document with only question as content
                    routing_doc = Document(
                        page_content=question_part,  # Only question for embedding
                        metadata={
                            "answer": answer_part,  # Answer stored in metadata
                            "category": "FQA",
                            "source": doc.metadata.get("source", "excel")
                        }
                    )
                    routing_documents.append(routing_doc)

            if routing_documents:
                await self.smart_router.routing_vector_store.add_questions(routing_documents)
                print(f"Added {len(routing_documents)} questions to routing vector store")
            else:
                print("No routing documents created from FQA data")

            print(f"Processed {len(processed_files)} Excel files with FQA data, created {len(splits)} chunks")

            # Delete Excel files after successful processing if requested
            if delete_after_load and processed_files:
                await self._delete_processed_files(processed_files, "Excel")

                # Also delete corresponding routing questions
                for excel_file in processed_files:
                    await self.smart_router.routing_vector_store.delete_questions_by_source(excel_file)
                    print(f"Deleted routing questions from source: {excel_file}")

            return splits

        except Exception as e:
            print(f"Error processing Excel files: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    async def load_and_process_pdf(self, delete_after_load=False):
        """
        Load PDF files from the data directory, convert to documents with metadata
        name as filename and type as "Decision", and add to vector store

        Args:
            delete_after_load (bool): If True, delete files after successfully loading them
        """
        try:
            # Check if directory exists
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                print(f"Created data directory at {self.data_dir}")
                return []

            # Find all PDF files in the directory
            pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]

            if not pdf_files:
                print("No PDF files found in the data directory")
                return []

            print(f"Found {len(pdf_files)} PDF files")

            documents = []

            for pdf_file in pdf_files:
                file_path = os.path.join(self.data_dir, pdf_file)

                try:
                    # Get filename without extension for metadata
                    name = pdf_file.replace('.pdf', '')

                    print(f"Processing {pdf_file} - Name: {name}, Type: Decision")

                    # First try with pdfplumber
                    full_text = ""
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                text = page.extract_text()
                                if text:
                                    full_text += text + "\n"
                    except Exception as e:
                        print(f"pdfplumber failed: {e}")

                    print(f"pdfplumber extracted: {len(full_text)} characters")

                    # If no text found, try OCR with PyMuPDF
                    if not full_text.strip():
                        print("Attempting OCR extraction...")
                        try:
                            # Use PyMuPDF for OCR
                            pdf_document = fitz.open(file_path)
                            ocr_text = ""

                            for page_num in range(len(pdf_document)):
                                page = pdf_document.load_page(page_num)

                                # Convert page to image
                                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                                pix = page.get_pixmap(matrix=mat)
                                img_data = pix.tobytes("png")

                                # Convert to PIL Image
                                img = Image.open(io.BytesIO(img_data))

                                # Use OCR to extract text
                                try:
                                    page_text = pytesseract.image_to_string(img, lang='vie+eng')
                                    if page_text.strip():
                                        ocr_text += page_text + "\n"
                                        print(f"OCR Page {page_num + 1}: {len(page_text)} characters")
                                except Exception as ocr_error:
                                    print(f"OCR failed for page {page_num + 1}: {ocr_error}")
                                    continue

                            pdf_document.close()

                            if ocr_text.strip():
                                full_text = ocr_text
                                print(f"OCR total extracted: {len(full_text)} characters")
                            else:
                                print("OCR extraction failed - no text found")

                        except Exception as ocr_error:
                            print(f"OCR process failed: {ocr_error}")

                    if full_text.strip():
                        # Create a single document for the entire PDF
                        doc = Document(
                            page_content=full_text.strip(),
                            metadata={
                                "name": name,
                                "type": "Decision"
                            }
                        )
                        documents.append(doc)
                        print(f"Document created with {len(full_text)} characters")
                    else:
                        print(f"No text content found in {pdf_file} even after OCR")

                    print(f"Successfully processed {pdf_file}")

                except Exception as file_error:
                    print(f"Error loading file {pdf_file}: {str(file_error)}")
                    continue

            if not documents:
                print("No documents were successfully loaded")
                return []

            print(f"Total loaded {len(documents)} documents from {len(pdf_files)} PDF files")

            # Split documents into chunks
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.text_splitter.split_documents(documents)
            )

            # Add filename prefix to each chunk for better search
            enhanced_splits = []
            for split in splits:
                # Get filename from metadata
                file_name = split.metadata.get("name", "unknown")

                # Add filename to the beginning of content
                enhanced_content = f"{file_name} : {split.page_content}"

                # Create new document with enhanced content
                enhanced_split = Document(
                    page_content=enhanced_content,
                    metadata=split.metadata
                )
                enhanced_splits.append(enhanced_split)

            # Add documents to vector store
            await self.vector_store.add_documents(enhanced_splits)

            print(f"Processed {len(documents)} PDF documents, created {len(enhanced_splits)} chunks with file names prefixed")

            # Delete PDF files after successful processing if requested
            if delete_after_load:
                await self._delete_processed_files(pdf_files, "PDF")

            return enhanced_splits

        except Exception as e:
            print(f"Error processing PDF documents: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    async def _delete_processed_files(self, file_list, file_type):
        """
        Delete a list of files after successful processing

        Args:
            file_list (list): List of filenames to delete
            file_type (str): Type of files being deleted (for logging)
        """
        deleted_count = 0
        failed_count = 0

        for filename in file_list:
            try:
                file_path = os.path.join(self.data_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✅ Deleted {file_type} file: {filename}")
                    deleted_count += 1
                else:
                    print(f"⚠️ File not found for deletion: {filename}")
                    failed_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {filename}: {str(e)}")
                failed_count += 1

        print(f"📊 File deletion summary for {file_type}: {deleted_count} deleted, {failed_count} failed")

    async def _delete_single_file(self, file_path, file_type):
        """
        Delete a single file after successful processing

        Args:
            file_path (str): Full path to the file to delete
            file_type (str): Type of file being deleted (for logging)
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                filename = os.path.basename(file_path)
                print(f"✅ Deleted {file_type} file: {filename}")
                return True
            else:
                print(f"⚠️ File not found for deletion: {file_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to delete {file_path}: {str(e)}")
            return False

    async def load_routing_questions_from_excel(self, excel_path=None):
        """
        Load questions from data.xlsx file and add them to the routing vector store
        """
        try:
            # If path not provided, use the data_router.xlsx file in app/data directory
            if excel_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                excel_path = os.path.join(current_dir, 'data', 'data_router.xlsx')

            if not os.path.exists(excel_path):
                print(f"File not found: {excel_path}")
                return False

            print(f"Loading routing questions from Excel file: {excel_path}")

            # Load Excel file asynchronously
            df = await asyncio.get_event_loop().run_in_executor(None, lambda: pd.read_excel(excel_path))
            print(f"Excel file loaded with {len(df)} rows")

            # Check if required columns exist
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Excel file must contain columns: {required_columns}")
                print(f"Missing columns: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                return False

            # Convert Excel data to routing questions format
            routing_questions = []

            for idx, row in df.iterrows():
                try:
                    # Get question and answer
                    question = str(row['question']).strip()
                    answer = str(row['answer']).strip()

                    # Skip empty rows
                    if not question or question.lower() in ['nan', 'none', '']:
                        continue

                    # Handle multiple questions separated by '|'
                    questions = [q.strip() for q in question.split('|') if q.strip()]

                    for q in questions:
                        # Determine category if available
                        category = "general"
                        if 'category' in df.columns:
                            category = str(row['category']).strip()
                            if category.lower() in ['nan', 'none', '']:
                                category = "general"

                        routing_questions.append({
                            "question": q,
                            "answer": answer,  # Thêm answer vào data
                            "category": category
                        })

                except Exception as row_error:
                    print(f"Error processing row {idx}: {row_error}")
                    continue

            if not routing_questions:
                print("No valid routing questions found in Excel file")
                return False

            print(f"Prepared {len(routing_questions)} routing questions")

            # Add questions to routing vector store
            success = await self.smart_router.add_routing_questions(routing_questions)

            if success:
                print(f"✅ Successfully added {len(routing_questions)} routing questions to vector store")

                # Get stats after adding
                stats = await self.smart_router.get_routing_stats()
                print(f"📊 Routing Vector Store Stats: {stats}")

                return True
            else:
                print("❌ Failed to add routing questions to vector store")
                return False

        except Exception as e:
            print(f"Error loading routing questions from Excel: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def load_and_process_all_with_routing(self, delete_after_load=False):
        """
        Load and process all documents including routing questions

        Args:
            delete_after_load (bool): If True, delete files after successfully loading them
        """
        print("=== Loading Routing Questions ===")
        routing_success = await self.load_routing_questions_from_excel()

        print("\n=== Loading Regular Documents ===")
        # Load text documents
        text_docs = await self.load_and_process_documents(delete_after_load=delete_after_load)

        # Load Excel data for RAG
        excel_docs = await self.load_and_process_excel(delete_after_load=delete_after_load)

        # Load PDF data
        pdf_docs = await self.load_and_process_pdf(delete_after_load=delete_after_load)

        total_docs = text_docs + excel_docs + pdf_docs

        print(f"\n=== Summary ===")
        print(f"Routing questions loaded: {'✅' if routing_success else '❌'}")
        print(f"Total RAG documents processed: {len(total_docs)}")

        if delete_after_load:
            print("🗑️ File deletion mode was enabled - processed files have been removed")

        return total_docs

    async def safe_load_and_delete(self, file_types=None, confirm_deletion=True):
        """
        Safely load documents and optionally delete them with confirmation

        Args:
            file_types (list): List of file types to process ['json', 'excel', 'pdf']
            confirm_deletion (bool): Ask for confirmation before deleting files
        """
        if file_types is None:
            file_types = ['json', 'excel', 'pdf']

        print("🔄 Starting safe document loading...")

        total_docs = []

        try:
            # Load routing questions first (no deletion needed)
            print("\n=== Loading Routing Questions ===")
            routing_success = await self.load_routing_questions_from_excel()

            # Process each file type
            if 'json' in file_types:
                print("\n=== Processing JSON Documents ===")
                json_docs = await self.load_and_process_documents(delete_after_load=True)
                total_docs.extend(json_docs)

            if 'excel' in file_types:
                print("\n=== Processing Excel Documents ===")
                excel_docs = await self.load_and_process_excel(delete_after_load=True)
                total_docs.extend(excel_docs)

            if 'pdf' in file_types:
                print("\n=== Processing PDF Documents ===")
                pdf_docs = await self.load_and_process_pdf(delete_after_load=True)
                total_docs.extend(pdf_docs)

            print(f"\n✅ Successfully processed {len(total_docs)} documents")
            print("🗑️ Source files have been deleted after successful processing")

            return total_docs

        except Exception as e:
            print(f"❌ Error during safe load and delete: {str(e)}")
            return []

    def _convert_json_to_plain_text(self, json_data):
        """
        Convert JSON data to clean plain text format

        Args:
            json_data: Parsed JSON data (dict, list, or other)

        Returns:
            str: Clean plain text representation
        """
        def clean_text(text):
            """Clean text by removing extra whitespace and formatting"""
            if not isinstance(text, str):
                text = str(text)

            # Remove newline characters within values and normalize whitespace
            text = re.sub(r'\s+', ' ', text.strip())

            # Remove common escape characters
            text = text.replace('\\n', ' ')
            text = text.replace('\\t', ' ')
            text = text.replace('\\r', ' ')

            return text.strip()

        def process_value(value, key=None):
            """Process a single value from JSON"""
            if value is None or value == "":
                return ""

            if isinstance(value, (dict, list)):
                return json_to_text(value)
            else:
                cleaned = clean_text(value)
                if key and cleaned:
                    return f"{key}: {cleaned}"
                return cleaned

        def json_to_text(data):
            """Convert JSON structure to plain text"""
            if isinstance(data, dict):
                text_parts = []
                for key, value in data.items():
                    if value is not None and value != "":
                        processed = process_value(value, key)
                        if processed:
                            text_parts.append(processed)
                return ". ".join(text_parts)

            elif isinstance(data, list):
                text_parts = []
                for item in data:
                    if isinstance(item, dict):
                        # For list of objects, process each object
                        item_text = json_to_text(item)
                        if item_text:
                            text_parts.append(item_text)
                    else:
                        # For list of simple values
                        processed = process_value(item)
                        if processed:
                            text_parts.append(processed)
                return ". ".join(text_parts)

            else:
                return clean_text(data)

        try:
            result = json_to_text(json_data)
            # Final cleanup - remove multiple spaces and normalize punctuation
            result = re.sub(r'\s+', ' ', result)
            result = re.sub(r'\s*\.\s*', '. ', result)
            result = re.sub(r'\s*:\s*', ': ', result)

            return result.strip()

        except Exception as e:
            print(f"Error converting JSON to plain text: {e}")
            return str(json_data)  # Fallback to string representation

async def main():
    """
    Main function to manually process files in the data directory
    Excludes log files and system files
    """
    print("=== Document Processor - Manual File Processing ===")

    # Initialize document processor
    processor = DocumentProcessor()

    try:
        # Check if data directory exists and list files
        if not os.path.exists(processor.data_dir):
            print(f"Data directory does not exist: {processor.data_dir}")
            return

        files = os.listdir(processor.data_dir)
        print(f"Files found in {processor.data_dir}:")

        # Filter out log files and system files
        excluded_files = ['processing_log.json', 'sync_metadata.json', '__init__.py']
        valid_files = []

        for file in files:
            if file not in excluded_files and not file.startswith('.'):
                valid_files.append(file)
                file_type = "Unknown"
                if file.endswith('.json'):
                    file_type = "JSON"
                elif file.endswith(('.xlsx', '.xls')):
                    file_type = "Excel"
                elif file.endswith('.pdf'):
                    file_type = "PDF"
                print(f"  - {file} ({file_type})")

        if not valid_files:
            print("No valid files found to process")
            return

        print(f"\nFound {len(valid_files)} valid files to process")

        # Ask user for confirmation
        response = input("\nDo you want to process all files and delete them after successful loading? (y/n): ")

        if response.lower() in ['y', 'yes']:
            delete_after_load = True
            print("Files will be deleted after successful processing")
        else:
            delete_after_load = False
            print("Files will be kept after processing")

        # Process all files
        print("\n" + "="*50)
        print("Starting file processing...")

        # Load and process all documents with routing
        total_docs = await processor.load_and_process_all_with_routing(delete_after_load=delete_after_load)

        print("\n" + "="*50)
        print("Processing completed!")
        print(f"Total documents processed: {len(total_docs)}")

        if delete_after_load:
            print("✅ Source files have been deleted after successful processing")
        else:
            print("📁 Source files have been preserved")

    except KeyboardInterrupt:
        print("\n❌ Processing cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
