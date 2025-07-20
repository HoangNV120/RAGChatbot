from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.vector_store import VectorStore
from app.smart_router import SmartQueryRouter
import os
import asyncio
import pandas as pd
import re
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

        # Initialize Smart Router for routing questions
        self.smart_router = SmartQueryRouter()

    def _determine_document_type(self, filename: str) -> str:
        """
        Xác định loại tài liệu dựa trên tên file:
        - Syllabus: nếu có pattern [3 chữ in hoa có thể có Đ][3 số][có thể có 1 chữ thường] hoặc chứa "LUK"
        - Curriculum: các file JSON khác
        """
        # Bỏ extension .json
        name_without_ext = filename.replace('.json', '')

        # Pattern cho Syllabus: 3 chữ in hoa (có thể có Đ) + 3 số + có thể c�� 1 chữ thường
        syllabus_pattern = r'^[A-ZĐ]{3}\d{3}[a-z]?$'

        # Kiểm tra pattern cho Syllabus
        if re.match(syllabus_pattern, name_without_ext):
            return "Syllabus"

        # Kiểm tra chứa "LUK"
        if "LUK" in name_without_ext.upper():
            return "Syllabus"

        # Mặc định là Curriculum
        return "Curriculum"

    async def load_and_process_documents(self):
        """
        Load JSON documents from the data directory, split them into chunks,
        and store them in the vector database with proper metadata
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
            # Tìm tất cả file JSON trong thư mục
            json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]

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

                    # Sử dụng JSONLoader để load file
                    loader = JSONLoader(
                        file_path=file_path,
                        jq_schema='.',  # Load toàn bộ JSON content
                        text_content=False
                    )

                    # Load documents từ JSON file
                    file_documents = await asyncio.get_event_loop().run_in_executor(
                        None, loader.load
                    )

                    # Thêm metadata cho mỗi document
                    for doc in file_documents:
                        doc.metadata.update({
                            "name": name,
                            "type": doc_type
                        })
                        documents.append(doc)

                    print(f"Successfully loaded {json_file} with {len(file_documents)} documents")

                except Exception as file_error:
                    print(f"Error loading file {json_file}: {str(file_error)}")
                    continue

            if not documents:
                print("No documents were successfully loaded")
                return []

            print(f"Total loaded {len(documents)} documents from {len(json_files)} JSON files")

            # Split documents into chunks
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.text_splitter.split_documents(documents)
            )

            # Thêm tên file vào đầu mỗi chunk để dễ tìm kiếm
            enhanced_splits = []
            for split in splits:
                # Lấy tên file từ metadata
                file_name = split.metadata.get("name", "unknown")

                # Thêm tên file vào đầu content
                enhanced_content = f"{file_name} : {split.page_content}"

                # Tạo document mới với content đã được enhance
                enhanced_split = Document(
                    page_content=enhanced_content,
                    metadata=split.metadata
                )
                enhanced_splits.append(enhanced_split)

            # Add documents to vector store
            await self.vector_store.add_documents(enhanced_splits)

            print(f"Processed {len(documents)} documents, created {len(enhanced_splits)} chunks with file names prefixed")
            return enhanced_splits

        except Exception as e:
            print(f"Error processing JSON documents: {str(e)}")
            return []

    async def load_and_process_excel(self, excel_path=None):
        """
        Load data from Excel file containing question and answer columns,
        convert to documents, and add to vector store
        """
        try:
            # If path not provided, try multiple locations for data_test.xlsx
            if excel_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                possible_paths = [
                    os.path.join(self.data_dir, 'data_test.xlsx'),  # app/data/data_test.xlsx
                ]

                # Try each path until we find the file
                for path in possible_paths:
                    if os.path.exists(path):
                        excel_path = path
                        break

                if excel_path is None:
                    print("Could not find data_test.xlsx in any of the expected locations:")
                    for path in possible_paths:
                        print(f"  - {path}")
                    return []

            print(f"Loading Excel file from: {excel_path}")

            # Load Excel file asynchronously
            df = await asyncio.get_event_loop().run_in_executor(None, lambda: pd.read_excel(excel_path))
            print(f"Excel file loaded with {len(df)} rows")

            if 'question' not in df.columns or 'answer' not in df.columns:
                print("Excel file must contain 'question' and 'answer' columns")
                return []

            # Convert Excel data to documents
            documents = []

            # Process each row asynchronously
            async def process_row(idx, row):
                questions = str(row['question']).split('|')
                answer = str(row['answer'])
                
                # Extract category if available
                category = "general"
                if 'category' in df.columns:
                    category = str(row['category']).strip()
                    if category.lower() in ['nan', 'none', '']:
                        category = "general"

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
                                "category": category,
                                "answer": answer
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

            print(f"Created {len(documents)} documents from Excel data")

            if not documents:
                print("No valid data found in Excel file")
                return []

            # Split documents into chunks (run in executor to avoid blocking)
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.text_splitter.split_documents(documents)
            )

            # Add documents to vector store - sửa lỗi ở đây
            await self.vector_store.add_documents(splits)

            print(f"Processed Excel file, created {len(splits)} chunks")
            return splits

        except Exception as e:
            print(f"Error processing Excel file: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    async def load_and_process_pdf(self):
        """
        Load PDF files from the data directory, convert to documents with metadata
        name as filename and type as "Decision", and add to vector store
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
            return enhanced_splits

        except Exception as e:
            print(f"Error processing PDF documents: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

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

    async def load_and_process_all_with_routing(self):
        """
        Load and process all documents including routing questions
        """
        print("=== Loading Routing Questions ===")
        routing_success = await self.load_routing_questions_from_excel()

        print("\n=== Loading Regular Documents ===")
        # Load text documents
        text_docs = await self.load_and_process_documents()

        # Load Excel data for RAG
        excel_docs = await self.load_and_process_excel()

        # Load PDF data
        pdf_docs = await self.load_and_process_pdf()

        total_docs = text_docs + excel_docs + pdf_docs

        print(f"\n=== Summary ===")
        print(f"Routing questions loaded: {'✅' if routing_success else '❌'}")
        print(f"Total RAG documents processed: {len(total_docs)}")

        return total_docs

# Add this if you want to test the document processor directly
if __name__ == "__main__":
    processor = DocumentProcessor()
    # To process only text files:
    # asyncio.run(processor.load_and_process_documents())

    # To process only Excel:
    # asyncio.run(processor.load_and_process_excel())

    # To process only PDF:
    # asyncio.run(processor.load_and_process_pdf())

    # To process all:
    asyncio.run(processor.load_and_process_all_with_routing())
