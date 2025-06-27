from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.vector_store import VectorStore
import os
import asyncio
import pandas as pd
from langchain.schema import Document

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

    async def load_and_process_documents(self):
        """
        Load documents from the data directory, split them into chunks,
        and store them in the vector database
        """
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory at {self.data_dir}")
            return []

        # List files in directory to verify
        print(f"Files in directory: {os.listdir(self.data_dir)}")

        # Load documents from directory using appropriate encoding
        try:
            # Create a custom TextLoader class with UTF-8 encoding
            class UTF8TextLoader(TextLoader):
                def __init__(self, file_path):
                    super().__init__(file_path, encoding="utf-8", autodetect_encoding=True)

            # Use the custom loader class
            loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.txt",
                loader_cls=UTF8TextLoader,
                show_progress=True
            )

            print(f"Loading files from {self.data_dir}...")
            # Run in a thread pool to avoid blocking the event loop
            documents = await asyncio.get_event_loop().run_in_executor(None, loader.load)
            print(f"Loaded {len(documents)} documents")

            if not documents:
                print("No documents found in the data directory")
                return []

            # Split documents into chunks (run in executor to avoid blocking)
            splits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.text_splitter.split_documents(documents)
            )

            # Add documents to vector store - sửa lỗi ở đây
            await self.vector_store.add_documents(splits)

            print(f"Processed {len(documents)} documents, created {len(splits)} chunks")
            return splits

        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            # Try to load individual files with different encodings as a fallback
            try:
                print("Attempting to load files individually...")
                documents = []
                for filename in os.listdir(self.data_dir):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(self.data_dir, filename)
                        try:
                            # Try different encodings
                            for encoding in ['utf-8-sig', 'utf-8', 'utf-16', 'cp1258']:
                                try:
                                    print(f"Trying to load {filename} with {encoding} encoding")
                                    text = await asyncio.get_event_loop().run_in_executor(
                                        None,
                                        lambda: open(file_path, 'r', encoding=encoding).read()
                                    )

                                    documents.append(Document(
                                        page_content=text,
                                        metadata={"source": file_path}
                                    ))
                                    print(f"Successfully loaded {filename} with {encoding} encoding")
                                    break
                                except UnicodeDecodeError:
                                    continue
                        except Exception as file_error:
                            print(f"Error loading file {filename}: {str(file_error)}")

                if documents:
                    # Split documents into chunks
                    splits = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.text_splitter.split_documents(documents)
                    )

                    # Add documents to vector store - sửa lỗi ở đây
                    await self.vector_store.add_documents(splits)

                    print(f"Processed {len(documents)} documents, created {len(splits)} chunks")
                    return splits

            except Exception as fallback_error:
                print(f"Fallback loading failed: {str(fallback_error)}")

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
                possible_paths = [  # app/data_test.xlsx
                    os.path.join(self.data_dir, 'data_test.xlsx')  # app/data/data_test.xlsx  # RAGChatbotai/data_test.xlsx
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

                row_docs = []
                for q in questions:
                    q = q.strip()
                    if q:
                        # Create a document with both question and answer
                        content = f"Question: {q}\nAnswer: {answer}"
                        row_docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": f"{excel_path}:row{idx+2}"
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

    async def load_and_process_all(self):
        """
        Load and process both text documents and Excel data
        """
        # Load text documents
        text_docs = await self.load_and_process_documents()

        # Load Excel data
        excel_docs = await self.load_and_process_excel()

        return text_docs + excel_docs

# Add this if you want to test the document processor directly
if __name__ == "__main__":
    processor = DocumentProcessor()
    # To process only text files:
    # asyncio.run(processor.load_and_process_documents())

    # To process only Excel:
    # asyncio.run(processor.load_and_process_excel())

    # To process both:
    asyncio.run(processor.load_and_process_all())
