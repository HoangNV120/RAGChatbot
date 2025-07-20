#!/usr/bin/env python3
"""
Document Processor cho VectorStoreSmall - chỉ embed cột question

Tính năng chính:
- Load dữ liệu từ Excel vào ragsmall collection
- Chỉ embed cột question (không phải Q&A format)
- Tự động kiểm tra collection đã có dữ liệu hay chưa
- Skip loading nếu collection đã có dữ liệu (tránh duplicate)
- Hỗ trợ force_reload để reload dữ liệu khi cần

Usage:
    python load_ragsmall.py                    # Load data nếu collection trống
    python load_ragsmall.py --force           # Force reload data
    
    # Hoặc trong code:
    processor = DocumentProcessorSmall()
    await processor.load_excel_to_ragsmall()                    # Auto skip if has data
    await processor.load_excel_to_ragsmall(force_reload=True)   # Force reload
"""

import asyncio
import sys
import os
import pandas as pd
from langchain.schema import Document

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store_small import VectorStoreSmall

class DocumentProcessorSmall:
    """
    Document processor để load data vào ragsmall collection
    Chỉ embed cột question, giữ nguyên metadata
    """
    
    def __init__(self):
        self.vector_store = VectorStoreSmall()
        
    async def load_excel_to_ragsmall(self, excel_path=None, force_reload=False):
        """
        Load Excel data vào ragsmall collection - chỉ embed question
        
        Args:
            excel_path (str): Path to Excel file, if None will use default
            force_reload (bool): If True, will reload data even if collection has data
        """
        try:
            # Check if collection already has data and force_reload is False
            if not force_reload and self.vector_store.collection_exists_and_has_data():
                info = self.vector_store.get_collection_info()
                print(f"🔍 Collection '{self.vector_store.collection_name}' already exists with {info.get('points_count', 0)} documents")
                print(f"⏭️  Skipping data loading. Use force_reload=True to reload data.")
                print(f"📊 Current collection info: {info}")
                return True
            
            # Tìm file Excel
            if excel_path is None:
                # Use absolute path to data_test.xlsx
                excel_path = r"d:\Python\Master_Chatbot\RAGChatbot\app\data\data_test.xlsx"
                
                if not os.path.exists(excel_path):
                    print(f"❌ Excel file not found at: {excel_path}")
                    
                    # Try alternative paths as backup
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    possible_paths = [
                        os.path.join(current_dir, 'app/data/data_test.xlsx'),     # Alternative format
                        'app/data/data_test.xlsx',                                 # Relative path
                    ]
                    
                    for path in possible_paths:
                        print(f"🔍 Checking path: {path}")
                        if os.path.exists(path):
                            excel_path = path
                            print(f"✅ Found Excel file at: {path}")
                            break
                    else:
                        print("❌ Could not find data_test.xlsx")
                        print("📁 Checked paths:")
                        for path in possible_paths:
                            print(f"   - {path} (exists: {os.path.exists(path)})")
                        return False
            
            print(f"📁 Loading Excel from: {excel_path}")
            
            # Load Excel file
            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pd.read_excel(excel_path)
            )
            print(f"📊 Loaded {len(df)} rows from Excel")
            
            if 'question' not in df.columns or 'answer' not in df.columns:
                print("❌ Excel file must contain 'question' and 'answer' columns")
                return False
            
            # Process documents - CHỈ EMBED QUESTION
            documents = []
            
            for idx, row in df.iterrows():
                try:
                    questions = str(row['question']).split('|')
                    answer = str(row['answer'])
                    
                    # Extract metadata
                    category = "general"
                    if 'category' in df.columns:
                        category = str(row['category']).strip()
                        if category.lower() in ['nan', 'none', '']:
                            category = "general"
                    
                    source = "unknown"
                    if 'nguồn' in df.columns:
                        source = str(row['nguồn']).strip()
                        if source.lower() in ['nan', 'none', '']:
                            source = "unknown"
                    
                    # Tạo document cho mỗi question - CHỈ EMBED QUESTION
                    for q in questions:
                        q = q.strip()
                        if q:
                            # Page content CHỈ LÀ QUESTION (không có "Question:" prefix)
                            doc = Document(
                                page_content=q,  # CHỈ question thôi
                                metadata={
                                    "question": q,
                                    "answer": answer,
                                    "category": category,
                                    "source": source,
                                    "type": "FQA_SMALL"
                                }
                            )
                            documents.append(doc)
                            
                except Exception as e:
                    print(f"❌ Error processing row {idx}: {e}")
                    continue
            
            if not documents:
                print("❌ No valid documents created")
                return False
            
            print(f"📝 Created {len(documents)} documents (question-only format)")
            
            # Check if we need to clear existing data first
            if force_reload and not self.vector_store.is_collection_empty():
                print(f"🗑️  force_reload=True: Collection has existing data, but will add new documents")
                print(f"⚠️  Note: This will add to existing data, not replace it")
                print(f"💡 To completely replace data, delete collection manually first")
            
            # Add to vector store
            print(f"🔄 Adding documents to ragsmall collection...")
            await self.vector_store.add_documents(documents)
            
            print(f"✅ Successfully added {len(documents)} documents to ragsmall!")
            
            # Show sample
            print(f"\n📊 Sample documents:")
            for i, doc in enumerate(documents[:3]):
                print(f"  {i+1}. Content: '{doc.page_content[:50]}...'")
                print(f"     Category: '{doc.metadata.get('category')}'")
                print(f"     Answer: '{doc.metadata.get('answer')[:50]}...'")
            
            # Show collection info
            info = self.vector_store.get_collection_info()
            print(f"\n📊 Collection Info: {info}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data to ragsmall: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main function để test DocumentProcessorSmall"""
    
    print("🚀 Loading data to ragsmall collection...")
    print("=" * 60)
    
    processor = DocumentProcessorSmall()
    
    # Check current collection status first
    info = processor.vector_store.get_collection_info()
    if 'error' not in info:
        print(f"📊 Current collection status:")
        print(f"   - Name: {info['name']}")
        print(f"   - Points: {info['points_count']}")
        print(f"   - Vectors: {info['vectors_count']}")
        print(f"   - Status: {info['status']}")
        print()
    
    # Load data (will skip if data already exists)
    success = await processor.load_excel_to_ragsmall()
    
    # Optional: Force reload example (uncomment to test)
    # print("\n🔄 Testing force reload...")
    # success = await processor.load_excel_to_ragsmall(force_reload=True)
    
    if success:
        print(f"\n✅ Data processing completed!")
        
        # Show final collection info
        final_info = processor.vector_store.get_collection_info()
        print(f"📊 Final collection info: {final_info}")
        
        # Test search
        print(f"\n🧪 Testing search in ragsmall...")
        test_query = "Học phí ngành CNTT"
        
        results = await processor.vector_store.similarity_search_with_score(
            query=test_query,
            k=3
        )
        
        print(f"📊 Search results for '{test_query}':")
        for i, (doc, score) in enumerate(results):
            print(f"  {i+1}. Score: {score:.4f}")
            print(f"     Question: {doc.page_content}")
            print(f"     Category: {doc.metadata.get('category')}")
            print(f"     Answer: {doc.metadata.get('answer')[:100]}...")
            print()
    else:
        print(f"\n❌ Failed to process data for ragsmall!")

if __name__ == "__main__":
    asyncio.run(main())
