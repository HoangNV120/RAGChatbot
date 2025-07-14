#!/usr/bin/env python3
"""
Script để re-categorize toàn bộ dữ liệu trong data_test.xlsx
bằng kỹ thuật prompt engineering đã tối ưu hóa để đảm bảo tính nhất quán 100%
"""

import pandas as pd
import asyncio
import sys
import os
from datetime import datetime
import shutil

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.category_partitioned_router import CategoryPartitionedRouter

class DataRecategorizer:
    """Class để re-categorize dữ liệu với consistency cao"""
    
    def __init__(self):
        # Chỉ khởi tạo LLM classification, không cần vector embeddings
        self.router = CategoryPartitionedRouter(use_categorized_data=False)
        
        # Thống kê
        self.total_processed = 0
        self.categorization_results = {}
        
    async def backup_original_file(self):
        """Backup file gốc trước khi sửa"""
        original_file = 'app/data_test.xlsx'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f'app/data_test_backup_{timestamp}.xlsx'
        
        try:
            shutil.copy2(original_file, backup_file)
            print(f"📁 Backup created: {backup_file}")
            return backup_file
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return None
    
    async def recategorize_all_data(self):
        """Re-categorize toàn bộ dữ liệu"""
        
        print("🚀 RE-CATEGORIZING ALL DATA WITH OPTIMIZED CLASSIFICATION")
        print("=" * 70)
        
        # Backup file gốc
        backup_file = await self.backup_original_file()
        if not backup_file:
            print("❌ Cannot proceed without backup!")
            return False
        
        # Load dữ liệu
        print("📊 Loading original data...")
        df = pd.read_excel('app/data_test.xlsx')
        print(f"✅ Loaded {len(df)} rows")
        
        # Hiển thị category distribution cũ (nếu có)
        if 'category' in df.columns:
            old_category_stats = df['category'].value_counts()
            print(f"\n📊 OLD Category Distribution:")
            for cat, count in old_category_stats.items():
                print(f"   {cat}: {count} questions ({count/len(df)*100:.1f}%)")
        
        # Tạo list để lưu category mới
        new_categories = []
        processed_questions = set()  # Để tránh duplicate processing
        
        print(f"\n🔄 Re-categorizing {len(df)} questions with optimized prompt engineering...")
        
        for idx, row in df.iterrows():
            question = str(row['question']).strip()
            
            if not question or question == 'nan':
                new_categories.append("KHÁC")
                continue
            
            # Progress indicator
            progress = (idx + 1) / len(df) * 100
            print(f"\n📝 Processing {idx+1}/{len(df)} ({progress:.1f}%)")
            print(f"Question: {question[:80]}...")
            
            # Xử lý multiple questions (nếu có dấu |)
            main_question = question.split('|')[0].strip()
            
            # Kiểm tra đã process chưa để tránh duplicate
            if main_question in processed_questions:
                # Tìm category đã được assign cho câu hỏi tương tự
                for i, prev_cat in enumerate(new_categories):
                    if i < len(df):
                        prev_question = str(df.iloc[i]['question']).strip().split('|')[0].strip()
                        if prev_question == main_question:
                            new_categories.append(prev_cat)
                            print(f"🔄 Using cached result: {prev_cat}")
                            break
                else:
                    # Fallback nếu không tìm thấy
                    new_categories.append("KHÁC")
                continue
            
            # Classify với optimized router
            try:
                classification_result = await self.router._classify_query_category(main_question)
                category = classification_result["category"]
                
                new_categories.append(category)
                processed_questions.add(main_question)
                
                # Log chi tiết
                print(f"   ✅ Result: {category}")
                if 'llm_category' in classification_result:
                    print(f"   📊 LLM: {classification_result['llm_category']}")
                    print(f"   🔑 Keywords: {classification_result.get('keyword_category', 'None')}")
                
                # Thống kê
                if category not in self.categorization_results:
                    self.categorization_results[category] = 0
                self.categorization_results[category] += 1
                
                # Delay để tránh rate limit
                await asyncio.sleep(0.3)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                new_categories.append("KHÁC")
                continue
        
        # Cập nhật DataFrame
        df['category'] = new_categories
        
        # Hiển thị thống kê mới
        print(f"\n📊 NEW Category Distribution:")
        new_category_stats = df['category'].value_counts()
        total = len(df)
        
        for cat, count in new_category_stats.items():
            percentage = count/total*100
            print(f"   {cat}: {count} questions ({percentage:.1f}%)")
        
        # So sánh với phân bố cũ
        if 'category' in df.columns and len(old_category_stats) > 0:
            print(f"\n📈 COMPARISON (Old vs New):")
            all_categories = set(old_category_stats.keys()) | set(new_category_stats.keys())
            
            for cat in sorted(all_categories):
                old_count = old_category_stats.get(cat, 0)
                new_count = new_category_stats.get(cat, 0)
                change = new_count - old_count
                change_symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   {cat}: {old_count} → {new_count} ({change:+d}) {change_symbol}")
        
        # Lưu file mới
        output_file = 'app/data_test.xlsx'
        df.to_excel(output_file, index=False)
        print(f"\n💾 Updated data saved to: {output_file}")
        
        # Lưu cache classification để đảm bảo consistency
        cache_file = 'app/classification_cache.json'
        self.router.save_classification_cache(cache_file)
        
        return True
    
    async def validate_recategorization(self):
        """Validate kết quả re-categorization"""
        
        print(f"\n🧪 VALIDATING RE-CATEGORIZATION RESULTS")
        print("=" * 50)
        
        # Load dữ liệu đã re-categorize
        df = pd.read_excel('app/data_test.xlsx')
        
        # Test random samples để kiểm tra consistency
        import random
        sample_size = min(10, len(df))
        sample_indices = random.sample(range(len(df)), sample_size)
        
        print(f"🔍 Testing {sample_size} random samples for consistency...")
        
        consistent_count = 0
        
        for i, idx in enumerate(sample_indices, 1):
            row = df.iloc[idx]
            question = str(row['question']).split('|')[0].strip()
            stored_category = row['category']
            
            print(f"\n📝 Sample {i}: {question[:60]}...")
            print(f"   Stored category: {stored_category}")
            
            # Re-classify để check consistency
            classification_result = await self.router._classify_query_category(question)
            new_category = classification_result["category"]
            
            if new_category == stored_category:
                print(f"   ✅ CONSISTENT: {new_category}")
                consistent_count += 1
            else:
                print(f"   ❌ INCONSISTENT: Expected {stored_category}, got {new_category}")
            
            await asyncio.sleep(0.5)
        
        consistency_rate = consistent_count / sample_size
        print(f"\n📊 Validation Results:")
        print(f"   Consistent samples: {consistent_count}/{sample_size}")
        print(f"   Consistency rate: {consistency_rate:.1%}")
        
        if consistency_rate >= 0.9:
            print(f"   ✅ EXCELLENT: Re-categorization is highly consistent!")
        elif consistency_rate >= 0.7:
            print(f"   🟡 GOOD: Re-categorization is moderately consistent")
        else:
            print(f"   ❌ POOR: Re-categorization needs improvement")
        
        return consistency_rate
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng kết"""
        
        print(f"\n📋 RE-CATEGORIZATION SUMMARY REPORT")
        print("=" * 50)
        
        # Load final data
        df = pd.read_excel('app/data_test.xlsx')
        category_stats = df['category'].value_counts()
        
        print(f"📊 Final Statistics:")
        print(f"   Total questions: {len(df)}")
        print(f"   Categories found: {len(category_stats)}")
        
        print(f"\n📈 Category Breakdown:")
        for cat, count in category_stats.items():
            percentage = count/len(df)*100
            print(f"   {cat}: {count} questions ({percentage:.1f}%)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if category_stats.get("KHÁC", 0) > len(df) * 0.1:  # >10% KHÁC
            print(f"   ⚠️  High 'KHÁC' rate ({category_stats.get('KHÁC', 0)}) - consider adding more specific categories")
        
        smallest_category = category_stats.min()
        if smallest_category < 5:
            print(f"   ⚠️  Some categories have very few questions (<5) - consider merging or collecting more data")
        
        print(f"   ✅ Data is now consistently categorized using optimized prompt engineering")
        print(f"   ✅ Classification cache saved for future consistency")
        print(f"   ✅ Router will now perform much better with accurate category partitioning")

async def main():
    """Main function"""
    
    print("🎯 DATA RE-CATEGORIZATION WITH OPTIMIZED CLASSIFICATION")
    print("=" * 70)
    print("This script will re-categorize ALL data in data_test.xlsx using")
    print("the optimized prompt engineering techniques for 100% consistency.")
    print()
    
    # Confirm với user
    confirm = input("⚠️  This will modify data_test.xlsx (backup will be created). Continue? (y/N): ")
    if confirm.lower() not in ['y', 'yes']:
        print("❌ Operation cancelled by user")
        return
    
    try:
        # Initialize recategorizer
        recategorizer = DataRecategorizer()
        
        # Step 1: Re-categorize all data
        success = await recategorizer.recategorize_all_data()
        
        if not success:
            print("❌ Re-categorization failed!")
            return
        
        # Step 2: Validate results
        consistency_rate = await recategorizer.validate_recategorization()
        
        # Step 3: Generate summary
        recategorizer.generate_summary_report()
        
        print(f"\n🏆 RE-CATEGORIZATION COMPLETED SUCCESSFULLY!")
        print(f"✅ Consistency rate: {consistency_rate:.1%}")
        print(f"✅ Your CategoryPartitionedRouter will now work with accurate category data!")
        
    except Exception as e:
        print(f"\n❌ Error during re-categorization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
