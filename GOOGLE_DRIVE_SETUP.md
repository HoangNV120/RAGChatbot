# Hướng dẫn thiết lập Google Drive Integration và Auto Scheduler (Service Account)

## 1. Thiết lập Google Drive API

### Tạo Google Cloud Project và Enable Drive API
1. Truy cập [Google Cloud Console](https://console.cloud.google.com/)
2. Tạo project mới hoặc chọn project hiện có
3. Enable Google Drive API:
   - Vào "APIs & Services" > "Library"
   - Tìm "Google Drive API" và click "Enable"

### Tạo Service Account
1. Vào "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Đặt tên service account (ví dụ: "rag-chatbot-service")
4. Không cần gán role đặc biệt, click "Done"
5. Click vào service account vừa tạo
6. Vào tab "Keys" > "Add Key" > "Create new key"
7. Chọn "JSON" và download
8. Đặt tên file là `service-account.json`
9. Copy file này vào thư mục gốc của project

### Chia sẻ Google Drive folder với Service Account
**⚠️ Bước quan trọng**: Bạn phải chia sẻ Google Drive folder với Service Account
1. Mở file `service-account.json` và copy email của service account:
   ```json
   {
     "client_email": "rag-chatbot-service@your-project.iam.gserviceaccount.com"
   }
   ```
2. Vào Google Drive, click chuột phải vào folder cần sync
3. Chọn "Share" > "Add people and groups"
4. Paste email của service account
5. Chọn role "Viewer" (chỉ cần đọc)
6. Click "Send"

### Lấy Google Drive Folder ID
1. Truy cập Google Drive
2. Mở folder đã chia sẻ với service account
3. Copy ID từ URL:
   ```
   https://drive.google.com/drive/folders/1abcdef123456789
   ```
   ID folder là: `1abcdef123456789`

## 2. Cấu hình Environment Variables

### Tạo file .env
```bash
cp .env.example .env
```

### Cập nhật file .env
```env
GOOGLE_DRIVE_FOLDER_ID=1abcdef123456789  # Thay bằng ID folder thực tế
GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json
SCHEDULER_INTERVAL_HOURS=1  # Sync mỗi 1 giờ
```

## 3. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

## 4. Chạy ứng dụng

### Local/Development
```bash
python main.py
```

### Production/VPS
```bash
# Đảm bảo có file service-account.json
export GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json
python main.py
```

- ✅ Không cần browser, xác thực tự động
- ✅ Hoàn hảo cho VPS/Docker/Headless server
- ✅ Bảo mật cao và ổn định

## 5. Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements và install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Copy service account file
COPY service-account.json .

# Set environment variables
ENV GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json
ENV GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here

EXPOSE 8000

CMD ["python", "main.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  rag-chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_DRIVE_FOLDER_ID=1abcdef123456789
      - GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json
      - SCHEDULER_INTERVAL_HOURS=1
    volumes:
      - ./service-account.json:/app/service-account.json:ro
      - ./app/data:/app/app/data
```

## 6. Tính năng chính

### Service Account Authentication
```
🔧 Google Drive authentication: Service Account (service-account.json)
🔑 Authenticating with service account: service-account.json
✅ Service Account authentication successful
```

### Auto Sync và Processing
- **Khởi tạo**: Chạy sync ngay khi start ứng dụng
- **Định kỳ**: Sync theo interval đã cấu hình (mặc định 1 giờ)
- **Smart Processing**: 
  - Tự động phát hiện file mới/cập nhật/xóa
  - Update documents đã tồn tại trong vector store
  - Insert documents mới
  - Delete documents khi file bị xóa khỏi Drive
- **Auto cleanup**: Xóa file source sau khi process thành công

### Loại file được hỗ trợ
- **JSON**: Syllabus, Curriculum documents
- **PDF**: Decision documents với OCR support (tiếng Việt + tiếng Anh)
- **Excel**: FAQ documents (.xlsx, .xls)

## 7. API Endpoints

### Kiểm tra trạng thái scheduler
```bash
GET /scheduler/status
```
Response:
```json
{
  "is_running": true,
  "schedule_interval_hours": 1,
  "data_directory": "/app/app/data",
  "google_drive_folder_id": "1abcdef123456789",
  "last_run": {
    "timestamp": "2025-07-10T10:30:00",
    "sync_results": {
      "downloaded": 2,
      "updated": 1,
      "deleted": 0,
      "skipped": 5
    }
  },
  "total_runs": 15
}
```

### Chạy sync thủ công
```bash
POST /scheduler/manual-sync
```

### Quản lý documents trong vector store
```bash
GET /vector-store/documents           # List all documents
GET /vector-store/documents/{name}    # Get document by name
DELETE /vector-store/documents/{name} # Delete document by name
PUT /vector-store/documents           # Update document
```

## 8. Workflow hoạt động

### Sync Process
1. **Authenticate** với Google Drive bằng Service Account
2. **List files** trong folder được chia sẻ
3. **Compare** với metadata local để phát hiện thay đổi
4. **Download/Update** file nếu có thay đổi
5. **Delete** file local nếu bị xóa khỏi Drive

### Smart Processing
1. **Load existing documents** từ vector store
2. **Check file type** (JSON/PDF/Excel)
3. **Smart decision**:
   - File mới → Insert vào vector store
   - File cập nhật → Update document trong vector store
   - File xóa → Delete document khỏi vector store
4. **Clean up** source file sau khi xử lý

## 9. Cấu trúc thư mục

```
RAGChatbotai/
├── service-account.json      # Service Account credentials
├── .env                      # Environment variables
├── main.py                   # Application entry point
├── app/
│   ├── data/
│   │   ├── sync_metadata.json    # File sync tracking
│   │   ├── processing_log.json   # Processing history
│   │   └── [downloaded files]    # Files from Google Drive
│   ├── google_drive_sync.py      # Google Drive integration
│   ├── smart_scheduler.py        # Auto scheduler
│   └── ...
└── ...
```

## 10. Troubleshooting

### Service Account Issues
```bash
❌ Service Account authentication failed: [403] The caller does not have permission
```
**Giải pháp**: 
- Đảm bảo đã chia sẻ folder với service account email
- Kiểm tra service account có quyền truy cập Google Drive API

```bash
❌ Service Account authentication failed: [404] File not found: service-account.json
```
**Giải pháp**: 
- Kiểm tra file `service-account.json` có tồn tại không
- Kiểm tra đường dẫn trong environment variable `GOOGLE_SERVICE_ACCOUNT_FILE`

### Sync Issues
```bash
📁 Found 0 files in Google Drive folder
```
**Giải pháp**:
- Kiểm tra Folder ID có đúng không
- Đảm bảo folder đã được chia sẻ với service account
- Kiểm tra folder có chứa file JSON/PDF/Excel không

## 11. Best Practices

### Security
- Không commit `service-account.json` vào git
- Sử dụng environment variables cho production
- Chỉ cấp quyền "Viewer" cho service account
- Định kỳ rotate service account keys

### Performance
- Điều chỉnh `SCHEDULER_INTERVAL_HOURS` phù hợp với nhu cầu
- Monitor logs để theo dõi hiệu suất sync
- Sử dụng file nhỏ hơn 10MB để tối ưu tốc độ

### Monitoring
- Theo dõi logs trong console
- Kiểm tra file `processing_log.json` để debug
- Sử dụng API `/scheduler/status` để monitor

### .gitignore
```gitignore
service-account.json
token.json
.env
app/data/sync_metadata.json
app/data/processing_log.json
app/data/*.json
app/data/*.pdf
app/data/*.xlsx
```

## 12. Tính năng nâng cao

### Cấu hình Interval khác nhau
```env
SCHEDULER_INTERVAL_HOURS=0.5  # Sync mỗi 30 phút
SCHEDULER_INTERVAL_HOURS=6    # Sync mỗi 6 giờ
```

### Chạy manual operations
```python
# Trigger sync thủ công
curl -X POST http://localhost:8000/scheduler/manual-sync

# Xem trạng thái
curl http://localhost:8000/scheduler/status

# List documents
curl http://localhost:8000/vector-store/documents
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

Hệ thống đã được tối ưu hoàn toàn cho Service Account - phương pháp tốt nhất cho production deployment! 🚀
