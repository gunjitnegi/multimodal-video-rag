import traceback
from app.services.embedding_service import embedding_service

try:
    print(embedding_service.search("XCIYHCXQoxQ", "test query"))
except Exception as e:
    traceback.print_exc()
