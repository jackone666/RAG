"""
文档对象存储中间件 — 基于 MinIO 的原始文件持久化层

设计目的（参考字节 RAG 实践 §2.2「数据层设计」）：
- 原始文档与向量/索引解耦存储，保证 embedding 模型可更换
- 切换 embedding 模型时可直接从对象存储重取文档重新入库
- 支持文档的完整生命周期管理（存储→检索→删除）

存储结构：
  s3://{bucket}/{tenant_id}/{doc_id}/{original_filename}
"""
import io
from pathlib import Path

from minio import Minio
from minio.error import S3Error
from loguru import logger

from src.config.settings import settings


class DocumentObjectStore:
    """MinIO 文档对象存储：保存原始上传文件，供模型迁移时重新入库。"""

    def __init__(self):
        self._client: Minio | None = None

    @property
    def client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        """确保文档存储桶存在。"""
        bucket = settings.minio_bucket
        try:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info(f"MinIO bucket 已创建: {bucket}")
        except Exception as e:
            logger.warning(f"MinIO bucket 检查失败: {e}")

    def _object_path(self, tenant_id: str, doc_id: str, filename: str) -> str:
        """构造对象存储路径：{tenant_id}/{doc_id}/{filename}"""
        return f"{tenant_id}/{doc_id}/{Path(filename).name}"

    def store_document(
        self, tenant_id: str, doc_id: str, filename: str, content: bytes
    ) -> str:
        """存储原始文档到 MinIO。

        Returns:
            存储路径
        """
        path = self._object_path(tenant_id, doc_id, filename)
        try:
            self.client.put_object(
                bucket_name=settings.minio_bucket,
                object_name=path,
                data=io.BytesIO(content),
                length=len(content),
                content_type="application/octet-stream",
                metadata={
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                },
            )
            logger.info(f"文档已存储至 MinIO: {path} ({len(content)} bytes)")
        except S3Error as e:
            logger.error(f"MinIO 存储失败: {e}")
            raise
        return path

    def get_document(self, tenant_id: str, doc_id: str, filename: str) -> bytes | None:
        """从 MinIO 读取原始文档内容。"""
        path = self._object_path(tenant_id, doc_id, filename)
        try:
            resp = self.client.get_object(
                bucket_name=settings.minio_bucket,
                object_name=path,
            )
            data = resp.read()
            resp.close()
            return data
        except S3Error as e:
            logger.warning(f"MinIO 读取失败: {path}: {e}")
            return None

    def delete_document(self, tenant_id: str, doc_id: str, filename: str) -> bool:
        """从 MinIO 删除文档。"""
        path = self._object_path(tenant_id, doc_id, filename)
        try:
            self.client.remove_object(settings.minio_bucket, path)
            logger.info(f"MinIO 文档已删除: {path}")
            return True
        except S3Error as e:
            logger.warning(f"MinIO 删除失败: {path}: {e}")
            return False

    def delete_tenant_objects(self, tenant_id: str) -> int:
        """删除某租户的所有文档对象（GDPR 场景）。"""
        prefix = f"{tenant_id}/"
        count = 0
        try:
            objs = self.client.list_objects(
                settings.minio_bucket, prefix=prefix, recursive=True
            )
            for obj in objs:
                self.client.remove_object(settings.minio_bucket, obj.object_name)
                count += 1
            logger.info(f"MinIO 已清除租户 {tenant_id} 的 {count} 个对象")
        except S3Error as e:
            logger.warning(f"MinIO 清理租户失败: {e}")
        return count


# 模块级单例
doc_store = DocumentObjectStore()
