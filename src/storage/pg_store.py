"""
PostgreSQL 数据持久化模块 — 复用 Langfuse 的 PostgreSQL 实例

存储项目运行数据：bad_cases（评估坏例）、docstore（文档哈希缓存）。
"""
import json
import threading

import psycopg2
import psycopg2.extras
from loguru import logger

# PostgreSQL 连接配置（复用 langfuse-db）
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "langfuse",
    "user": "langfuse",
    "password": "langfuse",
}

_lock = threading.Lock()

# 初始建表
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS bad_cases (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(128) NOT NULL DEFAULT 'default',
    query TEXT,
    answer TEXT,
    context_nodes JSONB DEFAULT '[]',
    faithfulness FLOAT DEFAULT 0,
    relevancy FLOAT DEFAULT 0,
    correctness FLOAT DEFAULT 0,
    completeness FLOAT DEFAULT 0,
    overall FLOAT DEFAULT 0,
    passing BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bad_cases_tenant ON bad_cases(tenant_id);
CREATE INDEX IF NOT EXISTS idx_bad_cases_overall ON bad_cases(overall);
CREATE INDEX IF NOT EXISTS idx_bad_cases_created ON bad_cases(created_at);

-- 新增检索指标列（幂等）
ALTER TABLE bad_cases ADD COLUMN IF NOT EXISTS precision FLOAT DEFAULT 0;
ALTER TABLE bad_cases ADD COLUMN IF NOT EXISTS recall FLOAT DEFAULT 0;
ALTER TABLE bad_cases ADD COLUMN IF NOT EXISTS mrr FLOAT DEFAULT 0;
ALTER TABLE bad_cases ADD COLUMN IF NOT EXISTS hit_rate FLOAT DEFAULT 0;
ALTER TABLE bad_cases ADD COLUMN IF NOT EXISTS relevance FLOAT DEFAULT 0;

CREATE TABLE IF NOT EXISTS docstore (
    doc_id VARCHAR(256) PRIMARY KEY,
    content_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_docstore_hash ON docstore(content_hash);
"""


def _get_conn():
    """获取 PostgreSQL 连接。"""
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """初始化数据库表（幂等）。"""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(_INIT_SQL)
        conn.commit()
        conn.close()
        logger.info("PostgreSQL 表初始化完成 (bad_cases, docstore)")
    except Exception as e:
        logger.warning(f"PostgreSQL 初始化失败（降级为文件存储）: {e}")


# ==================== Bad Cases ====================

def insert_bad_case(case: dict) -> None:
    """插入坏例记录。"""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO bad_cases (tenant_id, query, answer, context_nodes,
                   precision, recall, mrr, hit_rate, faithfulness, relevance, overall, passing)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    case.get("tenant_id", "default"),
                    case.get("query", ""),
                    case.get("answer", ""),
                    json.dumps(case.get("context_nodes", []), ensure_ascii=False),
                    case.get("precision", 0),
                    case.get("recall", 0),
                    case.get("mrr", 0),
                    case.get("hit_rate", 0),
                    case.get("faithfulness", 0),
                    case.get("relevance", 0),
                    case.get("overall", 0),
                    case.get("passing", False),
                ),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"写入 bad_cases 失败: {e}")


def get_bad_case_stats() -> dict:
    """获取评估统计数据。"""
    try:
        conn = _get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) as total FROM bad_cases")
            total = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) as bad FROM bad_cases WHERE passing = FALSE")
            bad = cur.fetchone()["bad"]

            cur.execute("SELECT AVG(overall) as avg_score FROM bad_cases")
            avg = cur.fetchone()["avg_score"] or 0

            cur.execute(
                "SELECT AVG(precision) as p, AVG(recall) as r, AVG(mrr) as m, AVG(hit_rate) as h, AVG(faithfulness) as f, AVG(relevance) as rel FROM bad_cases")
            avgs = cur.fetchone()

            cur.execute(
                "SELECT * FROM bad_cases WHERE passing = FALSE ORDER BY overall ASC LIMIT 10"
            )
            recent = [dict(r) for r in cur.fetchall()]
            for r in recent:
                r["created_at"] = str(r.get("created_at", ""))
                if isinstance(r.get("context_nodes"), str):
                    try:
                        r["context_nodes"] = json.loads(r["context_nodes"])
                    except Exception:
                        pass

        conn.close()

        return {
            "total_queries": total,
            "bad_cases": bad,
            "pass_rate": round((total - bad) / total, 4) if total > 0 else 1.0,
            "avg_score": round(avg, 4),
            "avg_precision": round(avgs["p"] or 0, 4),
            "avg_recall": round(avgs["r"] or 0, 4),
            "avg_mrr": round(avgs["m"] or 0, 4),
            "avg_hit_rate": round(avgs["h"] or 0, 4),
            "avg_faithfulness": round(avgs["f"] or 0, 4),
            "avg_relevance": round(avgs["rel"] or 0, 4),
            "recent_bad_cases": recent,
        }
    except Exception as e:
        logger.warning(f"查询 bad_cases 统计失败: {e}")
        return {
            "total_queries": 0, "bad_cases": 0, "pass_rate": 1.0,
            "avg_score": 0, "avg_faithfulness": 0, "avg_relevancy": 0,
            "avg_correctness": 0, "avg_completeness": 0, "recent_bad_cases": [],
        }


# ==================== Docstore ====================

def set_doc_hash(doc_id: str, content_hash: str) -> None:
    """记录文档哈希。"""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO docstore (doc_id, content_hash) VALUES (%s, %s) ON CONFLICT (doc_id) DO UPDATE SET content_hash = %s, created_at = NOW()",
                (doc_id, content_hash, content_hash),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"写入 docstore 失败: {e}")


def get_doc_hash(doc_id: str) -> str | None:
    """获取文档哈希。"""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT content_hash FROM docstore WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"查询 docstore 失败: {e}")
        return None
