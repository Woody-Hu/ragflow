import logging
import psycopg2
import psycopg2.extras
import pandas as pd
from typing import List, Dict, Tuple, Any
import json
from rag import settings
from rag.utils import singleton
from rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchExpr,
    MatchTextExpr,
    MatchDenseExpr,
    OrderByExpr,
)
import re

logger = logging.getLogger('ragflow.postgres_conn')

@singleton
class PostgresConnection(DocStoreConnection):
    def __init__(self):
        self.conn_params = settings.POSTGRES
        self.text_search_lang = settings.POSTGRES.get('text_search_lang', 'english')
        self.enable_total_count = settings.POSTGRES.get('enable_total_count', True)
        self.conn = psycopg2.connect(**self.conn_params)
        self._init_db()

    def _init_db(self):
        with self.conn.cursor() as cur:
            # Only create vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()

    def dbType(self) -> str:
        return "postgres"

    def health(self) -> dict:
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return {"type": "postgres", "status": "green"}
        except Exception as e:
            return {"type": "postgres", "status": "red", "error": str(e)}

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        table_name = f"{indexName}_{knowledgebaseId}"
        vector_col = f"q_{vectorSize}_vec"
        
        with self.conn.cursor() as cur:
            # Create table with schema matching infinity_mapping.json
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR PRIMARY KEY,
                    doc_id VARCHAR DEFAULT '',
                    kb_id VARCHAR DEFAULT '',
                    create_time VARCHAR DEFAULT '',
                    create_timestamp_flt FLOAT DEFAULT 0.0,
                    img_id VARCHAR DEFAULT '',
                    docnm_kwd VARCHAR DEFAULT '',
                    title_tks VARCHAR DEFAULT '',
                    title_sm_tks VARCHAR DEFAULT '',
                    name_kwd VARCHAR DEFAULT '',
                    important_kwd TEXT DEFAULT '',
                    tag_kwd TEXT DEFAULT '',
                    important_tks VARCHAR DEFAULT '',
                    question_kwd TEXT DEFAULT '',
                    question_tks VARCHAR DEFAULT '',
                    content_with_weight TEXT DEFAULT '',
                    content_ltks TEXT DEFAULT '',
                    content_sm_ltks VARCHAR DEFAULT '',
                    authors_tks VARCHAR DEFAULT '',
                    authors_sm_tks VARCHAR DEFAULT '',
                    page_num_int TEXT DEFAULT '',
                    top_int TEXT DEFAULT '',
                    position_int TEXT DEFAULT '',
                    weight_int INTEGER DEFAULT 0,
                    weight_flt FLOAT DEFAULT 0.0,
                    rank_int INTEGER DEFAULT 0,
                    rank_flt FLOAT DEFAULT 0,
                    available_int INTEGER DEFAULT 1,
                    knowledge_graph_kwd VARCHAR DEFAULT '',
                    entities_kwd TEXT DEFAULT '',
                    pagerank_fea INTEGER DEFAULT 0,
                    tag_feas TEXT DEFAULT '',
                    {vector_col} vector({vectorSize})
                )
            """)
            
            # Create vector index
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_vec 
                ON {table_name} USING ivfflat ({vector_col} vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            # Create text search indexes for base fields
            text_fields = ["content_ltks", "important_kwd", "question_kwd"]
            for field in text_fields:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_{field}_fts 
                    ON {table_name} USING GIN (to_tsvector(%s, {field}))
                """, (self.text_search_lang,))
            
            self.conn.commit()
            logger.info(f"Created table and indexes {table_name} with vector size {vectorSize}")

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        table_name = f"{indexName}_{knowledgebaseId}"
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.commit()
            logger.info(f"Dropped table {table_name}")

    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:
        table_name = f"{indexName}_{knowledgebaseId}"
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            return cur.fetchone()[0]

    def search(
            self, selectFields: list[str],
            highlightFields: list[str],
            condition: dict,
            matchExprs: list[MatchExpr],
            orderBy: OrderByExpr,
            offset: int,
            limit: int,
            indexNames: str | list[str],
            knowledgebaseIds: list[str],
            aggFields: list[str] = [],
            rank_feature: dict | None = None
    ) -> tuple[pd.DataFrame, int]:
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")

        # Create UNION query for all tables
        queries = []
        all_params = []
        
        for indexName in indexNames:
            for knowledgebaseId in knowledgebaseIds:
                table_name = f"{indexName}_{knowledgebaseId}"
                select_clause = ", ".join(selectFields) if selectFields else "*"
                where_conditions = []
                params = []

                # Handle base conditions
                if condition:
                    for k, v in condition.items():
                        if isinstance(v, list):
                            placeholders = ",".join(["%s" for _ in v])
                            where_conditions.append(f"{k} IN ({placeholders})")
                            params.extend(v)
                        else:
                            where_conditions.append(f"{k} = %s")
                            params.append(v)

                # Handle match expressions
                for expr in matchExprs:
                    if isinstance(expr, MatchTextExpr):
                        fields = expr.fields
                        for field in fields:
                            where_conditions.append(
                                f"to_tsvector(%s, {field}) @@ plainto_tsquery(%s, %s)"
                            )
                            params.extend([self.text_search_lang, self.text_search_lang, expr.matching_text])
                    elif isinstance(expr, MatchDenseExpr):
                        where_conditions.append(
                            f"cosine_distance({expr.vector_column_name}, %s::vector) <= %s"
                        )
                        params.extend([expr.embedding_data, expr.extra_options.get("threshold", 0.8)])

                # Construct query for this table
                query = f"SELECT {select_clause} FROM {table_name}"
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
                queries.append(query)
                all_params.extend(params)

        # Combine all queries with UNION ALL
        combined_sql = " UNION ALL ".join(f"({q})" for q in queries)

        # Add ordering and pagination to combined results
        if orderBy and orderBy.fields:
            order_clauses = []
            for field, direction in orderBy.fields:
                order_clauses.append(f"{field} {'DESC' if direction else 'ASC'}")
            combined_sql += " ORDER BY " + ", ".join(order_clauses)

        combined_sql += f" LIMIT %s OFFSET %s"
        all_params.extend([limit, offset])

        # Execute query
        total_count = 0
        with self.conn.cursor() as cur:
            # Get total count only if enabled
            if self.enable_total_count:
                count_sql = " UNION ALL ".join(f"(SELECT COUNT(*) FROM ({q}) t)" for q in queries)
                cur.execute(count_sql, all_params[:-2])
                total_count = sum(row[0] for row in cur.fetchall())

            # Get actual results
            cur.execute(combined_sql, all_params)
            columns = [desc[0] for desc in cur.description]
            results = cur.fetchall()
            df = pd.DataFrame(results, columns=columns)
            
            # If total count is disabled, use result length
            if not self.enable_total_count:
                total_count = len(df)

        return df, total_count

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            placeholders = ",".join(["%s" for _ in knowledgebaseIds])
            sql = f"SELECT * FROM {indexName}_{knowledgebaseIds[0]} WHERE id = %s AND kb_id IN ({placeholders})"
            cur.execute(sql, [chunkId] + knowledgebaseIds)
            row = cur.fetchone()
            if row:
                return dict(row)
        return None

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        if not documents:
            return []

        # Process documents similar to InfinityConnection
        docs = []
        for doc in documents:
            processed_doc = {}
            for k, v in doc.items():
                if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd"]:
                    processed_doc[k] = "###".join(v) if isinstance(v, list) else v
                elif k == "position_int":
                    if isinstance(v, list):
                        arr = [num for row in v for num in row]
                        processed_doc[k] = "_".join(f"{num:08x}" for num in arr)
                    else:
                        processed_doc[k] = v
                elif k in ["page_num_int", "top_int"]:
                    if isinstance(v, list):
                        processed_doc[k] = "_".join(f"{num:08x}" for num in v)
                    else:
                        processed_doc[k] = v
                elif k == "tag_feas":
                    processed_doc[k] = json.dumps(v) if v else "{}"
                else:
                    processed_doc[k] = v
            docs.append(processed_doc)

        # Insert using psycopg2
        with self.conn.cursor() as cur:
            columns = docs[0].keys()
            values_template = ",".join(["%s" for _ in columns])
            insert_sql = f"""
                INSERT INTO {indexName}_{knowledgebaseId} ({",".join(columns)}) 
                VALUES ({values_template})
                ON CONFLICT (id) DO UPDATE SET 
                {",".join([f"{col}=EXCLUDED.{col}" for col in columns])}
            """
            
            values = [[doc[col] for col in columns] for doc in docs]
            psycopg2.extras.execute_batch(cur, insert_sql, values)
            self.conn.commit()

        return []

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        # Process new values
        processed_values = {}
        for k, v in newValue.items():
            if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd"]:
                processed_values[k] = "###".join(v) if isinstance(v, list) else v
            elif k == "position_int":
                if isinstance(v, list):
                    arr = [num for row in v for num in row]
                    processed_values[k] = "_".join(f"{num:08x}" for num in arr)
                else:
                    processed_values[k] = v
            elif k in ["page_num_int", "top_int"]:
                if isinstance(v, list):
                    processed_values[k] = "_".join(f"{num:08x}" for num in v)
                else:
                    processed_values[k] = v
            elif k == "tag_feas":
                processed_values[k] = json.dumps(v) if v else "{}"
            else:
                processed_values[k] = v

        # Build update SQL
        set_clause = ", ".join([f"{k} = %s" for k in processed_values.keys()])
        where_conditions = []
        params = list(processed_values.values())

        for k, v in condition.items():
            if isinstance(v, list):
                placeholders = ",".join(["%s" for _ in v])
                where_conditions.append(f"{k} IN ({placeholders})")
                params.extend(v)
            else:
                where_conditions.append(f"{k} = %s")
                params.append(v)

        where_clause = " AND ".join(where_conditions)
        sql = f"UPDATE {indexName}_{knowledgebaseId} SET {set_clause} WHERE {where_clause}"

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            self.conn.commit()
            return True

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        where_conditions = []
        params = []

        for k, v in condition.items():
            if isinstance(v, list):
                placeholders = ",".join(["%s" for _ in v])
                where_conditions.append(f"{k} IN ({placeholders})")
                params.extend(v)
            else:
                where_conditions.append(f"{k} = %s")
                params.append(v)

        where_clause = " AND ".join(where_conditions)
        sql = f"DELETE FROM {indexName}_{knowledgebaseId} WHERE {where_clause}"

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            self.conn.commit()
            return cur.rowcount

    # Helper methods for search results
    def getTotal(self, res: tuple[pd.DataFrame, int] | pd.DataFrame) -> int:
        if isinstance(res, tuple):
            return res[1]
        return len(res)

    def getChunkIds(self, res: tuple[pd.DataFrame, int] | pd.DataFrame) -> list[str]:
        if isinstance(res, tuple):
            res = res[0]
        return list(res["id"])

    def getFields(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, fields: list[str]) -> dict[str, dict]:
        if isinstance(res, tuple):
            res = res[0]
        if not fields:
            return {}

        result_dict = {}
        for _, row in res.iterrows():
            chunk_data = {}
            for field in fields:
                value = row.get(field)
                if field in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd"]:
                    chunk_data[field] = value.split("###") if value else []
                elif field == "position_int":
                    if value:
                        arr = [int(x, 16) for x in value.split("_")]
                        chunk_data[field] = [arr[i:i+5] for i in range(0, len(arr), 5)]
                    else:
                        chunk_data[field] = []
                elif field in ["page_num_int", "top_int"]:
                    chunk_data[field] = [int(x, 16) for x in value.split("_")] if value else []
                elif field == "tag_feas":
                    chunk_data[field] = json.loads(value) if value else {}
                else:
                    chunk_data[field] = value
            result_dict[row["id"]] = chunk_data

        return result_dict

    def getHighlight(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, keywords: list[str], fieldnm: str):
        if isinstance(res, tuple):
            res = res[0]
        
        highlights = {}
        for _, row in res.iterrows():
            if fieldnm not in row:
                continue
                
            text = row[fieldnm]
            highlighted_text = text
            for keyword in keywords:
                highlighted_text = re.sub(
                    f"({keyword})",
                    r"<em>\1</em>",
                    highlighted_text,
                    flags=re.IGNORECASE
                )
            if highlighted_text != text:
                highlights[row["id"]] = highlighted_text
                
        return highlights

    def getAggregation(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, fieldnm: str):
        return []

    def sql(self, sql: str, fetch_size: int, format: str):
        with self.conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            results = []
            while True:
                rows = cur.fetchmany(fetch_size)
                if not rows:
                    break
                results.extend(rows)
            
            if format == "dataframe":
                return pd.DataFrame(results, columns=columns)
            return results