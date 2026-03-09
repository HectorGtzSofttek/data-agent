import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from fastmcp import FastMCP
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

load_dotenv()

# Azure OpenAI client 
def make_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

# DB URI 
def build_database_uri() -> str:
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    db = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    sslmode = os.getenv("DB_SSLMODE", "require")

    if not all([host, user, password]):
        raise ValueError("Missing DB_HOST / DB_USER / DB_PASSWORD in .env")

    return f"postgres://{user}:{password}@{host}:{port}/{db}?sslmode={sslmode}"

# FastMCP client (postgres-mcp) 
def make_mcp_client() -> Client:
    transport = StdioTransport(
        command="postgres-mcp",
        args=["--access-mode=unrestricted", "--transport=stdio"],        
        env={"DATABASE_URI": build_database_uri()},
        keep_alive=True,
    )
    return Client(transport)

# Prompt: force JSON output from SQL 
SQL_SYSTEM = """You write SQL for PostgreSQL.

General rules:
- READ ONLY: only SELECT, WITH, or EXPLAIN queries.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or GRANT.
- Always return results as JSON in a single column named "rows".
- Always use this wrapper:

SELECT COALESCE(json_agg(t), '[]'::json) AS rows
FROM (
  -- inner query
) t;

Safety rules:
- Do NOT invent table names or column names.
- Use only table names and column names that appear in the provided schema context.
- If the user explicitly provides a table name and it exists in the schema context, query that table directly.
- If the user asks about a concept but does not provide an exact table name, choose the best matching table from the provided schema context.
- If no suitable table exists in the provided schema context, return a schema discovery query against information_schema.tables.
- Do not return schema metadata when the user asked for actual table data.

Query behavior:
- For counts, use COUNT(*).
- For "top", "highest", "largest", or "most", sort descending and use LIMIT.
- For grouped rankings, use GROUP BY when needed.
- For questions about zero values, filter with = 0.
- Trim text fields with TRIM().
- Alias aggregated metrics clearly, such as total_count, total_retries, or frequency.
- Prefer simple, direct SQL over unnecessary exploratory queries.

Return ONLY the SQL query. No explanations or commentary.
"""

def llm_to_sql(azure: AzureOpenAI, deployment: str, question: str, schema_context: str) -> str:
    resp = azure.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SQL_SYSTEM},
            {
                "role": "user",
                "content": f"""Question:
{question}

Available schema (real tables and columns from the database):
{schema_context}
"""
            },
        ],
    )
    return resp.choices[0].message.content.strip()

# Prompt: explain the DB result in natural language
ANSWER_SYSTEM = """You are a helpful assistant that explains database query results.

Your job:
- Given a user question and database rows (in JSON), produce a clear natural-language answer.

Guidelines:
- Be concise, accurate, and easy to understand.
- Focus on answering the user's question directly.
- Do not mention SQL unless necessary to explain an error.

Interpreting results:
- If rows contain numeric aggregates (counts, totals, averages), clearly state the number.
- If rows contain grouped results, explain the key finding (for example the most frequent item).
- If multiple rows are returned, summarize the important information rather than listing everything unless the user asks.

Empty results:
- If rows are empty (`[]`), say that no matching data was found.

Schema metadata:
- If the rows appear to describe database structure (for example columns like table_name, column_name, data_type),
  explain that the query returned schema information rather than actual table data.

Errors:
- If the result contains an error message, explain the issue clearly and briefly.
- If the error suggests a missing table or column, say that the referenced object does not exist in the database.

Clarity rules:
- Use simple language.
- Avoid repeating raw JSON in the explanation.
- Do not fabricate information that is not present in the rows.
"""

def llm_to_message(azure: AzureOpenAI, deployment: str, question: str, rows_json: str) -> str:
    resp = azure.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"User question:\n{question}\n\nDB rows (JSON):\n{rows_json}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# --- MCP Server ---
mcp = FastMCP(
    name="AI Database Assistant",
    instructions="Ask natural language questions about the database. Use ask_database to query and get answers.",
)

async def run_sql(client: Client, sql: str):
    result = await client.call_tool("execute_sql", {"sql": sql})

    if not result.content:
        raise ValueError("execute_sql returned no content")

    raw_text = getattr(result.content[0], "text", "")
    if not raw_text:
        raise ValueError("execute_sql returned empty text")

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "rows" in parsed[0]:
            return parsed[0]["rows"]
        return parsed
    except Exception:
        return raw_text


async def list_tables(client: Client):
    sql = """
    SELECT COALESCE(json_agg(t), '[]'::json) AS rows
    FROM (
      SELECT table_schema, table_name
      FROM information_schema.tables
      WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        AND table_type = 'BASE TABLE'
      ORDER BY table_schema, table_name
    ) t;
    """
    return await run_sql(client, sql)


async def list_columns(client: Client):
    sql = """
    SELECT COALESCE(json_agg(t), '[]'::json) AS rows
    FROM (
      SELECT table_schema, table_name, column_name, data_type, ordinal_position
      FROM information_schema.columns
      WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
      ORDER BY table_schema, table_name, ordinal_position
    ) t;
    """
    return await run_sql(client, sql)


@mcp.tool
async def ask_database(question: str) -> str:
    """
    Ask a natural language question about the database. Returns an answer in plain language plus the underlying data as JSON.
    """
    azure = make_azure_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT_NAME in .env")

    client = make_mcp_client()
    async with client:
        # 1) Discover real schema first
        tables = await list_tables(client)
        columns = await list_columns(client)

        schema_context = json.dumps(
            {
                "tables": tables,
                "columns": columns,
            },
            ensure_ascii=False,
            indent=2,
        )

        # 2) LLM → SQL using real schema context
        sql = llm_to_sql(azure, deployment, question, schema_context)

        # 3) Execute SQL
        rows = await run_sql(client, sql)

        rows_json = json.dumps(rows, ensure_ascii=False, indent=2) if not isinstance(rows, str) else rows

        # 4) LLM → natural language answer
        message = llm_to_message(azure, deployment, question, rows_json)

    return f"{message}\n\nData:\n{rows_json}"


if __name__ == "__main__":
    mcp.run()
