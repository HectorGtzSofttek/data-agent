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
        args=["--access-mode=restricted", "--transport=stdio"],
        env={"DATABASE_URI": build_database_uri()},
        keep_alive=True,
    )
    return Client(transport)

# Prompt: force JSON output from SQL 
SQL_SYSTEM = """You write SQL for PostgreSQL.
Rules:
- READ ONLY: only SELECT/WITH/EXPLAIN.
- Never INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
- ALWAYS return results as JSON in a single column named "rows".
- Use json_agg over a subquery.
- If no rows, return [].
- Trim string fields.
Return ONLY the SQL. No commentary.
"""

def llm_to_sql(azure: AzureOpenAI, deployment: str, question: str) -> str:
    resp = azure.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()

# Prompt: explain the DB result in natural language
ANSWER_SYSTEM = """You are a helpful assistant.
Given a user question and database rows, answer in natural language.
Be concise and clear.
If rows are empty, say you found no results.
If the user asks for a number, give the number.
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
        # 1) LLM → SQL (JSON-first)
        sql = llm_to_sql(azure, deployment, question)

        # 2) MCP → execute SQL
        result = await client.call_tool("execute_sql", {"sql": sql})
        print("result: ", result)
        raw_text = result.content[0].text
        print("raw_text: ", raw_text)

        try:
            parsed = json.loads(raw_text)
            print("parsed: ", parsed)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "rows" in parsed[0]:
                rows = parsed[0]["rows"]
            else:
                rows = parsed
        except Exception:
            rows = raw_text

        rows_json = json.dumps(rows, ensure_ascii=False, indent=2) if not isinstance(rows, str) else rows

        # 3) LLM → natural language message
        message = llm_to_message(azure, deployment, question, rows_json)

    return f"{message}\n\nData:\n{rows_json}"


if __name__ == "__main__":
    mcp.run()
