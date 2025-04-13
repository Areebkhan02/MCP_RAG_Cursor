import asyncio
from dotenv import load_dotenv
from linkup import LinkupClient
from tools.CrewAI_RAG import CrewAI_RAG
from mcp.server.fastmcp import FastMCP
import os

load_dotenv()

mcp = FastMCP('CrewAI_RAG_linkup-server')
client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))
rag_workflow = CrewAI_RAG()

@mcp.tool()
def deep_web_search(query: str) -> str:
    """Search the web for the given query."""
    search_response = client.search(
        query=query,
        depth="standard",  # "standard" or "deep"
        output_type="sourcedAnswer",  # "searchResults" or "sourcedAnswer" or "structured"
        structured_output_schema=None,  # must be filled if output_type is "structured"
    )
    return search_response

@mcp.tool()
async def CrewAI_RAG(query: str) -> str:
    """Use a simple RAG workflow to answer queries using documents from data directory about Deep Seek"""
    response = await rag_workflow.query(query)
    return str(response)

if __name__ == "__main__":
    
    mcp.run(transport="stdio")
    #mcp.run(transport="http", host="127.0.0.1", port=8000)

