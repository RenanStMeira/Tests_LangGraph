from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Meu servidor MCP")

@mcp.tool()
def get_community(Location: str) -> str:
    """Comunidade de Python para GenAi"""
    return "Code TI"

if __name__ == "__name__":
    mcp.run(transport="stdio")