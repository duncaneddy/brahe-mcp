from mcp.server.fastmcp import FastMCP

mcp = FastMCP("brahe")

# Import submodules to register their tools with the mcp instance
import brahe_mcp.constants  # noqa: F401, E402
