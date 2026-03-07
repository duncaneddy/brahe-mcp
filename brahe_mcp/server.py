import brahe
from mcp.server.fastmcp import FastMCP

# Initialize EOP and space weather data before any submodule import.
# Modules that perform frame transforms or propagation depend on this.
brahe.initialize_eop()
brahe.initialize_sw()

mcp = FastMCP("brahe")

# Import submodules to register their tools with the mcp instance
import brahe_mcp.constants  # noqa: F401, E402
import brahe_mcp.epochs  # noqa: F401, E402
import brahe_mcp.orbits  # noqa: F401, E402
import brahe_mcp.coordinates  # noqa: F401, E402
import brahe_mcp.celestrak  # noqa: F401, E402
import brahe_mcp.spacetrack  # noqa: F401, E402
import brahe_mcp.gcat  # noqa: F401, E402
import brahe_mcp.groundstations  # noqa: F401, E402
import brahe_mcp.propagation  # noqa: F401, E402
import brahe_mcp.accesses  # noqa: F401, E402
import brahe_mcp.plotting  # noqa: F401, E402
