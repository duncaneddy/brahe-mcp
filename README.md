# brahe-mcp

MCP server wrapping the [Brahe](https://github.com/duncaneddy/brahe) astrodynamics library.

## Installation

```bash
uv add brahe-mcp
```

or 

```bash
pip install brahe-mcp
```

## Local Setup

To run the server from a local clone (useful for development or testing before installing):

```bash
git clone https://github.com/duncaneddy/brahe-mcp.git
cd brahe-mcp
uv sync --group dev
```

Then configure your MCP client to launch the server via `uv run`:

**Claude Desktop** — edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "brahe": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/brahe-mcp", "brahe-mcp"]
    }
  }
}
```

**Claude Code** — edit `.claude/settings.json` (project-level or global):

```json
{
  "mcpServers": {
    "brahe": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/brahe-mcp", "brahe-mcp"]
    }
  }
}
```

Replace `/path/to/brahe-mcp` with the absolute path to your local clone.

## Development

```bash
uv sync --group dev
uv run pytest tests/
```
