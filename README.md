# brahe-mcp

[![Tests](https://github.com/duncaneddy/brahe-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/duncaneddy/brahe-mcp/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/brahe-mcp)](https://pypi.org/project/brahe-mcp/)

This project provides a Model Context Protocol (MCP) server that exposes the astrodynamics capabilities of the [Brahe](https://github.com/duncaneddy/brahe) library enabling language models to get smarter about astrodynamics and space situational awareness.

![demo](demo.gif)

## Installation

```bash
uv tool install brahe-mcp
```

or

```bash
pip install brahe-mcp
```

Then configure your MCP client to use the installed tool:

```json
{
  "mcpServers": {
    "brahe": {
      "command": "brahe-mcp"
    }
  }
}
```

## Local Setup

To run the server from a local clone (useful for development or testing before installing):

```bash
git clone https://github.com/duncaneddy/brahe-mcp.git
cd brahe-mcp
uv sync --group dev
```

Then configure your MCP client to launch the server via `uv run`. Add the following to your MCP settings file:

- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
- **Claude Code**: `.claude/settings.json` (project-level or global)

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

### SpaceTrack Configuration

The SpaceTrack tools require a [Space-Track.org](https://www.space-track.org) account. Add your credentials via the `env` key in the server config:

```json
{
  "mcpServers": {
    "brahe": {
      "command": "brahe-mcp",
      "env": {
        "SPACETRACK_USER": "your@email.com",
        "SPACETRACK_PASS": "your-password"
      }
    }
  }
}
```

> **Note:** Claude Desktop does not expand shell variables like `${SPACETRACK_USER}` — you must put the actual values in the config. Claude Code inherits your shell environment, so you can alternatively set the variables in `~/.zshrc` and omit the `env` block.

Without these variables, the CelesTrak tools will still work normally — only the SpaceTrack tools will return an error prompting you to set the credentials.

## Development

```bash
uv sync --group dev
uv run pytest tests/
```
