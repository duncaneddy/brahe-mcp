# brahe-mcp

<p align="center">
  <a href="https://github.com/duncaneddy/brahe-mcp/actions/workflows/test.yml/badge.svg">
    <img src="https://github.com/duncaneddy/brahe-mcp/actions/workflows/test.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://pypi.org/project/brahe-mcp/">
    <img src="https://img.shields.io/pypi/v/brahe-mcp" alt="PyPI">
  </a>
</p>

This project provides a Model Context Protocol (MCP) server that exposes the astrodynamics capabilities of the [Brahe](https://github.com/duncaneddy/brahe) library enabling language models to get smarter about astrodynamics and space situational awareness.

![demo](demo.gif)

## Capabilities

Recent additions expose more of brahe's 1.7.0 astrodynamics surface as MCP tools:

| Group | Tools | Notes |
| --- | --- | --- |
| Frame transforms | `list_frame_options`, `transform_frame` | Position/state/rotation transforms across GCRF, ITRF, EME2000, GSE, EMR, SER, EMBI, SSBI, lunar (LCI/LFME/LFPA), Mars (MCI/MCMF), the Synodic rotating frame, and generic body frames. |
| SPICE & body ephemerides | `list_ephemeris_options`, `list_spice_kernels`, `load_spice_kernel`, `load_common_spice_kernels`, `unload_spice_kernel`, `get_body_state` | Manage SPICE kernels and query planet/Moon/Sun/barycenter states via SPICE. |
| Small bodies | `list_smallbody_options`, `lookup_small_body`, `get_small_body_ephemeris` | Look up asteroids/comets via the JPL Small-Body Database (SBDB) and sample ephemerides generated on demand via JPL Horizons. Both make live JPL network calls. |
| 3D plots | `plot_trajectory_3d`, `plot_synodic_3d` | Interactive 3D trajectory plots about Earth or in a synodic (rotating two-body) frame; each returns an inline PNG plus a saved interactive HTML file. |
| RA/Dec coordinates | `list_radec_options`, `convert_radec`, `apply_proper_motion` | Right ascension/declination to inertial (ECI/GCRF) and topocentric (AZEL) frames, plus IAU SOFA proper-motion propagation. Proper motion is in mas/yr; `pm_ra` is the cos(dec)-weighted catalog convention. |
| Orbital elements | `convert_equinoctial`, `convert_mean_osculating`, `convert_mean_osculating_batch` | Equinoctial elements (with retrograde factor `fr`) and mean/osculating conversion via Brouwer-Lyddane or numerical windowed averaging. |
| Relative motion | `list_relative_motion_options`, `convert_rtn_state`, `convert_roe_state`, `compute_rtn_rotation` | RTN and quasi-nonsingular ROE conversions between a chief and deputy satellite. |
| Attitude | `list_attitude_options`, `convert_attitude`, `axis_rotation_matrix`, `compose_rotations`, `quaternion_slerp` | Quaternion, Euler axis, Euler angle, and rotation matrix representations, principal-axis rotations, composition, and spherical interpolation. |

### Numerical propagation

`propagate_numerical` supports non-Earth central bodies via the body-specific force model presets (`lunar_default`, `mars_default`, `cislunar_default`), or by setting `force_model="central_body"` with `central_body` set to `moon`, `mars`, or `emb`; the `bci`/`bcbf` output frames report state relative to that body. (`central_body` is only consulted for the `central_body` preset — other presets bake in their own body.) Two optional structured config dicts also replace the previous per-force keyword arguments:

- `force_config`: `{gravity, drag, srp, third_body, tides, relativity, frame_transform}`
- `integrator`: `{preset, method, abs_tol, rel_tol, initial_step, max_step, store_accelerations}`

Call `list_propagation_options()` to discover the valid keys and values for both dicts.

### Mean and osculating elements

`convert_mean_osculating` handles a single state using the Brouwer-Lyddane
analytical theory. `convert_mean_osculating_batch` handles a time series and
additionally supports the numerical windowed-averaging method.

Two things to know about the numerical method:

- It is **batch-only**. The single-state tool rejects it.
- With `edge="truncate"` (the default), osculating-to-mean returns **fewer
  states than it receives**, because the averaging window consumes the edges of
  the series. Read `n_output` and `dropped_by_edge_handling` from the response
  rather than assuming the length is preserved.

Numerical mean-to-osculating inverts the averaging by differential correction
and therefore requires a `force_config`; call `list_propagation_options()` for
the valid keys. Brouwer-Lyddane is a first-order theory, so mean-to-osculating
followed by osculating-to-mean does not return the input exactly.

### Plotting output

Plotting depends on the `brahe[plots]` extra, which is installed automatically as a dependency of `brahe-mcp`. The 3D plot tools also write an interactive HTML file to disk; the directory is configurable via the `BRAHE_MCP_OUTPUT_DIR` environment variable (default `<tempdir>/brahe-mcp-plots`).

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

The MCP configuration location depends on your client. For popular tools you can find it here:

| Client | Config Location |
| --- | --- |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Code | `.claude/settings.json` (project-level or global) |
| Gemini CLI | `~/.gemini/settings.json` |
| OpenAI Codex CLI | `~/.codex/config.toml` (see [below](#openai-codex-cli)) |

> [!NOTE]  
> ChatGPT Desktop does **not** support local stdio MCP servers — it requires remote HTTPS endpoints

### OpenAI Codex CLI

[Codex CLI](https://developers.openai.com/codex/mcp) stores MCP configuration in TOML format at `~/.codex/config.toml` (or project-scoped `.codex/config.toml`):

```toml
[mcp_servers.brahe]
command = "brahe-mcp"
args = []
```

You can also add it via the CLI:

```bash
codex mcp add brahe -- brahe-mcp
```

To include SpaceTrack credentials:

```bash
codex mcp add brahe --env SPACETRACK_USER=your@email.com --env SPACETRACK_PASS=your-password -- brahe-mcp
```

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

> [!NOTE]  
> Claude Desktop does not expand shell variables like `${SPACETRACK_USER}` — you must put the actual values in the config. Claude Code inherits your shell environment, so you can alternatively set the variables in `~/.zshrc` and omit the `env` block.

Without these variables, the CelesTrak tools will still work normally — only the SpaceTrack tools will return an error prompting you to set the credentials.

## Local Setup

To run the server from a local clone (useful for development or testing before installing):

```bash
git clone https://github.com/duncaneddy/brahe-mcp.git
cd brahe-mcp
uv sync --group dev
```

Then configure your MCP client to launch the server via `uv run`. Add the following to your MCP settings file:

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
