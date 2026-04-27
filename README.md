# Discraftium

A Luanti-based research environment for Minecraft-like 3D multi-agent reinforcement learning experiments.

Discraftium is built on top of the open-source Luanti engine and keeps the engine's modding/runtime foundation while focusing the repository presentation around agent environments, embodied tasks, and reproducible experimentation.

## Project Focus

| Area | What it covers |
|---|---|
| 3D embodied environments | Voxel worlds, navigation, interaction, and task setup using the Luanti engine foundation |
| Multi-agent research | Experiments where multiple agents can act, coordinate, compete, or share a world state |
| Mod-driven tasks | Environment behavior can be extended through Luanti's game/mod architecture |
| Reproducible engine base | C++ engine code, Lua game scripts, assets, and documented build paths are kept in-repo |

## Repository Layout

| Path | Purpose |
|---|---|
| `src/` | Luanti engine source |
| `games/` | Game definitions and test environments |
| `builtin/` | Built-in Lua runtime scripts |
| `client/` | Client-side engine assets and shaders |
| `doc/` | Upstream engine and development documentation |
| `CMakeLists.txt` | Main CMake build entry point |

## Build

This repository follows the Luanti build system. Start with the platform-specific compiler docs:

- [Common build notes](doc/compiling/README.md)
- [Linux build notes](doc/compiling/linux.md)
- [Windows build notes](doc/compiling/windows.md)
- [macOS build notes](doc/compiling/macos.md)

Typical CMake flow:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Upstream Foundation

Discraftium is based on [Luanti](https://www.luanti.org/), formerly Minetest, a free open-source voxel game engine with modding support.

Useful upstream references:

- [Luanti website](https://www.luanti.org/)
- [Luanti GitHub](https://github.com/luanti-org/luanti/)
- [Developer documentation](doc/developing/)
- [Lua API](doc/lua_api.md)

## License

The engine foundation follows Luanti's licensing. See the license files in this repository for details.