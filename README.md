# SX Tools 2 for Blender
NOTE: Please visit the [Documentation Site](https://secretexit.notion.site/SX-Tools-2-for-Blender-Documentation-1681c68851fb4d018d1f9ec762e5aec9)

## Comparison

SX Tools | SX Tools 2
---------|-------------
Static layer stack | Fully dynamic layerstack
CPU viewport compositing | GPU compositing
Vertex color exporting | Vertex color and palette texture atlas exporting
Transparency only as an export category | Transparency as a per-object viewport feature



## For Game Developers
SX Tools 2 is a lightweight content pipeline for vertex-colored assets that drive PBR (Physically Based Rendering) materials in a game. This tool ships with baseline shader networks for Unreal and Unity.

## For 3D Artists
SX Tools is a multi-layer vertex coloring toolbox, referencing a workflow from common 2D image editing programs.

## Highlights
Game artists can now work with vertex colors as layers with opacities and blend modes. Regular paint operations like gradients, color fills and even occlusion baking can be performed directly to vertex color layers. SX Tools 2 comes with a dynamic custom material that displays all edits in the viewport in realtime.

The artist can therefore apply per-vertex occlusion/metallic/smoothness/transmission/emission directly on the model, and tweak the results interactively.

### Features
- Multi-layer vertex color editing
- Layer blend modes (alpha, additive, multiply, overlay)
- Color filler with optional noise
- Color gradient tool with optional noise
- Curvature-based shading
- Vertex ambient occlusion baking
- Mesh thickness baking
- Luminance-based tone-mapping
- Master Palette Tool for quickly replacing the color scheme of object/objects
- PBR Material Library
- Quick-crease tools
- Modifier automations
- Material channel editing (subsurface scattering, occlusion, metallic, roughness)
- FBX-exporting multi-layer vertex-colors to game engines via UV channels to drive the standard PBR material

## Installation:
- Download the zip file, uncompress it, then install and enable sxtools2.py through the Blender add-on interface
- Point the library folder in SX Tools 2 prefs to the unzipped folder
- Pick your asset library .json file if you also use [SX Batcher](https://github.com/FrandSX/sxbatcher-blender)
- SX Tools 2 will appear as a tab in the top right corner of the 3D view

## Getting Started:
Now would be a good time to visit the [Documentation Site](https://secretexit.notion.site/SX-Tools-2-for-Blender-Documentation-1681c68851fb4d018d1f9ec762e5aec9)

## Exporting to Game Engines
The basic flow for using SX Tools in game development is as follows:
1) Model your assets as low-poly control cages for subdivision
2) Define the categories needed for your objects in your game
3) Assign objects to categories
4) Color the objects according to category-specific layer definitions
5) Batch-process the objects for export, leave as much work for automation as possible. Take a look at the existing batch functions and adapt to your project needs.
6) Export to your game engine as FBX
7) Run the assets using a the provided custom material (or make your own)

The package contains simple example shader graphs for Unreal4, and both HDRP and URP renderers in Unity. Dynamic palette changes in the game engine are not facilitated by the example shader graphs, but they do fully support driving PBR material channels with UV data.

Currently the export channels are set in the following way:

Channel | Function
---------|-------------
UV0 | Reserved for a regular texturing
UV§ | Reserved for lightmaps
U2 | Layer coverage masks for dynamic palettes
V2 | Ambient Occlusion
U3 | Transmission or SSS
V3 | Emission
U4 | Metallic
V4 | Smoothness
U5 | Alpha Overlay 1, an alpha mask channel
V5 | Alpha Overlay 2, an alpha mask channel
UV6/UV7 | RGBA Overlay, an optional additional color layer 
UV7 | Currently not in use

Vertex colors are exported from the Composite/VertexColor0 layer. Material properties are assigned according to the list above.
Note that emission is exported only as an 8-bit mask for base vertex color, not a full RGBA-channel.

(c) 2017-2023 Jani Kahrama / Secret Exit Ltd.


# Acknowledgments

Thanks to:

Rory Driscoll for tips on better sampling

Serge Scherbakov for tips on working with iterators
 
Jason Summers for sRGB conversion formulas


SX Tools 2 builds on the following work:

### Vertex-to-edge curvature calculation method 
Algorithm by Stepan Jirka

http://www.stepanjirka.com/maya-api-curvature-shader/

Integrated into SX Tools 2 under MIT license with permission from the author.

### V-HACD For Blender 2.80
Add-on by Alain Ducharme (original author), Andrew Palmer (Blender 2.80)

https://github.com/andyp123/blender_vhacd

SX Tools 2 detects the presence of the V-HACD add-on and utilizes it for mesh collider generation.

