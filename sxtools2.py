bl_info = {
    'name': 'SX Tools 2',
    'author': 'Jani Kahrama / Secret Exit Ltd.',
    'version': (2, 9, 5),
    'blender': (4, 2, 0),
    'location': 'View3D',
    'description': 'Multi-layer vertex coloring tool',
    'doc_url': 'https://secretexit.notion.site/SX-Tools-2-for-Blender-Documentation-1681c68851fb4d018d1f9ec762e5aec9',
    'tracker_url': 'https://github.com/FrandSX/sxtools2-blender/issues',
    'category': 'Development',
}


import bpy
import time
import random
import math
import bmesh
import json
import pathlib
import statistics
import sys
import os
from bpy.app.handlers import persistent
from collections import Counter, defaultdict
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from contextlib import redirect_stdout


# ------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------
class SXTOOLS2_sxglobals(object):
    def __init__(self):
        self.benchmark_modifiers = False
        self.benchmark_tool = False
        self.benchmark_ao = False
        self.benchmark_magic = False
        self.benchmark_cvx = False

        self.libraries_status = False
        self.refresh_in_progress = False
        self.layer_update_in_progress = False
        self.magic_in_progress = False
        self.export_in_progress = False
        self.hsl_update = False
        self.mat_update = False
        self.curvature_update = False
        self.copy_buffer = {}
        self.prev_mode = 'OBJECT'
        self.mode = None
        self.mode_id = None
        self.randomseed = 42

        self.selection_modal_status = False
        self.key_modal_status = False

        self.prev_selection = []
        self.prev_component_selection = []
        self.ramp_dict = {}
        self.category_dict = {}
        self.preset_lookup = {}
        self.master_palette_list = []
        self.material_list = []

        self.default_colors = {
            'COLOR': (0.0, 0.0, 0.0, 0.0),
            'OCC': (1.0, 1.0, 1.0, 1.0),
            'MET': (0.0, 0.0, 0.0, 0.0),
            'RGH': (0.5, 0.5, 0.5, 1.0),
            'TRN': (0.0, 0.0, 0.0, 0.0),
            'SSS': (0.0, 0.0, 0.0, 0.0),
            'EMI': (0.0, 0.0, 0.0, 0.0),
            'CMP': (0.0, 0.0, 0.0, 0.0),
            'CID': (0.0, 0.0, 0.0, 0.0),
            }

        # {layer_keys: (index, obj)}
        self.sx2material_dict = {}

        # {obj.name: {stack_layer_index: (layer.name, layer.color_attribute, layer.layer_type)}}
        self.layer_stack_dict = {}

        # Keywords used by Smart Separate. Avoid using these in regular object names
        self.keywords = ['_org', '_LOD0', '_top', '_bottom', '_front', '_rear', '_left', '_right']

        self.version, self.minorversion, _ = bpy.app.version

    def __del__(self):
        print('SX Tools: Exiting sxglobals')


# ------------------------------------------------------------------------
#    File IO
# ------------------------------------------------------------------------
class SXTOOLS2_files(object):
    def __init__(self):
        return None


    def __del__(self):
        print('SX Tools: Exiting files')


    # Loads palettes.json and materials.json
    def load_file(self, mode):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        directory = prefs.libraryfolder
        file_path = directory + mode + '.json'

        if len(directory) > 0:
            try:
                with open(file_path, 'r') as input:
                    temp_dict = {}
                    temp_dict = json.load(input)
                    if mode == 'palettes':
                        del sxglobals.master_palette_list[:]
                        bpy.context.scene.sx2palettes.clear()
                        sxglobals.master_palette_list = temp_dict['Palettes']
                    elif mode == 'materials':
                        del sxglobals.material_list[:]
                        bpy.context.scene.sx2materials.clear()
                        sxglobals.material_list = temp_dict['Materials']
                    elif mode == 'gradients':
                        sxglobals.ramp_dict.clear()
                        sxglobals.ramp_dict = temp_dict
                    elif mode == 'categories':
                        sxglobals.category_dict.clear()
                        sxglobals.category_dict = temp_dict

                    input.close()
                print(f'SX Tools: {mode} loaded from {file_path}')
            except ValueError:
                print(f'SX Tools Error: Invalid {mode} file.')
                prefs.libraryfolder = ''
                return False
            except IOError:
                print(f'SX Tools Error: {mode} file not found!')
                return False
        else:
            print(f'SX Tools: No {mode} file found - set Library Folder location in SX Tools 2 Preferences')
            return False

        if mode == 'palettes':
            self.load_swatches(sxglobals.master_palette_list)
        elif mode == 'materials':
            self.load_swatches(sxglobals.material_list)
        return True


    def save_file(self, mode):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        directory = prefs.libraryfolder
        file_path = directory + mode + '.json'
        # Palettes.json Materials.json

        if len(directory) > 0:
            with open(file_path, 'w') as output:
                if mode == 'palettes':
                    temp_dict = {}
                    temp_dict['Palettes'] = sxglobals.master_palette_list
                    json.dump(temp_dict, output, indent=4)
                elif mode == 'materials':
                    temp_dict = {}
                    temp_dict['Materials'] = sxglobals.material_list
                    json.dump(temp_dict, output, indent=4)
                elif mode == 'gradients':
                    temp_dict = {}
                    temp_dict = sxglobals.ramp_dict
                    json.dump(temp_dict, output, indent=4)
                output.close()
            message_box(mode + ' saved')
            # print('SX Tools: ' + mode + ' saved')
        else:
            message_box(mode + ' file location not set!', 'SX Tools Error', 'ERROR')
            # print('SX Tools Warning: ' + mode + ' file location not set!')


    def load_swatches(self, swatch_list):
        if swatch_list == sxglobals.material_list:
            swatch_count = 3
            sxlist = bpy.context.scene.sx2materials
        elif swatch_list == sxglobals.master_palette_list:
            swatch_count = 5
            sxlist = bpy.context.scene.sx2palettes

        for category_dict in swatch_list:
            for category in category_dict:
                if len(category_dict[category]) == 0:
                    item = sxlist.add()
                    item.name = 'Empty'
                    item.category = category
                    for i in range(swatch_count):
                        incolor = [0.0, 0.0, 0.0, 1.0]
                        setattr(item, 'color'+str(i), incolor[:])
                else:
                    for entry in category_dict[category]:
                        item = sxlist.add()
                        item.name = entry
                        item.category = category
                        for i in range(swatch_count):
                            incolor = category_dict[category][entry][i][:3] + [1.0]
                            if (swatch_count == 5):
                                setattr(item, 'color'+str(i), convert.srgb_to_linear(incolor))
                            else:
                                setattr(item, 'color'+str(i), incolor)


    def save_ramp(self, ramp_name):
        ramp_dict = utils.get_ramp()
        sxglobals.ramp_dict[ramp_name] = ramp_dict

        self.save_file('gradients')


    # In paletted export mode, gradients and overlays are
    # not composited as that will be
    # done by the shader on the game engine side
    def export_files(self, groups):
        scene = bpy.context.scene.sx2
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        export_dict = {'LIN': 'LINEAR', 'SRGB': 'SRGB'}
        empty = True

        groupNames = []
        for group in groups:
            bpy.context.view_layer.objects.active = group
            bpy.ops.object.select_all(action='DESELECT')
            group.select_set(True)
            org_loc = group.location.copy()
            group.location = (0, 0, 0)

            # Check for mesh colliders
            collider_list = []
            if 'SXColliders' in bpy.data.collections:
                colliders = bpy.data.collections['SXColliders'].objects
                # if group.name.endswith('_root'):
                #     collider_id = group.name[:-5]
                # else:
                #     collider_id = group.name

                for collider in colliders:
                    if collider.get('group_id') == group.name:
                        collider_list.append(collider)
                        collider['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))
                    # if collider.name.startswith(collider_id):
                    #     collider_list.append(collider)
                    #     collider['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))

                child_list = utils.find_children(group, recursive=True)
                sel_list = [child for child in child_list if child.name not in colliders.keys()]
            else:
                sel_list = utils.find_children(group, recursive=True)

            for sel in sel_list:
                sel.select_set(True)
            group.select_set(False)

            # Only groups with meshes and armatures as children are exported
            obj_list = []
            for sel in sel_list:
                if sel.type == 'MESH':
                    bg_layer = utils.find_color_layers(sel, 0)
                    obj_list.append(sel)
                    sel['staticVertexColors'] = sel.sx2.staticvertexcolors
                    sel['transparent'] = '0' if bg_layer.opacity == 1.0 else '1'
                    sel['specular'] = sel.sx2.mat_specular
                    sel['anisotropic'] = sel.sx2.mat_anisotropic
                    sel['clearcoat'] = sel.sx2.mat_clearcoat
                    sel['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))
                    sel['colorSpace'] = export_dict[prefs.exportspace]

            if obj_list:
                empty = False

                # Create palette masks
                export.reset_uv_layers(obj_list)
                export.generate_palette_masks(obj_list)
                export.generate_uv_channels(obj_list)
                export.composite_color_layers(obj_list)

                for obj in obj_list:
                    bpy.context.view_layer.objects.active = obj
                    obj.data.attributes.active_color = obj.data.color_attributes['Composite']
                    bpy.ops.geometry.color_attribute_render_set(name='Composite')

                    # bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                    if 'sxTiler' in obj.modifiers:
                        obj.modifiers.remove(obj.modifiers['sxTiler'])

                bpy.context.view_layer.objects.active = group

                category_data = sxglobals.category_dict[sxglobals.preset_lookup[obj_list[0].sx2.category]]
                export_subfolder = category_data["export_subfolder"]
                subfolder = None
                if export_subfolder != "":
                    subfolder = export_subfolder.replace('//', os.path.sep)

                category = obj_list[0].sx2.category.lower()
                print(f'Determining path: {group.name} {category}')
                if subfolder is None:
                    path = scene.exportfolder + category + os.path.sep
                else:
                    path = scene.exportfolder + subfolder + os.path.sep + category + os.path.sep
                pathlib.Path(path).mkdir(exist_ok=True, parents=True)

                if collider_list:
                    for collider in collider_list:
                        collider.select_set(True)

                if (prefs.exportformat == 'FBX'):
                    export_path = path + group.name + '.' + 'fbx'
                    export_settings = ['FBX_SCALE_UNITS', False, False, False, 'Z', '-Y', '-Y', '-X', export_dict[prefs.exportspace]]

                    with redirect_stdout(open(os.devnull, 'w')):
                        bpy.ops.export_scene.fbx(
                            filepath=export_path,
                            apply_scale_options=export_settings[0],
                            use_selection=True,
                            apply_unit_scale=export_settings[1],
                            use_space_transform=export_settings[2],
                            bake_space_transform=export_settings[3],
                            use_mesh_modifiers=True,
                            mesh_smooth_type='OFF',
                            prioritize_active_color=True,
                            axis_up=export_settings[4],
                            axis_forward=export_settings[5],
                            use_active_collection=False,
                            add_leaf_bones=False,
                            primary_bone_axis=export_settings[6],
                            secondary_bone_axis=export_settings[7],
                            object_types={'ARMATURE', 'EMPTY', 'MESH'},
                            use_custom_props=True,
                            use_metadata=False,
                            colors_type=export_settings[8],
                            bake_anim=False)

                elif (prefs.exportformat == 'GLTF'):
                    export_path = path + group.name + '.' + 'glb'
                    with redirect_stdout(open(os.devnull, 'w')):
                        bpy.ops.export_scene.gltf(
                            filepath=export_path,
                            check_existing=False,
                            export_format='GLB',
                            export_apply=True,
                            export_texcoords=True,
                            export_normals=True,
                            export_materials='NONE',
                            export_colors=True,
                            export_cameras=False,
                            use_selection=True,
                            export_extras=True,
                            export_yup=True,
                            export_animations=False,
                            export_rest_position_armature=True,
                            export_lights=False)

                if not bpy.app.background:
                    groupNames.append(group.name)
                    group.location = org_loc

                    modifiers.remove_modifiers(obj_list)
                    modifiers.add_modifiers(obj_list)

                    bpy.context.view_layer.objects.active = group

        if empty:
            message_box('No objects exported!')
        else:
            message_box('Exported:\n' + str('\n').join(groupNames))
            for group_name in groupNames:
                print(f'Completed: {group_name}')


    def export_files_simple(self, objs, reset_locations=False):
        scene = bpy.context.scene.sx2
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        export_dict = {'LIN': 'LINEAR', 'SRGB': 'SRGB'}

        obj_names = []
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            org_loc = obj.location.copy()
            if reset_locations:
                obj.location = (0, 0, 0)

            bg_layer = utils.find_color_layers(obj, 0)
            obj['transparent'] = '0' if bg_layer.opacity == 1.0 else '1'
            obj['specular'] = obj.sx2.mat_specular
            obj['anisotropic'] = obj.sx2.mat_anisotropic
            obj['clearcoat'] = obj.sx2.mat_clearcoat
            obj['staticVertexColors'] = obj.sx2.staticvertexcolors
            obj['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))
            obj['colorSpace'] = export_dict[prefs.exportspace]

            # bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            if 'sxTiler' in obj.modifiers:
                obj.modifiers.remove(obj.modifiers['sxTiler'])

            category = objs[0].sx2.category.lower()
            print(f'Determining path: {objs[0].name} {category}')
            path = scene.exportfolder + category + os.path.sep
            pathlib.Path(path).mkdir(exist_ok=True)

            export_path = path + obj.name + '.' + 'fbx'
            export_settings = ['FBX_SCALE_UNITS', False, False, False, 'Z', '-Y', '-Y', '-X', 'NONE']

            with redirect_stdout(open(os.devnull, 'w')):
                bpy.ops.export_scene.fbx(
                    filepath=export_path,
                    apply_scale_options=export_settings[0],
                    use_selection=True,
                    apply_unit_scale=export_settings[1],
                    use_space_transform=export_settings[2],
                    bake_space_transform=export_settings[3],
                    use_mesh_modifiers=True,
                    axis_up=export_settings[4],
                    axis_forward=export_settings[5],
                    use_active_collection=False,
                    add_leaf_bones=False,
                    primary_bone_axis=export_settings[6],
                    secondary_bone_axis=export_settings[7],
                    object_types={'ARMATURE', 'EMPTY', 'MESH'},
                    use_custom_props=True,
                    use_metadata=False,
                    colors_type=export_settings[8])

            obj_names.append(obj.name)
            obj.location = org_loc

        modifiers.remove_modifiers(objs)
        modifiers.add_modifiers(objs)

        message_box('Exported:\n' + str('\n').join(obj_names))
        for obj_name in obj_names:
            print(f'Completed: {obj_name}')


    def export_palette_atlases(self, palette_dict):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        for palette_name, palette in palette_dict.items():
            swatch_size = 16
            grid = 1 if len(palette) == 0 else 2**math.ceil(math.log2(math.sqrt(len(palette))))
            pixels = [0.0, 0.0, 0.0, 0.1] * grid * grid * swatch_size * swatch_size

            z = 0
            for y in range(grid):
                row = [0.0, 0.0, 0.0, 1.0] * grid * swatch_size
                row_length = 4 * grid * swatch_size
                for x in range(grid):
                    color = palette[z] if z < len(palette) else (0.0, 0.0, 0.0, 1.0)
                    if (palette_name == 'Composite') and (prefs.exportspace == 'SRGB'):
                        color = convert.linear_to_srgb(color)
                    elif (palette_name == 'Roughness') and (prefs.exportroughness == 'SMOOTH'):
                        color = [1.0 - color[0], 1.0 - color[1], 1.0 - color[2], color[3]]
                    z += 1
                    for i in range(swatch_size):
                        row[x*swatch_size*4+i*4:x*swatch_size*4+i*4+4] = [color[0], color[1], color[2], color[3]]
                for j in range(swatch_size):
                    pixels[y*swatch_size*row_length+j*row_length:y*swatch_size*row_length+(j+1)*row_length] = row

            path = bpy.context.scene.sx2.exportfolder + os.path.sep
            pathlib.Path(path).mkdir(exist_ok=True)

            image = bpy.data.images.new('palette_atlas', alpha=True, width=grid*swatch_size, height=grid*swatch_size)
            image.alpha_mode = 'STRAIGHT'
            image.pixels = pixels
            image.filepath_raw = path + 'atlas_' + palette_name.lower() + '.png'
            image.file_format = 'PNG'
            image.save()


# ------------------------------------------------------------------------
#    Useful Miscellaneous Functions
# ------------------------------------------------------------------------
class SXTOOLS2_utils(object):
    def __init__(self):
        return None


    def mode_manager(self, objs, set_mode=False, mode_id=None):
        if set_mode:
            if sxglobals.mode_id is None:
                sxglobals.mode_id = mode_id
                sxglobals.mode = objs[0].mode
                if objs[0].mode != 'OBJECT':
                    if bpy.context.view_layer.objects.active is None:
                        bpy.context.view_layer.objects.active = objs[0]
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
            else:
                if objs[0].mode != 'OBJECT':
                    if bpy.context.view_layer.objects.active is None:
                        bpy.context.view_layer.objects.active = objs[0]
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        else:
            if sxglobals.mode_id == mode_id:
                if bpy.context.view_layer.objects.active is None:
                    bpy.context.view_layer.objects.active = objs[0]
                bpy.ops.object.mode_set(mode=sxglobals.mode)
                sxglobals.mode_id = None


    def clear_component_selection(self, objs):
        for obj in objs:
            mesh = obj.data
            mesh.vertices.foreach_set('select', [False] * len(mesh.vertices))
            mesh.edges.foreach_set('select', [False] * len(mesh.edges))
            mesh.polygons.foreach_set('select', [False] * len(mesh.polygons))
            mesh.update()


    def create_collection(self, collection_name):
        return bpy.data.collections[collection_name] if collection_name in bpy.data.collections else bpy.data.collections.new(collection_name)


    def find_sx2material_name(self, obj):
        mat_id = self.get_mat_id(obj)
        if mat_id in sxglobals.sx2material_dict:
            return 'SX2Material_' + sxglobals.sx2material_dict[mat_id][1]
        else:
            return None


    # SX2Materials are generated according to differentiators combined in per-object mat_ids
    def get_mat_id(self, obj):
        sel_layer = obj.sx2.selectedlayer
        mat_opaque = True if (self.find_color_layers(obj, 0) is not None) and (self.find_color_layers(obj, 0).opacity == 1.0) else False
        mat_culling = obj.sx2.backfaceculling
        mat_specular = obj.sx2.mat_specular
        mat_anisotropic = obj.sx2.mat_anisotropic
        mat_clearcoat = obj.sx2.mat_clearcoat
        layer_keys = tuple(obj.sx2layers.keys())
        layer_attributes = tuple([layer.color_attribute for layer in obj.sx2layers])
        layer_vis = tuple([layer.visibility for layer in obj.sx2layers])
        layer_paletted = tuple([layer.paletted for layer in obj.sx2layers])
        layer_palette_ids = tuple([layer.palette_index for layer in obj.sx2layers])
        return (obj.sx2.shadingmode, sel_layer, mat_opaque, mat_culling, mat_specular, mat_anisotropic, mat_clearcoat, layer_keys, layer_attributes, layer_vis, layer_paletted, layer_palette_ids)


    def find_layer_by_stack_index(self, obj, index):
        if obj.sx2layers:
            stack_dict = sxglobals.layer_stack_dict.get(obj.name, {})
            if len(stack_dict) == 0:
                layers = [(layer.index, layer.color_attribute) for layer in obj.sx2layers]
                layers.sort(key=lambda y: y[0])

                for i, layer_tuple in enumerate(layers):
                    for layer in obj.sx2layers:
                        if layer.color_attribute == layer_tuple[1]:
                            stack_dict[i] = (layer.name, layer.color_attribute, layer.layer_type)

                sxglobals.layer_stack_dict[obj.name] = stack_dict

            layer_data = stack_dict.get(index, None)
            return obj.sx2layers[layer_data[0]] if layer_data is not None else None
        else:
            return None


    def find_layer_index_by_name(self, obj, name):
        for i, layer in enumerate(obj.sx2layers):
            if layer.name == name:
                return i


    # obj_sel_override allows object-level mask evaluation even if the object is in edit mode
    def find_colors_by_frequency(self, objs, layer_name, numcolors=None, masklayer=None, maskcolor=None, obj_sel_override=False, alphavalues=False):
        color_list = []

        for obj in objs:
            if obj_sel_override:
                values = layers.get_layer(obj, obj.sx2layers[layer_name], as_tuple=True)
            else:
                values = generate.mask_list(obj, layers.get_layer(obj, obj.sx2layers[layer_name]), masklayer=masklayer, maskcolor=maskcolor, as_tuple=True)

            if values is not None:
                color_list += values

        if alphavalues:
            colors = [convert.luminance_to_color(color[3]) for color in color_list if color[3] > 0.0]
        else:
            colors = [color for color in color_list if color[3] > 0.0]

        quantized_colors = [(self.round_stepped(color[0]), self.round_stepped(color[1]), self.round_stepped(color[2]), 1.0) for color in colors]
        sort_list = [color for color, count in Counter(quantized_colors).most_common(numcolors)]

        if numcolors is not None:
            sort_colors = sort_list[:numcolors]
            sort_colors += [[0.0, 0.0, 0.0, 1.0]] * (numcolors - len(sort_colors))
            return sort_colors
        else:
            return sort_list


    # The index parameter returns the nth color layer in the stack
    def find_color_layers(self, obj, index=None, staticvertexcolors=True):
        if obj.sx2layers:
            color_layers = [layer for layer in obj.sx2layers if (layer.layer_type == 'COLOR') and (layer.name != 'Flattened')]

            if not staticvertexcolors:
                filtered = [layer for layer in color_layers if ('Gradient' not in layer.name and 'Overlay' not in layer.name)]
                color_layers = filtered

            color_layers.sort(key=lambda x: x.index)

            index_layer = None
            if index is not None:
                if index <= (len(color_layers) - 1):
                    index_layer = color_layers[index]

            return color_layers if index is None else index_layer
        else:
            return None


    def find_root_pivot(self, objs):
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_object_bounding_box(objs)
        pivot = ((xmax + xmin)*0.5, (ymax + ymin)*0.5, zmin)

        return pivot


    # ARMATUREs check for grandparent
    def find_children(self, group, objs=None, recursive=False, child_type=None):

        def get_children(parent):
            children = []
            for child in objs:
                if child.parent == parent:
                    children.append(child)
                elif (child.parent is not None) and (child.parent.type == 'ARMATURE') and (child.parent.parent == parent):
                    children.append(child)
            return children


        def child_recurse(children):
            for child in children:
                child_list = get_children(child)
                if child_list:
                    results.extend(child_list)
                    child_recurse(child_list)

        results = []
        if objs is None:
            objs = bpy.data.objects.values()

        if recursive:
            child_list = [group, ]
            child_recurse(child_list)
        else:
            results = get_children(group)

        if child_type is not None:
            results = [child for child in results if child.type == child_type]

        return results
                        

    # Finds groups to be exported,
    # only EMPTY objects with no parents
    # treat ARMATURE objects as a special case
    def find_groups(self, objs=None, exportready=False):
        groups = []
        if objs is None:
            objs = bpy.context.view_layer.objects

        for obj in objs:
            if (obj.type == 'EMPTY') and (obj.parent is None):
                groups.append(obj)

            parent = obj.parent
            if (parent is not None) and (parent.type == 'EMPTY') and (parent.parent is None):
                groups.append(obj.parent)
            elif (parent is not None) and (parent.type == 'ARMATURE') and (parent.parent is not None) and (parent.parent.type == 'EMPTY'):
                groups.append(parent.parent)

        if exportready:
            groups = [group for group in groups if group.sx2.exportready]

        return list(set(groups))


    def get_selection_bounding_box(self, objs):
        vert_pos_list = []
        for obj in objs:
            mesh = obj.data
            mat = obj.matrix_world
            for vert in mesh.vertices:
                if vert.select:
                    vert_pos_list.append(mat @ vert.co)

        bbx = [[None, None], [None, None], [None, None]]
        for i, fvPos in enumerate(vert_pos_list):
            # first vert
            if i == 0:
                bbx[0][0] = bbx[0][1] = fvPos[0]
                bbx[1][0] = bbx[1][1] = fvPos[1]
                bbx[2][0] = bbx[2][1] = fvPos[2]
            else:
                for j in range(3):
                    if fvPos[j] < bbx[j][0]:
                        bbx[j][0] = fvPos[j]
                    elif fvPos[j] > bbx[j][1]:
                        bbx[j][1] = fvPos[j]

        return bbx[0][0], bbx[0][1], bbx[1][0], bbx[1][1], bbx[2][0], bbx[2][1]


    def get_object_bounding_box(self, objs, mode='world'):
        bbx_x = []
        bbx_y = []
        bbx_z = []
        for obj in objs:
            edg = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(edg)
            if (mode == 'local'):
                corners = [Vector(corner) for corner in obj_eval.bound_box]
            elif (mode == 'parent'):
                corners = [obj.matrix_local @ Vector(corner) for corner in obj_eval.bound_box]
            elif (mode == 'world'):
                corners = [obj.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]

            for corner in corners:
                bbx_x.append(corner[0])
                bbx_y.append(corner[1])
                bbx_z.append(corner[2])
        xmin, xmax = min(bbx_x), max(bbx_x)
        ymin, ymax = min(bbx_y), max(bbx_y)
        zmin, zmax = min(bbx_z), max(bbx_z)

        return xmin, xmax, ymin, ymax, zmin, zmax


    def bounding_box_overlap_test(self, obj1, obj2):
        bbx1 = self.get_object_bounding_box([obj1, ])
        bbx2 = self.get_object_bounding_box([obj2, ])
        min1, max1 = (bbx1[0], bbx1[2], bbx1[4]), (bbx1[1], bbx1[3], bbx1[5])
        min2, max2 = (bbx2[0], bbx2[2], bbx2[4]), (bbx2[1], bbx2[3], bbx2[5])
        return (min1.x <= max2.x and max1.x >= min2.x and
                min1.y <= max2.y and max1.y >= min2.y and
                min1.z <= max2.z and max1.z >= min2.z)


    # The default index of any added layer is 100
    def insert_layer_at_index(self, obj, inserted_layer, index):
        inserted_layer.index = 100
        if len(obj.sx2layers) == 1:
            index = 0

        sort_layers = self.sort_stack_indices(obj)
        del sort_layers[-1]
        sort_layers.insert(index, inserted_layer)

        self.sort_stack_indices(obj, sort_layers=sort_layers)
        return index


    def sort_stack_indices(self, obj, sort_layers=None):
        stack_dict = sxglobals.layer_stack_dict.get(obj.name, {})
        if stack_dict:
            sxglobals.layer_stack_dict[obj.name].clear()

        if obj.sx2layers:
            if sort_layers is None:
                sort_layers = obj.sx2layers[:]
                sort_layers.sort(key=lambda x: x.index)

            for i, sort_layer in enumerate(sort_layers):
                obj.sx2layers[sort_layer.name].index = i
                stack_dict[i] = (sort_layer.name, sort_layer.color_attribute, sort_layer.layer_type)

            sxglobals.layer_stack_dict[obj.name] = stack_dict

            return sort_layers
        else:
            return None


    def color_compare(self, color1, color2, tolerance=0.001):
        return (Vector(color1) - Vector(color2)).length <= tolerance


    def round_stepped(self, x, step=0.005):
        return round(round(x / step) * step, 3)


    # if vertex pos x y z is at bbx limit, and mirror axis is set, round vertex position
    def round_tiling_verts(self, objs):
        for obj in objs:
            if obj.sx2.tiling:
                vert_dict = generate.vertex_data_dict(obj)
                if vert_dict:
                    xmin, xmax, ymin, ymax, zmin, zmax = self.get_object_bounding_box([obj, ], mode='local')
                    for vert_id in vert_dict:
                        vertLoc = Vector(vert_dict[vert_id][0])

                        if (obj.sx2.tile_neg_x and (round(vertLoc[0], 2) == round(xmin, 2))) or (obj.sx2.tile_pos_x and (round(vertLoc[0], 2) == round(xmax, 2))):
                            vertLoc[0] = round(vertLoc[0], 2)
                        if (obj.sx2.tile_neg_y and (round(vertLoc[1], 2) == round(ymin, 2))) or (obj.sx2.tile_pos_y and (round(vertLoc[1], 2) == round(ymax, 2))):
                            vertLoc[1] = round(vertLoc[1], 2)
                        if (obj.sx2.tile_neg_z and (round(vertLoc[2], 2) == round(zmin, 2))) or (obj.sx2.tile_pos_z and (round(vertLoc[2], 2) == round(zmax, 2))):
                            vertLoc[2] = round(vertLoc[2], 2)

                        if vertLoc != Vector(vert_dict[vert_id][0]):
                            obj.data.vertices[vert_id].co = vertLoc
                            obj.data.vertices[vert_id].select = True

                    obj.data.update()


    def clear_parent_inverse_matrix(self, objs):
        active = bpy.context.view_layer.objects.active
        mtx_dict = {}
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            mtx_dict[obj] = obj.matrix_world.copy()

        task_list = objs[:]
        while len(task_list) > 0:
            for obj in objs:
                if (obj.parent not in task_list) and (obj in task_list):
                    bpy.context.view_layer.objects.active = obj
                    obj.matrix_parent_inverse.identity()
                    obj.matrix_world = mtx_dict[obj]
                    task_list.remove(obj)
                    if not task_list:
                        break

        bpy.context.view_layer.objects.active = active


    def get_ramp(self):
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp'].color_ramp
        ramp_dict = {
            'mode': ramp.color_mode,
            'interpolation': ramp.interpolation,
            'hue_interpolation': ramp.hue_interpolation,
            'elements': [[element.position, [element.color[0], element.color[1], element.color[2], element.color[3]], None] for element in ramp.elements]
        }
        return ramp_dict


    # uses raycheck to calculate safe distance for shrinking mesh colliders
    def find_safe_mesh_offset(self, obj):
        bias = 0.005

        mesh = obj.data
        safe_distance = None

        for vert in mesh.vertices:
            vert_loc = vert.co
            inv_normal = -vert.normal.normalized()
            bias_vec = inv_normal * bias
            ray_origin = vert_loc + bias_vec

            hit, loc, normal, index = obj.ray_cast(ray_origin, inv_normal)

            if hit:
                dist = (loc - vert_loc).length
                if (safe_distance is None) or (dist < safe_distance):
                    safe_distance = dist

        return safe_distance if safe_distance is not None else 0


    def __del__(self):
        print('SX Tools: Exiting utils')


# ------------------------------------------------------------------------
#    Color Conversions
# ------------------------------------------------------------------------
class SXTOOLS2_convert(object):
    def __init__(self):
        return None


    def color_to_luminance(self, in_rgba, premul=True):
        lumR = 0.212655
        lumG = 0.715158
        lumB = 0.072187
        alpha = in_rgba[3]

        linLum = lumR * in_rgba[0] + lumG * in_rgba[1] + lumB * in_rgba[2]
        # luminance = convert.linear_to_srgb((linLum, linLum, linLum, alpha))[0]
        if premul:
            return linLum * alpha  # luminance * alpha
        else:
            return linLum


    def luminance_to_color(self, value):
        return (value, value, value, 1.0)


    def luminance_to_alpha(self, value):
        return (1.0, 1.0, 1.0, value)


    def srgb_to_linear(self, in_rgba):
        out_rgba = []
        for i in range(3):
            if in_rgba[i] < 0.0:
                out_rgba.append(0.0)
            elif 0.0 <= in_rgba[i] <= 0.0404482362771082:
                out_rgba.append(float(in_rgba[i]) / 12.92)
            elif 0.0404482362771082 < in_rgba[i] <= 1.0:
                out_rgba.append(((in_rgba[i] + 0.055) / 1.055) ** 2.4)
            elif in_rgba[i] > 1.0:
                out_rgba.append(1.0)
        out_rgba.append(in_rgba[3])
        return out_rgba


    def linear_to_srgb(self, in_rgba):
        out_rgba = []
        for i in range(3):
            if in_rgba[i] < 0.0:
                out_rgba.append(0.0)
            elif 0.0 <= in_rgba[i] <= 0.00313066844250063:
                out_rgba.append(float(in_rgba[i]) * 12.92)
            elif 0.00313066844250063 < in_rgba[i] <= 1.0:
                out_rgba.append(1.055 * in_rgba[i] ** (float(1.0)/2.4) - 0.055)
            elif in_rgba[i] > 1.0:
                out_rgba.append(1.0)
        out_rgba.append(in_rgba[3])
        return out_rgba


    def rgb_to_hsl(self, in_rgba):
        R, G, B = in_rgba[:3]
        Cmax = max(R, G, B)
        Cmin = min(R, G, B)

        H = 0.0
        S = 0.0
        L = (Cmax+Cmin)/2.0

        if L == 1.0:
            S = 0.0
        elif 0.0 < L < 0.5:
            S = (Cmax-Cmin)/(Cmax+Cmin)
        elif L >= 0.5:
            S = (Cmax-Cmin)/(2.0-Cmax-Cmin)

        if S > 0.0:
            if R == Cmax:
                H = ((G-B)/(Cmax-Cmin))*60.0
            elif G == Cmax:
                H = ((B-R)/(Cmax-Cmin)+2.0)*60.0
            elif B == Cmax:
                H = ((R-G)/(Cmax-Cmin)+4.0)*60.0

        return [H/360.0, S, L]


    def hsl_to_rgb(self, in_value):
        H, S, L = in_value

        v1 = 0.0
        v2 = 0.0

        rgb = [0.0, 0.0, 0.0]

        if S == 0.0:
            rgb = [L, L, L]
        else:
            if L < 0.5:
                v1 = L*(S+1.0)
            elif L >= 0.5:
                v1 = L+S-L*S

            v2 = 2.0*L-v1

            # H = H/360.0

            tR = H + 0.333333
            tG = H
            tB = H - 0.333333

            tList = [tR, tG, tB]

            for i, t in enumerate(tList):
                if t < 0.0:
                    t += 1.0
                elif t > 1.0:
                    t -= 1.0

                if t*6.0 < 1.0:
                    rgb[i] = v2+(v1-v2)*6.0*t
                elif t*2.0 < 1.0:
                    rgb[i] = v1
                elif t*3.0 < 2.0:
                    rgb[i] = v2+(v1-v2)*(0.666666 - t)*6.0
                else:
                    rgb[i] = v2

        return rgb


    def colors_to_values(self, colors, as_rgba=False):
        count = len(colors) // 4
        if as_rgba:
            for i in range(count):
                color = colors[(0+i*4):(4+i*4)]
                lum = self.color_to_luminance(color, premul=False)
                colors[(0+i*4):(4+i*4)] = [lum, lum, lum, color[3]]
            return colors
        else:
            values = [None] * count
            for i in range(count):
                values[i] = self.color_to_luminance(colors[(0+i*4):(4+i*4)])
            return values


    def values_to_colors(self, values, invert=False, as_alpha=False, with_mask=False, as_tuple=False):
        if invert:
            values = self.invert_values(values)

        count = len(values)
        colors = [None] * count * 4
        for i in range(count):
            if as_alpha:
                colors[(0+i*4):(4+i*4)] = [1.0, 1.0, 1.0, values[i]]
            elif with_mask:
                alpha = 1.0 if values[i] > 0.0 else 0.0
                colors[(0+i*4):(4+i*4)] = [values[i], values[i], values[i], alpha]
            else:
                colors[(0+i*4):(4+i*4)] = self.luminance_to_color(values[i])

        if as_tuple:
            rgba = [None] * count
            for i in range(count):
                rgba[i] = tuple(colors[(0+i*4):(4+i*4)])
            return rgba
        else:
            return colors


    def invert_values(self, values):
        return [1.0 - value for value in values]


    def __del__(self):
        print('SX Tools: Exiting convert')


# ------------------------------------------------------------------------
#    Value Generators and Utils
#    NOTE: Switching between EDIT and OBJECT modes is slow.
#          Make sure OBJECT mode is enabled before calling
#          any functions in this class!
# ------------------------------------------------------------------------
class SXTOOLS2_generate(object):
    def __init__(self):
        return None


    def blur_list(self, obj, layer, masklayer=None, returndict=False):

        def average_color(colors):
            sum_color = Vector((0.0, 0.0, 0.0, 0.0))
            for color in colors:
                sum_color += color
            return (sum_color / len(colors)) if colors else sum_color


        color_dict = {}
        vert_blur_dict = {}
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.types.BMVertSeq.ensure_lookup_table(bm.verts)

        colors = layers.get_layer(obj, layer)
        for vert in bm.verts:
            loop_colors = []
            for loop in vert.link_loops:
                loop_color = Vector(colors[(loop.index*4):(loop.index*4+4)])
                loop_colors.append(loop_color)
            
            color_dict[vert.index] = average_color(loop_colors)

        for vert in bm.verts:
            num_connected = len(vert.link_edges)
            if num_connected > 0:
                edge_weights = [(edge.other_vert(vert).co - vert.co).length for edge in vert.link_edges]
                max_weight = max(edge_weights)
                if (max_weight == 0):
                    vert_blur_dict[vert.index] = color_dict[vert.index]
                    continue
                else:
                    edge_weights = [weight if (weight > 0) else max_weight for weight in edge_weights]

                    # weights are inverted so near colors are more important
                    edge_weights = [int((1.1 - (weight / max_weight)) * 10) for weight in edge_weights]

                    neighbor_colors = [Vector(color_dict[vert.index])] * 20
                    for i, edge in enumerate(vert.link_edges):
                        for j in range(edge_weights[i]):
                            neighbor_colors.append(Vector(color_dict[edge.other_vert(vert).index]))

                    vert_blur_dict[vert.index] = average_color(neighbor_colors)
            else:
                vert_blur_dict[vert.index] = color_dict[vert.index]

        bm.free()

        if returndict:
            return vert_blur_dict
        else:
            vert_blur_list = self.vert_dict_to_loop_list(obj, vert_blur_dict, 4, 4)
            blur_list = self.mask_list(obj, vert_blur_list, masklayer)

            return blur_list


    def curvature_list(self, obj, norm_objs, masklayer=None, returndict=False):

        def generate_curvature_dict(curv_obj):
            vert_curv_dict = {}
            mesh = curv_obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.normal_update()

            # pass 1: calculate curvatures
            for vert in bm.verts:
                numConnected = len(vert.link_edges)
                if numConnected > 0:
                    edgeWeights = []
                    angles = []
                    for edge in vert.link_edges:
                        edgeWeights.append(edge.calc_length())
                        angles.append(math.acos(vert.normal.normalized() @ (edge.other_vert(vert).co - vert.co).normalized()))

                    total_weight = sum(edgeWeights)

                    vtxCurvature = 0.0
                    for i in range(numConnected):
                        curvature = angles[i] / math.pi - 0.5
                        vtxCurvature += curvature
                        # weighted_curvature = (curvature * edgeWeights[i]) / total_weight
                        # vtxCurvature += weighted_curvature

                    vtxCurvature = min(vtxCurvature / float(numConnected), 1.0)

                    vert_curv_dict[vert.index] = round(vtxCurvature, 5)
                else:
                    vert_curv_dict[vert.index] = 0.0

            # pass 2: if tiling, copy the curvature value from the connected vert of the edge that's pointing away from the mirror axis
            if curv_obj.sx2.tiling:
                if 'sxTiler' in curv_obj.modifiers:
                    curv_obj.modifiers['sxTiler'].show_viewport = False
                    curv_obj.modifiers.update()
                    bpy.context.view_layer.update()

                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([curv_obj, ], mode='local')
                tiling_props = [('tile_neg_x', 'tile_pos_x'), ('tile_neg_y', 'tile_pos_y'), ('tile_neg_z', 'tile_pos_z')]
                axis_vectors = [(Vector((-1.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0))), (Vector((0.0, -1.0, 0.0)), Vector((0.0, 1.0, 0.0))), (Vector((0.0, 0.0, -1.0)), Vector((0.0, 0.0, 1.0)))]
                bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

                for vert in bm.verts:
                    bound_ids = None
                    for i, coord in enumerate(vert.co):
                        for j, prop in enumerate(tiling_props[i]):
                            if getattr(obj.sx2, prop) and (round(coord, 2) == round(bounds[i][j], 2)):
                                bound_ids = (i, j)

                    if bound_ids is not None:
                        numConnected = len(vert.link_edges)
                        if numConnected > 0:
                            angles = []
                            other_verts = []
                            for i, edge in enumerate(vert.link_edges):
                                edgeVec = edge.other_vert(vert).co - vert.co
                                angles.append(axis_vectors[bound_ids[0]][bound_ids[1]].dot(edgeVec))
                                other_verts.append(edge.other_vert(vert))
                            value_vert_id = other_verts[angles.index(min(angles))].index
                            vert_curv_dict[vert.index] = vert_curv_dict[value_vert_id]
                        else:
                            vert_curv_dict[vert.index] = 0.0

                if 'sxTiler' in curv_obj.modifiers:
                    curv_obj.modifiers['sxTiler'].show_viewport = curv_obj.sx2.tile_preview
                    curv_obj.modifiers.update()
                    bpy.context.view_layer.update()

            bm.free()
            return vert_curv_dict


        scene = bpy.context.scene.sx2
        normalizeconvex = scene.normalizeconvex
        normalizeconcave = scene.normalizeconcave
        invert = scene.toolinvert

        # Generate curvature dictionary of all objects in a multi-selection
        if normalizeconvex or normalizeconcave:
            norm_obj_curvature_dict = {norm_obj: generate_curvature_dict(norm_obj) for norm_obj in norm_objs}
            min_curv = min(min(curv_dict.values()) for curv_dict in norm_obj_curvature_dict.values())
            max_curv = max(max(curv_dict.values()) for curv_dict in norm_obj_curvature_dict.values())
            vert_curv_dict = norm_obj_curvature_dict[obj]

            # Normalize convex and concave separately
            # to maximize artist ability to crease
            for vert, vtxCurvature in vert_curv_dict.items():
                if (vtxCurvature < 0.0) and normalizeconcave:
                    vert_curv_dict[vert] = (vtxCurvature / float(min_curv)) * -0.5 + 0.5
                elif (vtxCurvature > 0.0) and normalizeconvex:
                    vert_curv_dict[vert] = (vtxCurvature / float(max_curv)) * 0.5 + 0.5
                else:
                    vert_curv_dict[vert] = (vtxCurvature + 0.5)
        else:
            vert_curv_dict = generate_curvature_dict(obj)
            for vert, vtxCurvature in vert_curv_dict.items():
                vert_curv_dict[vert] = (vtxCurvature + 0.5)

        if invert:
            for key, value in vert_curv_dict.items():
                vert_curv_dict[key] = 1.0 - value

        if returndict:
            return vert_curv_dict

        else:
            vert_curv_list = self.vert_dict_to_loop_list(obj, vert_curv_dict, 1, 4)
            curv_list = self.mask_list(obj, vert_curv_list, masklayer)

            return curv_list


    def direction_list(self, obj, masklayer=None):
        scene = bpy.context.scene.sx2
        cone_angle = scene.dirCone
        half_cone_angle = cone_angle * 0.5

        vert_dict = self.vertex_data_dict(obj, masklayer)

        if vert_dict:
            vert_dir_dict = {vert_id: 0.0 for vert_id in vert_dict}
            inclination = math.radians(scene.dirInclination - 90.0)
            angle = math.radians(scene.dirAngle + 90)
            direction = Vector((math.sin(inclination) * math.cos(angle), math.sin(inclination) * math.sin(angle), math.cos(inclination)))

            for vert_id in vert_dict:
                vert_world_normal = Vector(vert_dict[vert_id][3])
                angle_diff = math.degrees(math.acos(min(1.0, max(-1.0, vert_world_normal @ direction))))
                vert_dir_dict[vert_id] = max(0.0, (90.0 + half_cone_angle - angle_diff) / (90.0 + half_cone_angle))

            values = self.vert_dict_to_loop_list(obj, vert_dir_dict, 1, 1)
            vert_dir_list = [None] * len(values) * 4
            for i in range(len(values)):
                vert_dir_list[(0+i*4):(4+i*4)] = [values[i], values[i], values[i], 1.0]

            return self.mask_list(obj, vert_dir_list, masklayer)
        else:
            return None


    def noise_list(self, obj, amplitude=0.5, offset=0.5, mono=False, masklayer=None):

        def make_noise(amplitude, offset, mono):
            col = [None, None, None, 1.0]
            if mono:
                monoval = offset+random.uniform(-amplitude, amplitude)
                for i in range(3):
                    col[i] = monoval
            else:
                for i in range(3):
                    col[i] = offset+random.uniform(-amplitude, amplitude)
            return col

        random.seed(sxglobals.randomseed)
        vert_ids = self.vertex_id_list(obj)
        noise_dict = {vtx_id: make_noise(amplitude, offset, mono) for vtx_id in vert_ids}
        noise_list = self.vert_dict_to_loop_list(obj, noise_dict, 4, 4)
        return self.mask_list(obj, noise_list, masklayer)


    def ray_randomizer(self, count):
        hemisphere = [None] * count
        random.seed(sxglobals.randomseed)

        for i in range(count):
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(u1)
            theta = 2*math.pi*u2

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = math.sqrt(max(0, 1 - u1))

            ray = Vector((x, y, z))
            up_vector = Vector((0, 0, 1))
            
            dot_product = ray.dot(up_vector)
            hemisphere[i] = (ray, dot_product)

        sorted_hemisphere = sorted(hemisphere, key=lambda x: x[1], reverse=True)
        return sorted_hemisphere


    def ground_plane(self, size, pos):
        size *= 0.5
        vert_list = [
            (pos[0]-size, pos[1]-size, pos[2]),
            (pos[0]+size, pos[1]-size, pos[2]),
            (pos[0]-size, pos[1]+size, pos[2]),
            (pos[0]+size, pos[1]+size, pos[2])]
        face_list = [(0, 1, 3, 2)]

        mesh = bpy.data.meshes.new('groundPlane_mesh')
        groundPlane = bpy.data.objects.new('groundPlane', mesh)
        bpy.context.scene.collection.objects.link(groundPlane)

        mesh.from_pydata(vert_list, [], face_list)
        mesh.update(calc_edges=True)

        # groundPlane.location = pos
        return groundPlane, mesh


    def thickness_list(self, obj, raycount, masklayer=None):

        def dist_caster(obj, vert_dict):
            hemisphere = self.ray_randomizer(20)
            for vert_id, vert_data in vert_dict.items():
                vertLoc = vert_data[0]
                vertNormal = vert_data[1]
                invNormal = -vertNormal
                rotQuat = forward.rotation_difference(invNormal)
                bias = 0.001

                # Raycast for bias
                hit, loc, normal, _ = obj.ray_cast(vertLoc, invNormal)
                if hit and (normal.dot(invNormal) < 0):
                    hit_dist = (loc - vertLoc).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                # offset ray origin with normal bias
                vertPos = vertLoc + (bias * invNormal)

                # Raycast for distance
                for ray, _ in hemisphere:
                    hit, loc, normal, _ = obj.ray_cast(vertPos, rotQuat @ Vector(ray))
                    if hit:
                        dist_list.append((loc - vertPos).length)

                bias_vert_dict[vert_id] = (vertPos, invNormal)


        def sample_caster(obj, raycount, vert_dict, raydistance=1.70141e+38):
            hemisphere = self.ray_randomizer(raycount)
            for vert_id, vert_data in vert_dict.items():
                vertPos = vert_data[0]
                invNormal = vert_data[1]
                rotQuat = forward.rotation_difference(invNormal)

                for ray, _ in hemisphere:
                    hit = obj.ray_cast(vertPos, rotQuat @ Vector(ray), distance=raydistance)[0]
                    vert_occ_dict[vert_id] += contribution * hit


        vert_dict = self.vertex_data_dict(obj, masklayer)
        if vert_dict:
            edg = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(edg)
            contribution = 1.0 / float(raycount)
            forward = Vector((0.0, 0.0, 1.0))
            dist_list = []
            bias_vert_dict = {}
            vert_occ_dict = {vert_id: 0.0 for vert_id in vert_dict}

            # Pass 1: analyze ray hit distances, set max ray distance to half of median distance
            dist_caster(obj_eval, vert_dict)
            distance = statistics.median(dist_list) * 0.5

            # Pass 2: final results
            sample_caster(obj_eval, raycount, bias_vert_dict, raydistance=distance)

            vert_occ_list = generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)
            return self.mask_list(obj, vert_occ_list, masklayer)
        else:
            return None


    def occlusion_list(self, obj, raycount=250, blend=0.5, dist=10.0, groundplane=False, groundheight=-0.5, masklayer=None):
        if sxglobals.benchmark_ao:
            then = time.perf_counter()

        scene = bpy.context.scene
        contribution = 1.0/float(raycount)
        hemisphere = self.ray_randomizer(raycount)
        mix = max(min(blend, 1.0), 0.0)
        forward = Vector((0.0, 0.0, 1.0))

        if obj.sx2.tiling:
            blend = 0.0
            groundplane = False
            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ], mode='local')
            dist = 2.0 * min(xmax-xmin, ymax-ymin, zmax-zmin)
            if 'sxTiler' not in obj.modifiers:
                modifiers.add_modifiers([obj, ])
            obj.modifiers['sxTiler'].show_viewport = True

        vert_occ_dict = {}
        vert_dict = self.vertex_data_dict(obj, masklayer, dots=True)

        if vert_dict:
            if (blend > 0.0) and groundplane:
                pivot = utils.find_root_pivot([obj, ])
                pivot = (pivot[0], pivot[1], groundheight)
                size = max(obj.dimensions) * 10
                ground, groundmesh = self.ground_plane(size, pivot)

            edg = bpy.context.evaluated_depsgraph_get()
            # edg.update()
            # obj_eval = obj.evaluated_get(edg)
            bvh = BVHTree.FromObject(obj, edg)

            for vert_id in vert_dict:
                bias = 0.001
                occValue = 1.0
                scnOccValue = 1.0
                vertLoc, vertNormal, vertWorldLoc, vertWorldNormal, min_dot = vert_dict[vert_id]

                # use modified tile-border normals to reduce seam artifacts
                # if vertex pos x y z is at bbx limit, and mirror axis is set, modify respective normal vector component to zero
                if obj.sx2.tiling:
                    mod_normal = list(vertNormal)
                    match = False

                    tiling_props = [('tile_neg_x', 'tile_pos_x'), ('tile_neg_y', 'tile_pos_y'), ('tile_neg_z', 'tile_pos_z')]
                    bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

                    for i, coord in enumerate(vertLoc):
                        for j, prop in enumerate(tiling_props[i]):
                            if getattr(obj.sx2, prop) and (round(coord, 2) == round(bounds[i][j], 2)):
                                match = True
                                mod_normal[i] = 0.0

                    if match:
                        vertNormal = Vector(mod_normal[:]).normalized()

                # Pass 0: Raycast for bias
                # hit, loc, normal, _ = obj.ray_cast(vertLoc, vertNormal, distance=dist)
                result = bvh.ray_cast(vertLoc, vertNormal, dist)
                hit = not all(x is None for x in result)
                _, normal, _, hit_dist = result
                if hit and (normal.dot(vertNormal) > 0):
                    # hit_dist = (loc - vertLoc).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                # Pass 1: Mark hits for rays that are inside the mesh
                first_hit_index = raycount
                for i, (_, dot) in enumerate(hemisphere):
                    if dot < min_dot:
                        first_hit_index = i
                        break

                valid_rays = [ray for ray, _ in hemisphere[:first_hit_index]]
                occValue -= contribution * (raycount - first_hit_index)

                # Store Pass 2 valid ray hits
                pass2_hits = [False] * len(valid_rays)

                # Pass 2: Local space occlusion for individual object
                if 0.0 <= mix < 1.0:
                    rotQuat = forward.rotation_difference(vertNormal)

                    # offset ray origin with normal bias
                    vertPos = vertLoc + (bias * vertNormal)

                    # for every object ray hit, subtract a fraction from the vertex brightness
                    for i, ray in enumerate(valid_rays):
                        # hit = obj_eval.ray_cast(vertPos, rotQuat @ Vector(ray), distance=dist)[0]
                        result = bvh.ray_cast(vertPos, rotQuat @ Vector(ray), dist)
                        hit = not all(x is None for x in result)
                        occValue -= contribution * hit
                        pass2_hits[i] = hit

                # Pass 3: Worldspace occlusion for scene
                if 0.0 < mix <= 1.0:
                    rotQuat = forward.rotation_difference(vertWorldNormal)

                    # offset ray origin with normal bias
                    scnVertPos = vertWorldLoc + (bias * vertWorldNormal)

                    # Include previous pass results
                    scnOccValue = occValue

                    # Fire rays only for samples that had not hit in Pass 2
                    for i, ray in enumerate(valid_rays):
                        if not pass2_hits[i]:
                            hit = scene.ray_cast(edg, scnVertPos, rotQuat @ Vector(ray), distance=dist)[0]
                            scnOccValue -= contribution * hit

                vert_occ_dict[vert_id] = float((occValue * (1.0 - mix)) + (scnOccValue * mix))

            if (blend > 0.0) and groundplane:
                bpy.data.objects.remove(ground, do_unlink=True)
                bpy.data.meshes.remove(groundmesh, do_unlink=True)

            if obj.sx2.tiling:
                obj.modifiers['sxTiler'].show_viewport = False

            vert_occ_list = generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)
            result = self.mask_list(obj, vert_occ_list, masklayer)

            if sxglobals.benchmark_ao:
                now = time.perf_counter()
                print(f'SX Tools: {obj.name} AO rendered in {round(now-then, 4)} seconds')

            return result

        else:
            return None


    def emission_list(self, obj, raycount=250, masklayer=None):

        def calculate_face_colors(obj):
            colors = layers.get_layer(obj, obj.sx2layers['Emission'], as_tuple=True)
            face_colors = [None] * len(obj.data.polygons)

            for face in obj.data.polygons:
                face_color = Vector((0.0, 0.0, 0.0, 0.0))
                for loop_index in face.loop_indices:
                    loop_color = Vector(colors[loop_index])
                    if loop_color[3] > 0.0:
                        face_color += Vector(colors[loop_index])
                face_colors[face.index] = face_color / len(face.loop_indices)

            return face_colors


        def face_colors_to_loop_list(obj, face_colors):
            loop_list = self.empty_list(obj, 4)

            for poly in obj.data.polygons:
                for loop_idx in poly.loop_indices:
                    loop_list[(0+loop_idx*4):(4+loop_idx*4)] = face_colors[poly.index]

            return loop_list


        _, empty = layers.get_layer_mask(obj, obj.sx2layers['Emission'])
        vert_dict = self.vertex_data_dict(obj, masklayer, dots=False)

        if vert_dict and not empty:
            mod_vis = [modifier.show_viewport for modifier in obj.modifiers]
            for modifier in obj.modifiers:
                modifier.show_viewport = False

            hemi_up = Vector((0.0, 0.0, 1.0))
            vert_dict = self.vertex_data_dict(obj, masklayer, dots=False)
            face_colors = calculate_face_colors(obj)
            original_emissive_vertex_colors = {}
            original_emissive_vertex_face_count = [0 for _ in obj.data.vertices]
            dist = max(utils.get_object_bounding_box([obj, ], mode='local')) * 5
            bias = 0.001

            for face in obj.data.polygons:
                color = face_colors[face.index]
                if color.length > 0:
                    for vert_idx in face.vertices:
                        vert_color = original_emissive_vertex_colors.get(vert_idx, Vector((0.0, 0.0, 0.0, 0.0)))
                        original_emissive_vertex_colors[vert_idx] = vert_color + color
                        original_emissive_vertex_face_count[vert_idx] += 1

            # Pass 1: Propagate emission to face colors
            hemisphere = self.ray_randomizer(raycount)
            contribution = 1.0 / float(raycount)
            for i in range(10):
                for j, face in enumerate(obj.data.polygons):
                    face_emission = Vector((0.0, 0.0, 0.0, 0.0))
                    vertices = [obj.data.vertices[face_vert_id].co for face_vert_id in face.vertices]
                    face_center = (sum(vertices, Vector()) / len(vertices)) + (bias * face.normal)
                    rotQuat = hemi_up.rotation_difference(face.normal)

                    for sample, _ in hemisphere:
                        sample_ray = rotQuat @ sample
                        hit, _, hit_normal, hit_face_index = obj.ray_cast(face_center, sample_ray, distance=dist)
                        if hit and (hit_normal.dot(sample_ray) < 0):
                            face_color = face_colors[hit_face_index].copy()
                            addition = face_color * contribution
                            face_emission += addition

                    if face_emission.length > face_colors[j].length:
                        face_colors[j] = face_emission.copy()

            # Pass 2: Average face colors to vertices
            vertex_colors = [Vector((0, 0, 0, 0)) for _ in obj.data.vertices]
            vertex_faces = [[] for _ in obj.data.vertices]
            vert_emission_list = self.empty_list(obj, 4)

            for face in obj.data.polygons:
                color = face_colors[face.index]
                if color.length > 0:
                    for vert_idx in face.vertices:
                        vertex_colors[vert_idx] += color
                        vertex_faces[vert_idx].append(face.index)

            for vert_idx, color_sum in enumerate(vertex_colors):
                if vert_idx in original_emissive_vertex_colors:
                    vertex_colors[vert_idx] = original_emissive_vertex_colors[vert_idx] / original_emissive_vertex_face_count[vert_idx]
                else:
                    if len(vertex_faces[vert_idx]) > 0:
                        vertex_colors[vert_idx] = color_sum / len(vertex_faces[vert_idx])

            for loop in obj.data.loops:
                vert_emission_list[(0+loop.index*4):(4+loop.index*4)] = vertex_colors[loop.vertex_index]

            # vert_emission_list = face_colors_to_loop_list(obj, face_colors)
            result = self.mask_list(obj, vert_emission_list, masklayer)

            for i, modifier in enumerate(obj.modifiers):
                modifier.show_viewport = mod_vis[i]

            return result
        else:
            return None


    def mask_list(self, obj, colors, masklayer=None, maskcolor=None, as_tuple=False, override_mask=False):
        count = len(colors)//4

        # No mask, colors pass through
        if (masklayer is None) and (maskcolor is None) and (sxglobals.mode != 'EDIT'):
            if as_tuple:
                rgba = [None] * count
                for i in range(count):
                    rgba[i] = tuple(colors[(0+i*4):(4+i*4)])
                return rgba
            else:
                return colors

        # Colors are masked by selection or layer alpha
        else:
            if masklayer is None:
                mask, empty = self.get_selection_mask(obj, selected_color=maskcolor)
                if empty:
                    return None
            # Layer is locked and there is an edit mode component selection
            elif (masklayer is not None) and (sxglobals.mode == 'EDIT'):
                mask1, empty = self.get_selection_mask(obj, selected_color=maskcolor)
                if empty:
                    return None
                else:
                    mask2, empty = layers.get_layer_mask(obj, masklayer)
                    if empty:
                        mask = mask1
                    else:
                        mask = [0.0] * len(mask1)
                        for i, (m1, m2) in enumerate(zip(mask1, mask2)):
                            if (m1 > 0.0) and (m2 > 0.0):
                                mask[i] = min(m1, m2)
            else:
                mask, empty = layers.get_layer_mask(obj, masklayer)
                if empty:
                    return None

            if as_tuple:
                rgba = [None] * count
                for i in range(count):
                    rgba[i] = tuple(Vector(colors[(0+i*4):(4+i*4)]) * mask[i])
                return rgba
            else:
                color_list = [None, None, None, None] * count
                if override_mask:
                    for i in range(count):
                        color = colors[(0+i*4):(4+i*4)]
                        color[3] = mask[i]
                        color_list[(0+i*4):(4+i*4)] = color
                else:
                    for i in range(count):
                        color = colors[(0+i*4):(4+i*4)]
                        color[3] *= mask[i]
                        color_list[(0+i*4):(4+i*4)] = color

                return color_list


    def color_list(self, obj, color, masklayer=None, as_tuple=False):
        count = len(obj.data.loops)
        colors = [color[0], color[1], color[2], color[3]] * count
        return self.mask_list(obj, colors, masklayer, as_tuple=as_tuple)


    def ramp_list(self, obj, objs, rampmode, masklayer=None, mergebbx=True):
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp']

        # For OBJECT mode selections
        if sxglobals.mode == 'OBJECT':
            if mergebbx:
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box(objs)
            else:
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ])

        # For EDIT mode multi-obj component selection
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_selection_bounding_box(objs)

        xdiv = float(xmax - xmin) or 1.0
        ydiv = float(ymax - ymin) or 1.0
        zdiv = float(zmax - zmin) or 1.0

        vertPosDict = self.vertex_data_dict(obj, masklayer)
        ramp_dict = {}

        for vert_id in vertPosDict:
            ratioRaw = None
            ratio = None
            fvPos = vertPosDict[vert_id][2]

            if rampmode == 'X':
                ratioRaw = ((fvPos[0] - xmin) / xdiv)
            elif rampmode == 'Y':
                ratioRaw = ((fvPos[1] - ymin) / ydiv)
            elif rampmode == 'Z':
                ratioRaw = ((fvPos[2] - zmin) / zdiv)

            ratio = max(min(ratioRaw, 1.0), 0.0)
            ramp_dict[vert_id] = ramp.color_ramp.evaluate(ratio)

        ramp_list = self.vert_dict_to_loop_list(obj, ramp_dict, 4, 4)

        return self.mask_list(obj, ramp_list, masklayer)


    def luminance_remap_list(self, obj, layer=None, masklayer=None, values=None):
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp']

        if values is None:
            values = layers.get_luminances(obj, layer, as_rgba=False)
        colors = generate.empty_list(obj, 4)
        count = len(values)

        for i in range(count):
            ratio = max(min(values[i], 1.0), 0.0)
            colors[(0+i*4):(4+i*4)] = ramp.color_ramp.evaluate(ratio)

        return self.mask_list(obj, colors, masklayer)


    def vertex_id_list(self, obj):
        ids = [None] * len(obj.data.vertices)
        obj.data.vertices.foreach_get('index', ids)
        return ids


    def empty_list(self, obj, channelcount):
        return [0.0] * len(obj.data.loops) * channelcount


    def vert_dict_to_loop_list(self, obj, vert_dict, dictchannelcount, listchannelcount):
        mesh = obj.data
        loop_list = self.empty_list(obj, listchannelcount)

        if dictchannelcount < listchannelcount:
            if (dictchannelcount == 1) and (listchannelcount == 2):
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = [value, value]
            elif (dictchannelcount == 1) and (listchannelcount == 4):
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = [value, value, value, 1.0]
            elif (dictchannelcount == 3) and (listchannelcount == 4):
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, [0.0, 0.0, 0.0])
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = [value[0], value[1], value[2], 1.0]
        else:
            if listchannelcount == 1:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[loop_idx] = vert_dict.get(vert_idx, 0.0)
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = vert_dict.get(vert_idx, [0.0] * listchannelcount)

        return loop_list


    def vertex_data_dict(self, obj, masklayer=None, dots=False):

        def add_to_dict(vert_id):
            min_dot = None
            if dots:
                dot_list = []
                vert = bm.verts[vert_id]
                num_connected = len(vert.link_edges)
                if num_connected > 0:
                    for edge in vert.link_edges:
                        dot_list.append((vert.normal.normalized()).dot((edge.other_vert(vert).co - vert.co).normalized()))
                min_dot = min(dot_list)

            vertex_dict[vert_id] = (
                mesh.vertices[vert_id].co,
                mesh.vertices[vert_id].normal,
                mat @ mesh.vertices[vert_id].co,
                (mat @ mesh.vertices[vert_id].normal - mat @ Vector()).normalized(),
                min_dot
            )


        mesh = obj.data
        mat = obj.matrix_world
        ids = self.vertex_id_list(obj)

        if dots:
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.normal_update()
            bmesh.types.BMVertSeq.ensure_lookup_table(bm.verts)

        vertex_dict = {}
        if masklayer is not None:
            mask, empty = layers.get_layer_mask(obj, masklayer)
            if not empty:
                for poly in mesh.polygons:
                    for vert_id, loop_idx in zip(poly.vertices, poly.loop_indices):
                        if mask[loop_idx] > 0.0:
                            add_to_dict(vert_id)

        elif sxglobals.mode == 'EDIT':
            vert_sel = [None] * len(mesh.vertices)
            mesh.vertices.foreach_get('select', vert_sel)
            if True in vert_sel:
                for vert_id, sel in enumerate(vert_sel):
                    if sel:
                        add_to_dict(vert_id)

        else:
            for vert_id in ids:
                add_to_dict(vert_id)

        if dots:
            bm.free()

        return vertex_dict


    # Returns mask and empty status
    def get_selection_mask(self, obj, selected_color=None):
        mesh = obj.data
        mask = self.empty_list(obj, 1)

        if selected_color is None:
            pre_check = [None] * len(mesh.vertices)
            mesh.vertices.foreach_get('select', pre_check)
            if True not in pre_check:
                empty = True
            else:
                empty = False
                if bpy.context.tool_settings.mesh_select_mode[2]:
                    for poly in mesh.polygons:
                        for loop_idx in poly.loop_indices:
                            mask[loop_idx] = float(poly.select)
                else:
                    for poly in mesh.polygons:
                        for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                            mask[loop_idx] = float(mesh.vertices[vert_idx].select)
        else:
            export.composite_color_layers([obj, ])
            colors = layers.get_layer(obj, obj.sx2layers['Composite'])

            if bpy.context.tool_settings.mesh_select_mode[2]:
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        mask[loop_idx] = float(utils.color_compare(colors[(0+loop_idx*4):(4+loop_idx*4)], selected_color, 0.01))
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        mask[loop_idx] = float(utils.color_compare(colors[(0+loop_idx*4):(4+loop_idx*4)], selected_color, 0.01))

            empty = 1.0 not in mask

        return mask, empty


    def __del__(self):
        print('SX Tools: Exiting generate')


# ------------------------------------------------------------------------
#    Layer Functions
#    NOTE: Objects must be in OBJECT mode before calling layer functions,
#          use utils.mode_manager() before calling layer functions
#          to set and track correct state
# ------------------------------------------------------------------------
class SXTOOLS2_layers(object):
    def __init__(self):
        return None


    def add_layer(self, objs, name=None, layer_type=None, color_attribute=None, clear=True):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        data_types = {'FLOAT': 'FLOAT_COLOR', 'BYTE': 'BYTE_COLOR'}
        alpha_mats = {'OCC': 'Occlusion', 'MET': 'Metallic', 'RGH': 'Roughness', 'TRN': 'Transmission'}
        layer_dict = {}
        if objs:
            # Use largest layercount to avoid material mismatches
            for obj in objs:
                layercount = max([obj.sx2.layercount for obj in objs])

            for obj in objs:
                if name and name in obj.sx2layers.keys():
                    print(f'SX Tools Error: {obj.name} layer {name} already exists')
                    break
                else:
                    if name and name not in obj.sx2layers.keys():
                        layer_name = name
                    else:
                        index = 1
                        layer_name = 'Layer ' + str(layercount)
                        while layer_name in obj.sx2layers.keys():
                            layer_name = 'Layer ' + str(layercount + index)
                            index += 1

                    layer = obj.sx2layers.add()
                    layer.name = layer_name
                    if layer_type not in alpha_mats:
                        layer.color_attribute = color_attribute if color_attribute is not None else layer.name
                    else:
                        layer.color_attribute = 'Alpha Materials'
                    layer.layer_type = 'COLOR' if layer_type is None else layer_type
                    layer.default_color = sxglobals.default_colors[layer.layer_type]
                    # item.paletted is False by default

                    if layer_type not in alpha_mats:
                        if color_attribute not in obj.data.color_attributes.keys():
                            obj.data.color_attributes.new(name=layer.name, type=data_types[prefs.layerdatatype], domain='CORNER')
                    elif ('Alpha Materials' not in obj.data.color_attributes.keys()) and (layer_type in alpha_mats):
                        obj.data.color_attributes.new(name='Alpha Materials', type=data_types[prefs.layerdatatype], domain='CORNER')

                    if layer_type == 'CMP':
                        layer.index = utils.insert_layer_at_index(obj, layer, 0)
                    else:
                        layer.index = utils.insert_layer_at_index(obj, layer, obj.sx2layers[obj.sx2.selectedlayer].index + 1)

                    if clear:
                        colors = generate.color_list(obj, layer.default_color)
                        layers.set_layer(obj, colors, obj.sx2layers[len(obj.sx2layers) - 1])
                    layer_dict[obj] = layer
                    obj.sx2.layercount = layercount + 1
                    # obj.sx2.selectedlayer = utils.find_layer_index_by_name(obj, layer.name)

            # selectedlayer needs to point to an existing layer on all objs
            # NOTE: Potential bug here! Objects with less layers?
            objs[0].sx2.selectedlayer = len(objs[0].sx2layers) - 1

        return layer_dict


    def del_layer(self, objs, layer_name):
        for obj in objs:
            bottom_layer_index = obj.sx2layers[layer_name].index - 1 if obj.sx2layers[layer_name].index - 1 > 0 else 0
            if obj.sx2layers[layer_name].color_attribute == 'Alpha Materials':
                alpha_mat_count = sum(1 for layer in obj.sx2layers if layer.color_attribute == 'Alpha Materials')
                if alpha_mat_count == 1:
                    obj.data.attributes.remove(obj.data.attributes[obj.sx2layers[layer_name].color_attribute])
            else:
                variants = [color_attribute.name for color_attribute in obj.data.color_attributes if obj.sx2layers[layer_name].color_attribute.split('_var')[0] in color_attribute.name]
                for variant in variants:
                    obj.data.color_attributes.remove(obj.data.color_attributes[variant])
            idx = utils.find_layer_index_by_name(obj, layer_name)
            obj.sx2layers.remove(idx)

        for obj in objs:
            utils.sort_stack_indices(obj)

        for i, layer in enumerate(objs[0].sx2layers):
            if layer.index == bottom_layer_index:
                objs[0].sx2.selectedlayer = i


    def add_variant_layer(self, obj, layer, clear=True):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        data_types = {'FLOAT': 'FLOAT_COLOR', 'BYTE': 'BYTE_COLOR'}
        alpha_mats = {'OCC': 'Occlusion', 'MET': 'Metallic', 'RGH': 'Roughness', 'TRN': 'Transmission'}

        if (layer.layer_type in alpha_mats):
            message_box('Cannot add variants to alpha layers')
            return

        name = layer.color_attribute.split('_var')[0] + '_var' + str(layer.variant_count + 1)
        obj.data.color_attributes.new(name=name, type=data_types[prefs.layerdatatype], domain='CORNER')
        layer.variant_count += 1

        if clear:
            colors = generate.color_list(obj, layer.default_color)
        else:
            colors = layers.get_colors(obj, layer.color_attribute)

        layers.set_colors(obj, name, colors)
        layer.variant_index = layer.variant_count
        self.set_variant_layer(layer, layer.variant_index)


    def del_variant_layer(self, obj, layer, index):
        if index == 0:
            obj.data.color_attributes.remove(obj.data.color_attributes[layer.color_attribute.split('_var')[0]])
        else:
            obj.data.color_attributes.remove(obj.data.color_attributes[layer.color_attribute.split('_var')[0] + '_var' + str(index)])

        layer.variant_count -= 1
        self.sort_variant_layers(obj, layer)

        if index != 0:
            self.set_variant_layer(layer, index - 1)
            layer.variant_index = index - 1
        else:
            self.set_variant_layer(layer, 0)
            layer.variant_index = 0


    def set_variant_layer(self, layer, index):
        if index == 0:
            layer.color_attribute = layer.color_attribute.split('_var')[0]
        else:
            layer.color_attribute = layer.color_attribute + '_var' + str(index)


    def sort_variant_layers(self, obj, layer):
        variants = [attribute for attribute in obj.data.color_attributes if layer.color_attribute.split('_var')[0] in attribute.name]
        sorted_variants = sorted(variants, key=lambda color_attribute: color_attribute.name)
        base_name = layer.color_attribute.split('_var')[0]
        new_names = [base_name]

        for i, variant in enumerate(sorted_variants, start=1):
            new_name = f"{base_name}_var{i}"
            new_names.append(new_name)

        for i, variant in enumerate(sorted_variants):
            variant.name = new_names[i]


    # wrapper for low-level functions, always returns layerdata in RGBA
    def get_layer(self, obj, sourcelayer, as_tuple=False, single_as_alpha=False, apply_layer_opacity=False):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP', 'CID']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        dv = [1.0, 1.0, 1.0, 1.0]

        if sourcelayer.layer_type in rgba_targets:
            values = self.get_colors(obj, sourcelayer.color_attribute)

        elif sourcelayer.layer_type in alpha_targets:
            source_values = self.get_colors(obj, sourcelayer.color_attribute)
            values = [None] * len(source_values)
            for i in range(len(source_values)//4):
                value_slice = source_values[(0+i*4):(4+i*4)]
                value = value_slice[alpha_targets[sourcelayer.layer_type]]
                if single_as_alpha:
                    if value > 0.0:
                        values[(0+i*4):(4+i*4)] = [dv[0], dv[1], dv[2], value]
                    else:
                        values[(0+i*4):(4+i*4)] = [0.0, 0.0, 0.0, value]
                else:
                    values[(0+i*4):(4+i*4)] = [value, value, value, 1.0]

        if apply_layer_opacity and sourcelayer.opacity != 1.0:
            count = len(values)//4
            for i in range(count):
                values[3+i*4] *= sourcelayer.opacity

        if as_tuple:
            count = len(values)//4
            rgba = [None] * count
            for i in range(count):
                rgba[i] = tuple(values[(0+i*4):(4+i*4)])
            return rgba

        else:
            return values


    # takes RGBA buffers, converts and writes to appropriate channels
    def set_layer(self, obj, colors, targetlayer):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP', 'CID']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        target_type = targetlayer.layer_type

        if target_type in rgba_targets:
            layers.set_colors(obj, targetlayer.color_attribute, colors)

        elif target_type in alpha_targets:
            target_values = self.get_colors(obj, 'Alpha Materials')
            values = layers.get_luminances(obj, sourcelayer=None, colors=colors, as_rgba=False)
            for i in range(len(values)):
                target_values[alpha_targets[target_type]+i*4] = values[i]
            self.set_colors(obj, 'Alpha Materials', target_values)


    def get_layer_mask(self, obj, sourcelayer, channel='A'):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP', 'CID']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        channel_targets = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
        layer_type = sourcelayer.layer_type

        if layer_type in rgba_targets:
            colors = self.get_colors(obj, sourcelayer.color_attribute)
            values = colors[channel_targets[channel]::4]
        elif layer_type in alpha_targets:
            colors = self.get_colors(obj, sourcelayer.color_attribute)
            values = colors[alpha_targets[layer_type]::4]

        if any(v != 0.0 for v in values):
            return values, False
        else:
            return values, True


    def get_colors(self, obj, source_name):
        source_colors = obj.data.color_attributes[source_name].data
        colors = [None] * len(source_colors) * 4
        source_colors.foreach_get('color', colors)
        return colors


    def set_colors(self, obj, target, colors):
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()


    def set_alphas(self, obj, target, values):
        colors = self.get_colors(obj, target)
        count = len(values)
        for i in range(count):
            color = colors[(0+i*4):(4+i*4)]
            color = [color[0], color[1], color[2], values[i]]
            colors[(0+i*4):(4+i*4)] = color
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()


    def set_channel(self, obj, target, values, channel):
        channel_dict = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
        source_index = channel_dict[channel]
        colors = self.get_colors(obj, target)
        for i, value in enumerate(values):
            colors[i*4+source_index] = value
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()


    def get_luminances(self, obj, sourcelayer=None, colors=None, as_rgba=False, as_alpha=False):
        if colors is None:
            if sourcelayer is not None:
                colors = self.get_layer(obj, sourcelayer)
            else:
                colors = generate.empty_list(obj, 4)

        if as_rgba:
            values = generate.empty_list(obj, 4)
            count = len(values)//4
            for i in range(count):
                values[(0+i*4):(4+i*4)] = convert.luminance_to_color(convert.color_to_luminance(colors[(0+i*4):(4+i*4)]))
        elif as_alpha:
            values = generate.empty_list(obj, 4)
            count = len(values)//4
            for i in range(count):
                values[(0+i*4):(4+i*4)] = convert.luminance_to_alpha(convert.color_to_luminance(colors[(0+i*4):(4+i*4)]))
        else:
            values = generate.empty_list(obj, 1)
            count = len(values)
            for i in range(count):
                values[i] = convert.color_to_luminance(colors[(0+i*4):(4+i*4)])

        return values


    def get_uvs(self, obj, source, channel=None):
        channels = {'U': 0, 'V': 1}
        source_uvs = obj.data.uv_layers[source].data
        count = len(source_uvs)
        source_values = [None] * count * 2
        source_uvs.foreach_get('uv', source_values)

        if channel is None:
            uvs = source_values
        else:
            uvs = [None] * count
            sc = channels[channel]
            for i in range(count):
                uvs[i] = source_values[sc+i*2]

        return uvs


    # when targetchannel is None, sourceuvs is expected to contain data for both U and V
    def set_uvs(self, obj, target, sourceuvs, targetchannel=None):
        channels = {'U': 0, 'V': 1}
        target_uvs = obj.data.uv_layers[target].data

        if targetchannel is None:
            target_uvs.foreach_set('uv', sourceuvs)
        else:
            target_values = self.get_uvs(obj, target)
            tc = channels[targetchannel]
            count = len(sourceuvs)
            for i in range(count):
                target_values[tc+i*2] = sourceuvs[i]
            target_uvs.foreach_set('uv', target_values)


    def clear_layers(self, objs, targetlayer=None):
        scene = bpy.context.scene.sx2

        def clear_layer(obj, layer, reset=False):
            default_color = layer.default_color[:]
            if sxglobals.mode == 'OBJECT':
                setattr(obj.sx2layers[layer.name], 'opacity', 1.0)
                setattr(obj.sx2layers[layer.name], 'visibility', True)
                setattr(obj.sx2layers[layer.name], 'locked', False)
                if reset:
                    setattr(obj.sx2layers[layer.name], 'blend_mode', 'ALPHA')
                colors = generate.color_list(obj, default_color)
                layers.set_layer(obj, colors, layer)
            else:
                colors = layers.get_layer(obj, layer)
                mask, empty = generate.get_selection_mask(obj)
                if not empty:
                    for i in range(len(mask)):
                        if mask[i] == 1.0:
                            colors[(0+i*4):(4+i*4)] = default_color
                    layers.set_layer(obj, colors, layer)

        if targetlayer is None:
            print('SX Tools: Clearing all layers')
            scene.toolopacity = 1.0
            scene.toolblend = 'ALPHA'
            for obj in objs:
                for sxlayer in obj.sx2layers:
                    sxlayer.locked = False
                    clear_layer(obj, sxlayer, reset=True)
        else:
            for obj in objs:
                clear_layer(obj, obj.sx2layers[targetlayer.name])

    # NOTE: Active commented!
    def blend_layers(self, objs, topLayerList, baseLayer, resultLayer, single_as_alpha=False):
        # active = bpy.context.view_layer.objects.active
        # bpy.context.view_layer.objects.active = objs[0]
        top_layers = [layer.name for layer in topLayerList]

        for obj in objs:
            basecolors = self.get_layer(obj, baseLayer, single_as_alpha=single_as_alpha)
            for layer_name in top_layers:
                if getattr(obj.sx2layers[layer_name], 'visibility'):
                    blendmode = getattr(obj.sx2layers[layer_name], 'blend_mode')
                    layeralpha = getattr(obj.sx2layers[layer_name], 'opacity')
                    topcolors = self.get_layer(obj, obj.sx2layers[layer_name], single_as_alpha=single_as_alpha)
                    basecolors = tools.blend_values(topcolors, basecolors, blendmode, layeralpha)

            self.set_layer(obj, basecolors, resultLayer)
        # bpy.context.view_layer.objects.active = active


    def merge_layers(self, objs, toplayer, baselayer, targetlayer):
        for obj in objs:
            basecolors = self.get_layer(obj, baselayer, apply_layer_opacity=True, single_as_alpha=True)
            topcolors = self.get_layer(obj, toplayer, apply_layer_opacity=True, single_as_alpha=True)

            blendmode = toplayer.blend_mode
            colors = tools.blend_values(topcolors, basecolors, blendmode, 1.0)
            self.set_layer(obj, colors, targetlayer)

        for obj in objs:
            setattr(obj.sx2layers[targetlayer.name], 'visibility', True)
            setattr(obj.sx2layers[targetlayer.name], 'blend_mode', 'ALPHA')
            setattr(obj.sx2layers[targetlayer.name], 'opacity', 1.0)
            obj.sx2.selectedlayer = utils.find_layer_index_by_name(obj, targetlayer.name)


    def paste_layer(self, objs, targetlayer, fillmode):
        utils.mode_manager(objs, set_mode=True, mode_id='paste_layer')

        if fillmode == 'mask':
            for obj in objs:
                colors = layers.get_layer(obj, targetlayer)
                alphas = sxglobals.copy_buffer[obj.name]
                if alphas:
                    count = len(colors)//4
                    for i in range(count):
                        colors[(3+i*4):(4+i*4)] = alphas[(3+i*4):(4+i*4)]
                    layers.set_layer(obj, colors, targetlayer)
        elif fillmode == 'lumtomask':
            for obj in objs:
                colors = layers.get_layer(obj, targetlayer)
                values = sxglobals.copy_buffer[obj.name]
                if values:
                    alphas = convert.colors_to_values(values)
                    count = len(colors)//4
                    for i in range(count):
                        colors[3+i*4] = alphas[i]
                    layers.set_layer(obj, colors, targetlayer)
        else:
            for obj in objs:
                colors = sxglobals.copy_buffer[obj.name]
                if colors:
                    targetvalues = self.get_layer(obj, targetlayer)
                    if sxglobals.mode == 'EDIT':
                        colors = generate.mask_list(obj, colors)
                    colors = tools.blend_values(colors, targetvalues, 'ALPHA', 1.0)
                    layers.set_layer(obj, colors, targetlayer)

        utils.mode_manager(objs, set_mode=False, mode_id='paste_layer')


    def color_layers_to_swatches(self, objs):
        scene = bpy.context.scene.sx2

        for obj in objs:
            for i in range(5):
                for layer in obj.sx2layers:
                    if layer.paletted and (layer.palette_index == i):
                        palettecolor = utils.find_colors_by_frequency(objs, layer.name, numcolors=1, obj_sel_override=True)[0]
                        tabcolor = getattr(scene, 'newpalette' + str(i))

                        if not utils.color_compare(palettecolor, tabcolor) and not utils.color_compare(palettecolor, (0.0, 0.0, 0.0, 1.0)):
                            setattr(scene, 'newpalette' + str(i), palettecolor)


    def material_layers_to_swatches(self, objs):
        scene = bpy.context.scene.sx2
        layers = [objs[0].sx2.selectedlayer, 12, 13]

        for i, idx in enumerate(layers):
            layer = objs[0].sx2layers[idx]
            if sxglobals.mode == 'EDIT':
                masklayer = None
            else:
                masklayer = utils.find_layer_by_stack_index(objs[0], layers[0])
            layercolors = utils.find_colors_by_frequency(objs, layer.name, 8, masklayer=masklayer)
            tabcolor = getattr(scene, 'newmaterial' + str(i))

            if tabcolor in layercolors:
                palettecolor = tabcolor
            else:
                palettecolor = layercolors[0]

            if not utils.color_compare(palettecolor, tabcolor):
                setattr(scene, 'newmaterial' + str(i), palettecolor)


    def __del__(self):
        print('SX Tools: Exiting layers')


# ------------------------------------------------------------------------
#    Tool Actions
# ------------------------------------------------------------------------
class SXTOOLS2_tools(object):
    def __init__(self):
        return None


    def blend_values(self, topcolors, basecolors, blendmode, blendvalue, selectionmask=None):
        if topcolors is None:
            return basecolors
        else:
            if basecolors is None:
                basecolors = [0.0] * len(topcolors)

            count = len(topcolors)//4
            colors = [None] * count * 4
            midpoint = 0.5  # convert.srgb_to_linear([0.5, 0.5, 0.5, 1.0])[0]

            for i in range(count):
                top = Vector(topcolors[(0+i*4):(4+i*4)])
                base = Vector(basecolors[(0+i*4):(4+i*4)])
                a = top[3] * blendvalue

                if blendmode == 'ALPHA':
                    base = top * a + base * (1 - a)
                    base[3] = min(base[3]+a, 1.0)

                elif blendmode == 'ADD':
                    base += top * a
                    base[3] = min(base[3]+a, 1.0)

                elif blendmode == 'SUB':
                    base -= top * a
                    base[3] = min(base[3]+a, 1.0)

                elif blendmode == 'MUL':
                    for j in range(3):
                        base[j] *= top[j] * a + (1 - a)

                # Screen: (A+B) - (A*B)
                # elif blendmode == 'SCR':
                #     base = (base + top * a) - (base * top * a)

                elif blendmode == 'OVR':
                    over = Vector([0.0, 0.0, 0.0, top[3]])
                    b = over[3] * blendvalue
                    for j in range(3):
                        if base[j] < midpoint:
                            over[j] = 2.0 * base[j] * top[j]
                        else:
                            over[j] = 1.0 - 2.0 * (1.0 - base[j]) * (1.0 - top[j])
                    base = over * b + base * (1.0 - b)
                    base[3] = min(base[3]+a, 1.0)

                elif blendmode == 'REP':
                    if (selectionmask is None) or (selectionmask[i] != 0.0):
                        base = top

                elif blendmode == 'PAL':
                    alpha = base[3]
                    base = top * blendvalue + base * (1 - blendvalue)
                    base[3] = alpha

                # if (base[3] == 0.0) and (blendmode != 'REP'):
                #     base = [0.0, 0.0, 0.0, 0.0]

                colors[(0+i*4):(4+i*4)] = base
            return colors


    # NOTE: Make sure color parameter is always provided in linear color space!
    def apply_tool(self, objs, targetlayer, masklayer=None, invertmask=False, color=None, channel=None):
        if sxglobals.benchmark_tool:
            then = time.perf_counter()

        utils.mode_manager(objs, set_mode=True, mode_id='apply_tool')
        scene = bpy.context.scene.sx2
        amplitude = scene.noiseamplitude
        offset = scene.noiseoffset
        mono = scene.noisemono
        blendvalue = scene.toolopacity
        blendmode = scene.toolblend
        rampmode = scene.rampmode
        if sxglobals.mode == 'EDIT':
            mergebbx = False
        else:
            mergebbx = scene.rampbbox

        for obj in objs:
            if (masklayer is None) and (targetlayer.locked):
                masklayer = targetlayer

            # Get colorbuffer
            if color is not None:
                colors = generate.color_list(obj, color, masklayer)
            elif scene.toolmode == 'COL':
                colors = generate.color_list(obj, scene.fillcolor, masklayer)
            elif scene.toolmode == 'GRD':
                colors = generate.ramp_list(obj, objs, rampmode, masklayer, mergebbx)
            elif scene.toolmode == 'NSE':
                colors = generate.noise_list(obj, amplitude, offset, mono, masklayer)
            elif scene.toolmode == 'CRV':
                colors = generate.curvature_list(obj, objs, masklayer)
            elif scene.toolmode == 'OCC':
                colors = generate.occlusion_list(obj, scene.occlusionrays, scene.occlusionblend, scene.occlusiondistance, scene.occlusiongroundplane, scene.occlusiongroundheight, masklayer)
            elif scene.toolmode == 'THK':
                colors = generate.thickness_list(obj, scene.occlusionrays, masklayer)
            elif scene.toolmode == 'DIR':
                colors = generate.direction_list(obj, masklayer)
            elif scene.toolmode == 'LUM':
                colors = generate.luminance_remap_list(obj, targetlayer, masklayer)
            elif scene.toolmode == 'BLR':
                colors = generate.blur_list(obj, targetlayer, masklayer)
            elif scene.toolmode == 'EMI':
                colors = generate.emission_list(obj, scene.occlusionrays, masklayer)

            if colors is not None:
                mask = None
                if (blendmode == 'REP') and (sxglobals.mode == 'EDIT'):
                    bpy.context.tool_settings.mesh_select_mode[2] = False
                    mask, empty = generate.get_selection_mask(obj)

                if channel:
                    grayscales = convert.colors_to_values(colors, as_rgba=True)
                    values = layers.get_layer_mask(obj, targetlayer, channel)[0]
                    target_grayscales = convert.values_to_colors(values)

                    target_grayscales = self.blend_values(grayscales, target_grayscales, blendmode, blendvalue, selectionmask=mask)

                    target_values = convert.colors_to_values(target_grayscales)
                    layers.set_channel(obj, targetlayer.color_attribute, target_values, channel)
                else:
                    target_colors = layers.get_layer(obj, targetlayer)
                    colors = self.blend_values(colors, target_colors, blendmode, blendvalue, selectionmask=mask)

                    layers.set_layer(obj, colors, targetlayer)

        utils.mode_manager(objs, set_mode=False, mode_id='apply_tool')
        if sxglobals.benchmark_tool:
            now = time.perf_counter()
            print(f'Apply tool {scene.toolmode} duration: {round(now-then, 4)} seconds')


    # mode 0: hue, mode 1: saturation, mode 2: lightness
    def apply_hsl(self, objs, layer, hslmode, newValue):
        utils.mode_manager(objs, set_mode=True, mode_id='apply_hsl')

        colors = utils.find_colors_by_frequency(objs, layer.name)
        if colors:
            value = 0
            for color in colors:
                hsl = convert.rgb_to_hsl(color)
                modevalue = hsl[hslmode]
                value = modevalue if modevalue > value else value
            offset = newValue - value
        else:
            offset = newValue

        for obj in objs:
            colors = layers.get_layer(obj, layer)
            colors = generate.mask_list(obj, colors)
            if colors is not None:
                count = len(colors)//4
                for i in range(count):
                    color = colors[(0+i*4):(3+i*4)]
                    hsl = convert.rgb_to_hsl(color)
                    hsl[hslmode] += offset
                    rgb = convert.hsl_to_rgb(hsl)
                    colors[(0+i*4):(3+i*4)] = [rgb[0], rgb[1], rgb[2]]
                target_colors = layers.get_layer(obj, layer)
                colors = self.blend_values(colors, target_colors, 'ALPHA', 1.0)
                layers.set_layer(obj, colors, layer)

        utils.mode_manager(objs, set_mode=False, mode_id='apply_hsl')


    def preview_palette(self, objs, palette):
        utils.mode_manager(objs, set_mode=True, mode_id='preview_palette')
        scene = bpy.context.scene

        for i in range(5):
            palette_color = getattr(scene.sx2palettes[palette], 'color'+str(i), (0.0, 0.0, 0.0, 0.0))
            setattr(scene.sx2, 'newpalette'+str(i), palette_color)

        for obj in objs:
            obj.sx2.palettedshading = 1.0

        utils.mode_manager(objs, set_mode=False, mode_id='preview_palette')


    def apply_palette(self, objs, palette):
        if not sxglobals.refresh_in_progress:
            sxglobals.refresh_in_progress = True
            utils.mode_manager(objs, set_mode=True, mode_id='apply_palette')

            scene = bpy.context.scene.sx2
            palette = [
                scene.newpalette0,
                scene.newpalette1,
                scene.newpalette2,
                scene.newpalette3,
                scene.newpalette4]

            for obj in objs:
                if obj.sx2layers:
                    for layer in obj.sx2layers:
                        if layer.paletted:
                            source_colors = layers.get_layer(obj, layer)
                            colors = generate.color_list(obj, palette[layer.palette_index], masklayer=layer)
                            if colors is not None:
                                values = tools.blend_values(colors, source_colors, 'PAL', obj.sx2.palettedshading)
                                layers.set_layer(obj, values, layer)

            utils.mode_manager(objs, set_mode=False, mode_id='apply_palette')
            sxglobals.refresh_in_progress = False


    def apply_material(self, objs, targetlayer, material):
        if not sxglobals.refresh_in_progress:
            sxglobals.refresh_in_progress = True
            utils.mode_manager(objs, set_mode=True, mode_id='apply_material')

            material = bpy.context.scene.sx2materials[material]
            scene = bpy.context.scene.sx2

            setattr(scene, 'newmaterial0', material.color0)
            setattr(scene, 'newmaterial1', material.color1)
            setattr(scene, 'newmaterial2', material.color2)

            utils.mode_manager(objs, set_mode=False, mode_id='apply_material')
            sxglobals.refresh_in_progress = False


    def select_color_mask(self, objs, color, invertmask=False):
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        utils.clear_component_selection(objs)

        if objs[0].sx2.shadingmode == 'XRAY':
            objs[0].sx2.shadingmode = 'FULL'

        if objs[0].sx2.shadingmode == 'FULL':
            export.composite_color_layers(objs)

        for obj in objs:
            if objs[0].sx2.shadingmode == 'FULL':
                colors = layers.get_layer(obj, obj.sx2layers['Composite'])
            elif objs[0].sx2.shadingmode == 'CHANNEL':
                values = layers.get_layer_mask(obj, obj.sx2layers[obj.sx2.selectedlayer])[0]
                colors = convert.values_to_colors(values)
            else:
                colors = layers.get_layer(obj, obj.sx2layers[obj.sx2.selectedlayer])
            mesh = obj.data

            if bpy.context.tool_settings.mesh_select_mode[2]:
                for poly in mesh.polygons:
                    poly.select = any(utils.color_compare(colors[(0+loop_idx*4):(4+loop_idx*4)], color, 0.01) ^ invertmask for loop_idx in poly.loop_indices)
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        mesh.vertices[vert_idx].select = utils.color_compare(colors[(0+loop_idx*4):(4+loop_idx*4)], color, 0.01) ^ invertmask

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)


    def select_mask(self, objs, layer, invertmask=False):
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = objs[0]
        utils.clear_component_selection(objs)
        select_mode = bpy.context.tool_settings.mesh_select_mode[2]
    
        for obj in objs:
            mesh = obj.data
            mask = layers.get_layer_mask(obj, layer)[0]
            
            if select_mode:
                for poly in mesh.polygons:
                    poly.select = any((mask[loop_idx] > 0.0) ^ invertmask for loop_idx in poly.loop_indices)
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        mesh.vertices[vert_idx].select = (mask[loop_idx] > 0.0) ^ invertmask

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.context.view_layer.objects.active = active


    def update_recent_colors(self, color):
        scene = bpy.context.scene.sx2
        palCols = [
            scene.fillpalette1[:],
            scene.fillpalette2[:],
            scene.fillpalette3[:],
            scene.fillpalette4[:],
            scene.fillpalette5[:],
            scene.fillpalette6[:],
            scene.fillpalette7[:],
            scene.fillpalette8[:]]
        color_list = [
            color, palCols[0], palCols[1], palCols[2],
            palCols[3], palCols[4], palCols[5], palCols[6]]

        fillColor = color[:]
        if (fillColor not in palCols) and (fillColor[3] > 0.0):
            for i in range(8):
                setattr(scene, 'fillpalette' + str(i + 1), color_list[i])


    def __del__(self):
        print('SX Tools: Exiting tools')


# ------------------------------------------------------------------------
#    Modifier Module
# ------------------------------------------------------------------------
class SXTOOLS2_modifiers(object):
    def __init__(self):
        return None


    def assign_set(self, objs, setvalue, setmode):
        mode = objs[0].mode[:]
        mode_dict = {'CRS': 'crease', 'BEV': 'bevel_weight'}
        weight = setvalue
        if (weight == -1.0):
            weight = 0.0

        utils.mode_manager(objs, set_mode=True, mode_id='assign_set')
        active = bpy.context.view_layer.objects.active

        # Create necessary data layers
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            mesh = obj.data

            if 'crease_vert' not in mesh.attributes:
                mesh.vertex_creases_ensure()
            if 'crease_edge' not in mesh.attributes:
                mesh.edge_creases_ensure()
            if 'bevel_weight_vert' not in mesh.attributes:
                mesh.attributes.new(name='bevel_weight_vert', type='FLOAT', domain='POINT')
            if 'bevel_weight_edge' not in mesh.attributes:
                mesh.attributes.new(name='bevel_weight_edge', type='FLOAT', domain='EDGE')
            if 'sharp_edge' not in mesh.attributes:
                mesh.attributes.new(name='sharp_edge', type='BOOLEAN', domain='EDGE')
            if 'sharp_face' not in mesh.attributes:
                mesh.attributes.new(name='sharp_face', type='BOOLEAN', domain='FACE')

            if (mode == 'EDIT'):
                # EDIT mode vertex creasing
                # vertex beveling not supported by SX Tools modifier stack
                # a second bevel modifier would need to be added before subdivision modifier
                if (bpy.context.tool_settings.mesh_select_mode[0]):
                    weight_values = [None] * len(mesh.vertices)
                    select_values = [None] * len(mesh.vertices)
                    mesh.vertices.foreach_get('select', select_values)

                    mode_dict = {'CRS': 'crease_vert', 'BEV': 'bevel_weight_vert'}
                    mesh.attributes[mode_dict[setmode]].data.foreach_get('value', weight_values)

                    for i, sel in enumerate(select_values):
                        if sel:
                            weight_values[i] = weight

                    mode_dict = {'CRS': 'crease_vert', 'BEV': 'bevel_weight_vert'}
                    mesh.attributes[mode_dict[setmode]].data.foreach_set('value', weight_values)

                # EDIT mode edge creasing and beveling
                else:
                    weight_values = [None] * len(mesh.edges)
                    sharp_values = [None] * len(mesh.edges)
                    select_values = [None] * len(mesh.edges)
                    autocrease_values = [None] * len(mesh.edges)
                    mesh.edges.foreach_get('select', select_values)

                    mode_dict = {'CRS': 'crease_edge', 'BEV': 'bevel_weight_edge'}
                    mesh.attributes[mode_dict[setmode]].data.foreach_get('value', weight_values)
                    mesh.attributes['sharp_edge'].data.foreach_get('value', sharp_values)
                    if (bpy.context.scene.sx2.autocrease) and (setmode == 'BEV'):
                        mesh.attributes['crease_edge'].data.foreach_get('value', autocrease_values)

                    for i, sel in enumerate(select_values):
                        if sel:
                            weight_values[i] = weight
                            if setmode == 'CRS':
                                sharp_values[i] = (weight == 1.0) * (bpy.context.scene.sx2.autocrease)
                            else:
                                sharp_values[i] = (weight > 0.0) * (obj.sx2.hardmode == 'SHARP')
                                if (bpy.context.scene.sx2.autocrease) and (setmode == 'BEV') and (weight_values[i] > 0.0):
                                    autocrease_values[i] = 1.0

                    mode_dict = {'CRS': 'crease_edge', 'BEV': 'bevel_weight_edge'}
                    mesh.attributes['sharp_edge'].data.foreach_set('value', sharp_values)
                    mesh.attributes[mode_dict[setmode]].data.foreach_set('value', weight_values)
                    if (bpy.context.scene.sx2.autocrease) and (setmode == 'BEV'):
                        mesh.attributes['crease_edge'].data.foreach_set('value', autocrease_values)

            # OBJECT mode
            else:
                # vertex creasing and beveling
                if (bpy.context.tool_settings.mesh_select_mode[0]):
                    weight_values = [weight] * len(mesh.vertices)
                    mode_dict = {'CRS': 'crease_vert', 'BEV': 'bevel_weight_vert'}
                    mesh.attributes[mode_dict[setmode]].data.foreach_set('value', weight_values)

                # edge creasing and beveling
                else:
                    weight_values = [weight] * len(mesh.edges)

                    if setmode == 'CRS':
                        sharp_values = [(weight == 1.0) * (bpy.context.scene.sx2.autocrease)] * len(mesh.edges)
                    else:
                        sharp_values = [(weight > 0.0) * (obj.sx2.hardmode == 'SHARP')] * len(mesh.edges)

                    mode_dict = {'CRS': 'crease_edge', 'BEV': 'bevel_weight_edge'}
                    mesh.attributes['sharp_edge'].data.foreach_set('value', sharp_values)
                    mesh.attributes[mode_dict[setmode]].data.foreach_set('value', weight_values)
                    if (bpy.context.scene.sx2.autocrease) and (setmode == 'BEV'):
                        mesh.attributes['crease_edge'].data.foreach_set('value', weight_values)

            mesh.update()
        utils.mode_manager(objs, set_mode=False, mode_id='assign_set')
        bpy.context.view_layer.objects.active = active


    def select_set(self, objs, setvalue, setmode, clearsel=False):
        weight = setvalue
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        if clearsel:
            utils.clear_component_selection(objs)

        for obj in objs:
            mesh = obj.data

            if bpy.context.tool_settings.mesh_select_mode[0]:
                mode_dict = {'CRS': 'crease_vert', 'BEV': 'bevel_weight_vert'}
                if mode_dict[setmode] in mesh.attributes:
                    for vert in mesh.vertices:
                        attr_weight = mesh.attributes[mode_dict[setmode]].data[vert.index].value
                        if math.isclose(attr_weight, weight, abs_tol=0.1):
                            vert.select = True
            else:
                mode_dict = {'CRS': 'crease_edge', 'BEV': 'bevel_weight_edge'}
                if mode_dict[setmode] in mesh.attributes:
                    for edge in mesh.edges:
                        attr_weight = mesh.attributes[mode_dict[setmode]].data[edge.index].value
                        if math.isclose(attr_weight, weight, abs_tol=0.1):
                            edge.select = True

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)


    def add_modifiers(self, objs):
        for obj in objs:
            if sxglobals.benchmark_modifiers:
                then = time.perf_counter()

            if 'sxMirror' not in obj.modifiers:
                obj.modifiers.new(type='MIRROR', name='sxMirror')
                obj.modifiers['sxMirror'].show_viewport = True if obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror else False
                obj.modifiers['sxMirror'].show_expanded = False
                obj.modifiers['sxMirror'].use_axis[0] = obj.sx2.xmirror
                obj.modifiers['sxMirror'].use_axis[1] = obj.sx2.ymirror
                obj.modifiers['sxMirror'].use_axis[2] = obj.sx2.zmirror
                obj.modifiers['sxMirror'].mirror_object = obj.sx2.mirrorobject if obj.sx2.mirrorobject is not None else None
                obj.modifiers['sxMirror'].use_clip = True
                obj.modifiers['sxMirror'].use_mirror_merge = True

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Mirror: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()

            if 'sxTiler' not in obj.modifiers:
                if 'sx_tiler' not in bpy.data.node_groups:
                    setup.create_tiler()
                tiler = obj.modifiers.new(type='NODES', name='sxTiler')
                tiler.node_group = bpy.data.node_groups['sx_tiler']
                tiler['Socket_1'] = obj.sx2.tile_offset
                tiler['Socket_3'] = obj.sx2.tile_neg_x
                tiler['Socket_4'] = obj.sx2.tile_pos_x
                tiler['Socket_5'] = obj.sx2.tile_neg_y
                tiler['Socket_6'] = obj.sx2.tile_pos_y
                tiler['Socket_7'] = obj.sx2.tile_neg_z
                tiler['Socket_8'] = obj.sx2.tile_pos_z
                tiler.show_viewport = False
                tiler.show_expanded = False

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Tiler: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()

            # if 'sxAO' not in obj.modifiers:
            #     if 'sx_ao' not in bpy.data.node_groups:
            #         setup.create_occlusion_network(10)
            #     ao = obj.modifiers.new(type='NODES', name='sxAO')
            #     ao.node_group = bpy.data.node_groups['sx_ao']

            if 'sxSubdivision' not in obj.modifiers:
                obj.modifiers.new(type='SUBSURF', name='sxSubdivision')
                obj.modifiers['sxSubdivision'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxSubdivision'].show_expanded = False
                obj.modifiers['sxSubdivision'].quality = 6
                obj.modifiers['sxSubdivision'].levels = obj.sx2.subdivisionlevel
                obj.modifiers['sxSubdivision'].uv_smooth = 'NONE'
                obj.modifiers['sxSubdivision'].show_only_control_edges = True
                obj.modifiers['sxSubdivision'].show_on_cage = True

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Subdivision: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()

            if 'sxBevel' not in obj.modifiers:
                obj.modifiers.new(type='BEVEL', name='sxBevel')
                obj.modifiers['sxBevel'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxBevel'].show_expanded = False
                obj.modifiers['sxBevel'].width = obj.sx2.bevelwidth
                obj.modifiers['sxBevel'].width_pct = obj.sx2.bevelwidth
                obj.modifiers['sxBevel'].segments = obj.sx2.bevelsegments
                obj.modifiers['sxBevel'].use_clamp_overlap = True
                obj.modifiers['sxBevel'].loop_slide = True
                obj.modifiers['sxBevel'].mark_sharp = False
                obj.modifiers['sxBevel'].harden_normals = True
                obj.modifiers['sxBevel'].offset_type = obj.sx2.beveltype
                obj.modifiers['sxBevel'].limit_method = 'WEIGHT'
                obj.modifiers['sxBevel'].miter_outer = 'MITER_ARC'

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Bevel: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()

            if 'sxWeld' not in obj.modifiers:
                obj.modifiers.new(type='WELD', name='sxWeld')
                obj.modifiers['sxWeld'].show_viewport = False if obj.sx2.weldthreshold == 0 else obj.sx2.modifiervisibility
                obj.modifiers['sxWeld'].show_expanded = False
                obj.modifiers['sxWeld'].merge_threshold = obj.sx2.weldthreshold

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Weld: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()

            if 'sxDecimate' not in obj.modifiers:
                obj.modifiers.new(type='DECIMATE', name='sxDecimate')
                obj.modifiers['sxDecimate'].show_viewport = False if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0) else obj.sx2.modifiervisibility
                obj.modifiers['sxDecimate'].show_expanded = False
                obj.modifiers['sxDecimate'].decimate_type = 'DISSOLVE'
                obj.modifiers['sxDecimate'].angle_limit = math.radians(obj.sx2.decimation)
                obj.modifiers['sxDecimate'].use_dissolve_boundaries = True
                obj.modifiers['sxDecimate'].delimit = {'SHARP', 'UV'}

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Decimate1: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()

            if 'sxDecimate2' not in obj.modifiers:
                obj.modifiers.new(type='DECIMATE', name='sxDecimate2')
                obj.modifiers['sxDecimate2'].show_viewport = False if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0) else obj.sx2.modifiervisibility
                obj.modifiers['sxDecimate2'].show_expanded = False
                obj.modifiers['sxDecimate2'].decimate_type = 'COLLAPSE'
                obj.modifiers['sxDecimate2'].ratio = 0.99
                obj.modifiers['sxDecimate2'].use_collapse_triangulate = True

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Decimate2: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()

            if 'sxSmoothNormals' not in obj.modifiers:
                root = os.path.dirname(bpy.app.binary_path)
                blend_file_path = os.path.join(root, f'{sxglobals.version}.{sxglobals.minorversion}', 'datafiles', 'assets', 'geometry_nodes', 'smooth_by_angle.blend')
                node_group_name = "Smooth by Angle"
                node_group = bpy.data.node_groups.get(node_group_name)
                if not node_group:
                    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
                        data_to.node_groups = [name for name in data_from.node_groups if name == node_group_name]

                tiler = obj.modifiers.new(type='NODES', name='sxSmoothNormals')
                tiler.node_group = bpy.data.node_groups['Smooth by Angle']
                # bpy.ops.object.modifier_add_node_group(asset_library_type='ESSENTIALS', asset_library_identifier="", relative_asset_identifier="geometry_nodes\\smooth_by_angle.blend\\NodeTree\\Smooth by Angle")
                obj.modifiers['sxSmoothNormals'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxSmoothNormals'].show_expanded = False
                obj.modifiers["sxSmoothNormals"]["Input_1"] = math.radians(obj.sx2.smoothangle)

            if 'sxWeightedNormal' not in obj.modifiers:
                obj.modifiers.new(type='WEIGHTED_NORMAL', name='sxWeightedNormal')
                obj.modifiers['sxWeightedNormal'].show_viewport = obj.sx2.modifiervisibility if obj.sx2.weightednormals else False
                obj.modifiers['sxWeightedNormal'].show_expanded = False
                obj.modifiers['sxWeightedNormal'].mode = 'FACE_AREA_WITH_ANGLE'
                obj.modifiers['sxWeightedNormal'].weight = 50
                obj.modifiers['sxWeightedNormal'].keep_sharp = False if obj.sx2.hardmode == 'SMOOTH' else True

                # obj.sx2.smoothangle = obj.sx2.smoothangle

                if sxglobals.benchmark_modifiers:
                    now = time.perf_counter()
                    print(f'SX Tools: Weighted Normals: {obj.name} {round(now-then, 4)} seconds')
                    then = time.perf_counter()


    def apply_modifiers(self, objs):
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            if 'sxMirror' in obj.modifiers:
                if obj.modifiers['sxMirror'].show_viewport is False:
                    bpy.ops.object.modifier_remove(modifier='sxMirror')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxMirror')
            if 'sxTiler' in obj.modifiers:
                bpy.ops.object.modifier_remove(modifier='sxTiler')
            if 'sxAO' in obj.modifiers:
                bpy.ops.object.modifier_apply(modifier='sxAO')
            if 'sxSubdivision' in obj.modifiers:
                if obj.modifiers['sxSubdivision'].levels == 0:
                    bpy.ops.object.modifier_remove(modifier='sxSubdivision')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxSubdivision')
            if 'sxBevel' in obj.modifiers:
                if obj.sx2.bevelsegments == 0:
                    bpy.ops.object.modifier_remove(modifier='sxBevel')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxBevel')
            if 'sxWeld' in obj.modifiers:
                if obj.sx2.weldthreshold == 0:
                    bpy.ops.object.modifier_remove(modifier='sxWeld')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxWeld')
            if 'sxDecimate' in obj.modifiers:
                if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                    bpy.ops.object.modifier_remove(modifier='sxDecimate')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxDecimate')
            if 'sxDecimate2' in obj.modifiers:
                if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                    bpy.ops.object.modifier_remove(modifier='sxDecimate2')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxDecimate2')
            if 'sxSmoothNormals' in obj.modifiers:
                if obj.sx2.smoothangle == 0:
                    bpy.ops.object.modifier_remove(modifier='sxSmoothNormals')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxSmoothNormals')
            if 'sxWeightedNormal' in obj.modifiers:
                if not obj.sx2.weightednormals:
                    bpy.ops.object.modifier_remove(modifier='sxWeightedNormal')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxWeightedNormal')


    def remove_modifiers(self, objs, reset=False):
        modifiers = ['Auto Smooth', 'sxMirror', 'sxTiler', 'sxAO', 'sxGeometryNodes', 'sxSubdivision', 'sxBevel', 'sxWeld', 'sxDecimate', 'sxDecimate2', 'sxEdgeSplit', 'sxSmoothNormals', 'sxWeightedNormal']
        if reset and ('sx_tiler' in bpy.data.node_groups):
            bpy.data.node_groups.remove(bpy.data.node_groups['sx_tiler'], do_unlink=True)
        if reset and ('sx_ao' in bpy.data.node_groups):
            bpy.data.node_groups.remove(bpy.data.node_groups['sx_ao'], do_unlink=True)

        for obj in objs:
            for modifier in modifiers:
                if modifier in obj.modifiers:
                    obj.modifiers.remove(obj.modifiers.get(modifier))

            # obj.sx2.smoothangle = obj.sx2.smoothangle

            if reset:
                obj.sx2.xmirror = False
                obj.sx2.ymirror = False
                obj.sx2.zmirror = False
                obj.sx2.mirrorobject = None
                obj.sx2.autocrease = True
                obj.sx2.hardmode = 'SHARP'
                obj.sx2.beveltype = 'WIDTH'
                obj.sx2.bevelsegments = 2
                obj.sx2.bevelwidth = 0.05
                obj.sx2.subdivisionlevel = 1
                obj.sx2.smoothangle = 180.0
                obj.sx2.weldthreshold = 0.0
                obj.sx2.decimation = 0.0


    def calculate_triangles(self, objs):
        count = 0
        edg = bpy.context.evaluated_depsgraph_get()
        utils.mode_manager(objs, set_mode=True, mode_id='calculate_triangles')
        for obj in objs:
            if obj.type == 'MESH':
                eval_data = obj.evaluated_get(edg).data
                for face in eval_data.polygons:
                    count += len(face.vertices) - 2
        utils.mode_manager(objs, set_mode=False, mode_id='calculate_triangles')

        return str(count)


    def __del__(self):
        print('SX Tools: Exiting modifiers')


# ------------------------------------------------------------------------
#    Export Module
# ------------------------------------------------------------------------
class SXTOOLS2_export(object):
    def __init__(self):
        return None


    def composite_color_layers(self, objs):
        utils.mode_manager(objs, set_mode=True, mode_id='composite_color_layers')
        cmp_add_objs = [obj for obj in objs if 'Composite' not in obj.sx2layers.keys()]
        if cmp_add_objs:
            layers.add_layer(cmp_add_objs, name='Composite', layer_type='CMP')

        for obj in objs:
            comp_layers = utils.find_color_layers(obj, staticvertexcolors=int(obj.sx2.staticvertexcolors))
            layers.blend_layers([obj, ], comp_layers, comp_layers[0], obj.sx2layers['Composite'])

        utils.mode_manager(objs, set_mode=False, mode_id='composite_color_layers')


    def smart_separate(self, objs, override=False, parent=True):
        if objs:
            mirror_pairs = [('_top', '_bottom'), ('_front', '_rear'), ('_left', '_right'), ('_bottom', '_top'), ('_rear', '_front'), ('_right', '_left')]
            prefs = bpy.context.preferences.addons['sxtools2'].preferences
            scene = bpy.context.scene.sx2
            view_layer = bpy.context.view_layer
            mode = objs[0].mode
            objs = objs[:]
            separated_objs = []

            export_objects = utils.create_collection('ExportObjects')
            source_objects = utils.create_collection('SourceObjects')

            if override:
                sep_objs = objs
            else:
                sep_objs = [obj for obj in objs if obj.sx2.smartseparate]

            if sep_objs:
                for obj in sep_objs:
                    if (scene.exportquality == 'LO') and (obj.name not in source_objects.objects.keys()) and (obj.name not in export_objects.objects.keys()) and (obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror):
                        source_objects.objects.link(obj)

                active = view_layer.objects.active
                bpy.ops.object.select_all(action='DESELECT')
                for obj in sep_objs:
                    obj.select_set(True)
                    view_layer.objects.active = obj
                    ref_objs = view_layer.objects[:]
                    orgname = obj.name[:]
                    xmirror = obj.sx2.xmirror
                    ymirror = obj.sx2.ymirror
                    zmirror = obj.sx2.zmirror

                    if ('sxMirror' in obj.modifiers) and (obj.modifiers['sxMirror'].mirror_object is not None):
                        ref_loc = obj.modifiers['sxMirror'].mirror_object.matrix_world.to_translation()
                        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                        bpy.ops.object.modifier_apply(modifier='sxMirror')
                    elif obj.sx2.mirrorobject is not None:
                        ref_loc = obj.sx2.mirrorobject.matrix_world.to_translation()
                    else:
                        ref_loc = obj.matrix_world.to_translation()

                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.mesh.separate(type='LOOSE')

                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    new_obj_list = [view_layer.objects[orgname], ]

                    for vl_obj in view_layer.objects:
                        if vl_obj not in ref_objs:
                            new_obj_list.append(vl_obj)
                            export_objects.objects.link(vl_obj)

                    if len(new_obj_list) > 1:
                        export.set_pivots(new_obj_list)  # , force=True)
                        suffixDict = {}
                        for new_obj in new_obj_list:
                            view_layer.objects.active = new_obj
                            zstring = ystring = xstring = ''
                            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([new_obj, ])
                            pivot_loc = new_obj.matrix_world.to_translation()

                            # print(f'SX Tools: {new_obj.name} pivot: {pivot_loc} ref: {ref_loc}')

                            # 1) compare bounding box max to reference pivot
                            # 2) flip result with xor according to SX Tools prefs
                            # 3) look up matching sub-string from mirror_pairs
                            if zmirror and (abs(ref_loc[2] - pivot_loc[2]) > 0.001):
                                zstring = mirror_pairs[0][int(zmax < ref_loc[2])]
                            if ymirror and (abs(ref_loc[1] - pivot_loc[1]) > 0.001):
                                ystring = mirror_pairs[1][int((ymax < ref_loc[1]) ^ prefs.flipsmarty)]
                            if xmirror and (abs(ref_loc[0] - pivot_loc[0]) > 0.001):
                                xstring = mirror_pairs[2][int((xmax < ref_loc[0]) ^ prefs.flipsmartx)]

                            if len(new_obj_list) > 2 ** (zmirror + ymirror + xmirror):
                                if zstring+ystring+xstring not in suffixDict:
                                    suffixDict[zstring+ystring+xstring] = 0
                                else:
                                    suffixDict[zstring+ystring+xstring] += 1
                                new_obj.name = orgname + str(suffixDict[zstring+ystring+xstring]) + zstring + ystring + xstring
                            else:
                                new_obj.name = orgname + zstring + ystring + xstring

                            new_obj.data.name = new_obj.name + '_mesh'

                    separated_objs.extend(new_obj_list)

                # Parent new children to their matching parents
                if parent:
                    for obj in separated_objs:
                        for mirror_pair in mirror_pairs:
                            if mirror_pair[0] in obj.name:
                                if mirror_pair[1] in obj.parent.name:
                                    new_parent_name = obj.parent.name.replace(mirror_pair[1], mirror_pair[0])
                                    obj.parent = view_layer.objects[new_parent_name]
                                    obj.matrix_parent_inverse = obj.parent.matrix_world.inverted()

                view_layer.objects.active = active
            bpy.ops.object.mode_set(mode=mode)
            return separated_objs

    # NOTE: In case of bevels, prefer even-numbered segment counts!
    #       Odd-numbered bevels often generate incorrect vertex colors
    def generate_lods(self, objs):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        org_obj_list = objs[:]
        name_list = []
        new_obj_list = []
        active_obj = bpy.context.view_layer.objects.active
        scene = bpy.context.scene.sx2

        xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box(org_obj_list)
        bbxheight = zmax - zmin

        # Make sure source and export Collections exist
        export_objects = utils.create_collection('ExportObjects')
        source_objects = utils.create_collection('SourceObjects')

        name_list = [(obj.name[:], obj.data.name[:]) for obj in org_obj_list]

        for i in range(3):
            print(f'SX Tools: Generating LOD {i}')
            if i == 0:
                for obj in org_obj_list:
                    obj.data.name = obj.data.name + '_LOD' + str(i)
                    obj.name = obj.name + '_LOD' + str(i)
                    new_obj_list.append(obj)
                    if scene.exportquality == 'LO':
                        source_objects.objects.link(obj)
            else:
                for j, obj in enumerate(org_obj_list):
                    lod_values = [(obj.sx2.lod1_bevels, obj.sx2.lod1_subdiv), (obj.sx2.lod2_bevels, obj.sx2.lod2_subdiv)]
                    new_obj = obj.copy()
                    new_obj.data = obj.data.copy()

                    new_obj.data.name = name_list[j][1] + '_LOD' + str(i)
                    new_obj.name = name_list[j][0] + '_LOD' + str(i)

                    new_obj.location += Vector((0.0, 0.0, (bbxheight+prefs.lodoffset)*i))

                    bpy.context.scene.collection.objects.link(new_obj)
                    export_objects.objects.link(new_obj)

                    new_obj.parent = bpy.context.view_layer.objects[obj.parent.name]

                    bpy.ops.object.select_all(action='DESELECT')
                    new_obj.select_set(True)
                    bpy.context.view_layer.objects.active = new_obj

                    if obj.sx2.subdivisionlevel > lod_values[i-1][1]:
                        new_obj.sx2.subdivisionlevel = lod_values[i-1][1]
                        new_obj.modifiers['sxSubdivision'].levels = lod_values[i-1][1]

                    if obj.sx2.bevelsegments > lod_values[i-1][0]:
                        new_obj.sx2.bevelsegments = lod_values[i-1][0]
                        new_obj.modifiers['sxBevel'].segments = lod_values[i-1][0]
                    
                    if new_obj.sx2.subdivisionlevel == 0:
                        new_obj.sx2.weldthreshold = 0
                        new_obj.modifiers['sxWeld'].merge_threshold = 0

                    new_obj_list.append(new_obj)

        # active_obj.select_set(True)
        bpy.context.view_layer.objects.active = active_obj

        return new_obj_list


    # Multi-pass convex hull generator
    def generate_hulls(self, objs, group=None):

        def sign(x):
            return 1 if x >= 0 else -1

        def shrink_colliders(collider_obj):
            collider_obj.select_set(True)
            collider_obj.sx2.smartseparate = False
            view_layer.objects.active = collider_obj
            if sxglobals.benchmark_cvx:
                print(f'SX Tools: {collider_obj.name} Initial Tri Count {modifiers.calculate_triangles([collider_obj, ])}')
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.convex_hull(use_existing_faces=False, face_threshold=math.radians(40.0), shape_threshold=math.radians(40.0), sharp=True)
            if sxglobals.benchmark_cvx:
                print(f'SX Tools: {collider_obj.name} Convex Tri Count {modifiers.calculate_triangles([collider_obj, ])}')
            tri_opt = False
            angle = 0.0

            # Shrink hull according to factor
            if collider_obj.sx2.collideroffsetfactor > 0.0:
                bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                offset = -1.0 * min(collider_obj.sx2.collideroffsetfactor, utils.find_safe_mesh_offset(collider_obj))
                bpy.ops.object.mode_set(mode='EDIT', toggle=False)
                bpy.context.tool_settings.mesh_select_mode = (True, False, False)
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.transform.shrink_fatten(value=offset)

                # Weld after shrink to clean up clumped verts
                # bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                # collider_obj.modifiers.new(type='WELD', name='hullWeld')
                # collider_obj.modifiers["hullWeld"].merge_threshold = min(collider_obj.dimensions) * 0.15
                # bpy.ops.object.modifier_apply(modifier='hullWeld')

            if sxglobals.benchmark_cvx:
                print(f'SX Tools: {collider_obj.name} Pre-Optimized Tri Count {modifiers.calculate_triangles([collider_obj, ])}')


            if int(modifiers.calculate_triangles([collider_obj, ])) > collider_obj.sx2.hulltrimax:
                # Decimate until below tri count limit
                if sxglobals.benchmark_cvx:
                    then = time.perf_counter()

                bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                collider_obj.modifiers.new(type='DECIMATE', name='hullDecimate')
                collider_obj.modifiers['hullDecimate'].show_viewport = True
                collider_obj.modifiers['hullDecimate'].show_expanded = False
                collider_obj.modifiers['hullDecimate'].decimate_type = 'DISSOLVE'
                collider_obj.modifiers['hullDecimate'].angle_limit = math.radians(angle)
                collider_obj.modifiers['hullDecimate'].use_dissolve_boundaries = False

                # Slow iterator
                # while (int(modifiers.calculate_triangles([collider_obj, ])) > collider_obj.sx2.hulltrimax) and (angle < 180.0):
                #     angle += 1.5   # 0.5
                #     collider_obj.modifiers['hullDecimate'].angle_limit = math.radians(angle)

                # Use binary search approach for decimation angle
                lower_bound = collider_obj.sx2.hulltrimax * collider_obj.sx2.hulltrifactor

                # Initialize search range
                low_angle = 0.0
                high_angle = 180.0
                angle = high_angle
                fallback_angle = high_angle
                best_result = collider_obj.sx2.hulltrimax
                fallback_result = 0
                epsilon = 0.5
                best_found = False

                while high_angle - low_angle > epsilon:
                    mid_angle = (low_angle + high_angle) / 2.0
                    collider_obj.modifiers['hullDecimate'].angle_limit = math.radians(mid_angle)
                    triangle_count = int(modifiers.calculate_triangles([collider_obj]))

                    if sxglobals.benchmark_cvx:
                        print(f'SX Tools: Low: {low_angle}, High: {high_angle}, Mid: {mid_angle}, Best: {angle}, Triangles: {triangle_count}')

                    if triangle_count > collider_obj.sx2.hulltrimax:
                        low_angle = mid_angle
                    elif (best_result >= triangle_count >= lower_bound):
                        best_found = True
                        angle = mid_angle
                        best_result = triangle_count
                        high_angle = mid_angle
                    elif (lower_bound > triangle_count > fallback_result):
                        fallback_angle = mid_angle
                        fallback_result = triangle_count
                        high_angle = mid_angle
                    else:
                        high_angle = mid_angle

                if not best_found:
                    angle = fallback_angle

                collider_obj.modifiers['hullDecimate'].angle_limit = math.radians(angle)
                bpy.ops.object.modifier_apply(modifier='hullDecimate')

                if sxglobals.benchmark_cvx:
                    now = time.perf_counter()
                    print(f'SX Tools: {collider_obj.name} Tris/limit: {modifiers.calculate_triangles([collider_obj, ])}/{collider_obj.sx2.hulltrimax} Angle: {angle} Iteration Time: {now-then}')

                tri_opt = True

            # Final convex conversion
            if tri_opt:
                pass
                # Weld after shrink to clean up clumped verts
                # bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                # collider_obj.modifiers.new(type='WELD', name='hullWeld')
                # collider_obj.modifiers["hullWeld"].merge_threshold = min(collider_obj.dimensions) * 0.15
                # bpy.ops.object.modifier_apply(modifier='hullWeld')

                bpy.ops.object.mode_set(mode='EDIT', toggle=False)
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.convex_hull(use_existing_faces=False, face_threshold=math.radians(angle), shape_threshold=math.radians(angle))
                bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

            bpy.ops.object.mode_set(mode='EDIT', toggle=False)
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(angle))
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

            if sxglobals.benchmark_cvx:
                print(f'SX Tools: {collider_obj.name} Tri Count {modifiers.calculate_triangles([collider_obj, ])}')


        if objs:
            view_layer = bpy.context.view_layer
            org_objs = []
            hull_objs = [obj for obj in objs if (obj.sx2.generatehulls and not obj.sx2.use_cids)]
            cid_objs = [obj for obj in objs if (obj.sx2.generatehulls and obj.sx2.use_cids)]

            if sxglobals.benchmark_cvx:
                then = time.perf_counter()
                print(f'SX Tools: Convex hull generation starting')

            if (hull_objs or cid_objs):
                new_objs = []
                new_cid_objs = []
                new_hull_objs = []
                active_obj = view_layer.objects.active
                colliders = utils.create_collection('SXColliders')
                if colliders.name not in bpy.context.scene.collection.children:
                    bpy.context.scene.collection.children.link(colliders)
                pivot_ref = {}

                # Create hull meshes via object copies or Collider IDs
                if cid_objs:
                    org_objs += cid_objs[:]
                    # Map color regions to mesh islands
                    color_to_faces = defaultdict(list)
                    color_ref_obj = {}

                    for obj in cid_objs:
                        if obj.type == 'MESH':
                            id_layer = obj.sx2layers['Collider IDs'].color_attribute
                            org_subdiv = obj.sx2.subdivisionlevel
                            if ('sxSubdivision' in obj.modifiers):
                                obj.modifiers['sxSubdivision'].levels = 1 if org_subdiv > 1 else org_subdiv

                            # Get evaluated mesh at subdiv 1
                            edg = bpy.context.evaluated_depsgraph_get()
                            temp_mesh = obj.evaluated_get(edg).to_mesh()
                            bm = bmesh.new()
                            bm.from_mesh(temp_mesh)
                            color_layer = bm.loops.layers.float_color[id_layer]
                            if ('sxSubdivision' in obj.modifiers):
                                obj.modifiers['sxSubdivision'].levels = org_subdiv
                            
                            for face in bm.faces:
                                color = face.loops[0][color_layer]
                                color = tuple(round(c, 2) for c in color)
                                # color = tuple(round(c * 20) / 20 for c in color)

                                # Create color buckets, discard uncolored faces
                                if (color != (0.0, 0.0, 0.0, 0.0)):
                                    transformed_face_verts = []
                                    first_obj_matrix_world_inv = obj.matrix_world.inverted()

                                    # Transform all color island vertices to the local space of reference object
                                    # (first source obj with respective color)
                                    for vert in face.verts:
                                        world_coord = obj.matrix_world @ vert.co
                                        local_coord_first_obj = first_obj_matrix_world_inv @ world_coord
                                        transformed_face_verts.append(tuple(local_coord_first_obj))

                                    # In case of preserved borders, check face centroid against mirror axes
                                    if obj.sx2.preserveborders:
                                        centroid = sum((Vector(vert) for vert in transformed_face_verts), Vector((0, 0, 0))) / len(transformed_face_verts)
                                        signs = [
                                                sign(centroid[0]) if obj.sx2.xmirror else 1,
                                                sign(centroid[1]) if obj.sx2.ymirror else 1,
                                                sign(centroid[2]) if obj.sx2.zmirror else 1
                                                ]

                                        color = tuple((c + 10) * s for c, s in zip(color, signs))

                                    if (color not in color_ref_obj):
                                        color_ref_obj[color] = (obj.location.copy(), obj.matrix_world.inverted(), obj.sx2.hulltrimax, obj.name)
                                    color_to_faces[color].append(transformed_face_verts)
                            
                            bm.free()
                            obj.evaluated_get(edg).to_mesh_clear()

                    if sxglobals.benchmark_cvx:
                        now = time.perf_counter()
                        print(f'SX Tools: Convex hull generation: faces mapped to color regions. Time: {now-then}')

                    # Create a new bmesh and mesh for each color group
                    for i, (color, faces) in enumerate(color_to_faces.items()):
                        new_bm = bmesh.new()
                        vert_map = {}  # Map old verts to new verts
                        
                        for face_verts in faces:
                            new_verts = []
                            
                            for vert_co in face_verts:
                                if vert_co not in vert_map:
                                    new_vert = new_bm.verts.new(vert_co)
                                    vert_map[vert_co] = new_vert

                                if vert_map[vert_co] not in new_verts:
                                    new_verts.append(vert_map[vert_co])
                            
                            if len(new_verts) > 2:
                                existing_face = False
                                for f in new_bm.faces:
                                    if set(f.verts[:]) == set(new_verts):
                                        existing_face = True
                                        break

                                if not existing_face:
                                    new_bm.faces.new(new_verts)
                        
                        new_bm.verts.index_update()
                        new_bm.edges.index_update()
                        new_bm.faces.index_update()

                        name = f'{color_ref_obj[color][3]}_hull_{i}'
                        mesh_data = bpy.data.meshes.new(name+'_mesh')
                        new_bm.to_mesh(mesh_data)
                        new_bm.free()

                        new_obj = bpy.data.objects.new(name, mesh_data)
                        bpy.context.scene.collection.objects.link(new_obj)
                        colliders.objects.link(new_obj)

                        # New meshes local to origin
                        # Proceed to smart separate manually and move to correct location
                        new_obj.sx2.smartseparate = False
                        new_obj.location = color_ref_obj[color][0]
                        new_obj.parent = group
                        new_obj.sx2.hulltrimax = color_ref_obj[color][2]
                        new_obj.sx2.mirrorobject = view_layer.objects[color_ref_obj[color][3]].sx2.mirrorobject
                        new_obj.sx2.xmirror = view_layer.objects[color_ref_obj[color][3]].sx2.xmirror
                        new_obj.sx2.ymirror = view_layer.objects[color_ref_obj[color][3]].sx2.ymirror
                        new_obj.sx2.zmirror = view_layer.objects[color_ref_obj[color][3]].sx2.zmirror
                        new_obj.sx2.separate_cids = view_layer.objects[color_ref_obj[color][3]].sx2.separate_cids
                        new_obj.sx2.mergefragments = view_layer.objects[color_ref_obj[color][3]].sx2.mergefragments
                        new_obj.sx2.preserveborders = view_layer.objects[color_ref_obj[color][3]].sx2.preserveborders
                        new_obj.sx2.collideroffsetfactor = view_layer.objects[color_ref_obj[color][3]].sx2.collideroffsetfactor
                        new_obj.sx2.pivotmode = 'CID'
                        new_obj['group_id'] = view_layer.objects[color_ref_obj[color][3]].parent.name

                        # Set pivots manually for split convex hulls
                        if bpy.data.objects[color_ref_obj[color][3]].sx2.smartseparate:
                            ref_obj = bpy.data.objects[color_ref_obj[color][3]]
                            pivot_obj = ref_obj.copy()
                            pivot_obj.data = ref_obj.data.copy()

                            pivot_obj.sx2.weldthreshold = 0.0
                            pivot_obj.sx2.decimation = 0.0

                            bpy.context.scene.collection.objects.link(pivot_obj)
                            view_layer.objects.active = pivot_obj

                            if 'sxMirror' in pivot_obj.modifiers:
                                pivot_obj.modifiers.remove(pivot_obj.modifiers.get('sxMirror'))
                            modifiers.apply_modifiers([pivot_obj, ])
                            self.set_pivots([pivot_obj, ])
                            pivot_loc = list(pivot_obj.matrix_world.to_translation())

                            # Fix pivot location for object halves
                            adjustables = ['MASS', 'BBOX']
                            if new_obj.sx2.mirrorobject:
                                mirror_pos = new_obj.sx2.mirrorobject.matrix_world.to_translation()
                            else:
                                mirror_pos = new_obj.parent.matrix_world.to_translation() if new_obj.parent else (0.0, 0.0, 0.0)

                            # Check if object has vertices at mirror axes
                            if pivot_obj.sx2.pivotmode in adjustables:
                                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([pivot_obj, ])

                                if pivot_obj.sx2.xmirror:
                                    for bound in [xmin, xmax]:
                                        if (abs(mirror_pos[0] - bound) < 0.001):
                                            pivot_loc[0] = mirror_pos[0]

                                if pivot_obj.sx2.ymirror:
                                    for bound in [ymin, ymax]:
                                        if (abs(mirror_pos[1] - bound) < 0.001):
                                            pivot_loc[1] = mirror_pos[1]

                                if pivot_obj.sx2.zmirror:
                                    for bound in [zmin, zmax]:
                                        if (abs(mirror_pos[2] - bound) < 0.001):
                                            pivot_loc[2] = mirror_pos[2]
                            
                            bpy.data.objects.remove(pivot_obj)
                        else:
                            pivot_loc = view_layer.objects[color_ref_obj[color][3]].matrix_world.to_translation()

                        view_layer.objects.active = new_obj
                        bpy.context.scene.cursor.location = pivot_loc
                        bpy.ops.object.select_all(action='DESELECT')
                        new_obj.select_set(True)
                        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
                        parent_pivot = new_obj.parent.matrix_world.to_translation() if new_obj.parent else (0.0, 0.0, 0.0)
                        pivot_ref[new_obj] = new_obj.sx2.mirrorobject.matrix_world.to_translation() if new_obj.sx2.mirrorobject else parent_pivot

                        new_obj.sx2.generatelodmeshes = False
                        new_obj.sx2.generateemissionmeshes = False
                        new_obj.sx2.weldthreshold = 0.0
                        new_obj.sx2.decimation = 0.0
                        new_cid_objs.append(new_obj)

                    if sxglobals.benchmark_cvx:
                        now = time.perf_counter()
                        print(f'SX Tools: Convex hull generation: collision ID objects created. Time: {now-then}')
                        
                if hull_objs:
                    org_objs = hull_objs[:]
                    for obj in org_objs:
                        new_obj = obj.copy()
                        new_obj.data = obj.data.copy()
                        new_obj.data.name = obj.name[:] + '_hull_mesh'
                        new_obj.name = obj.name[:] + '_hull'

                        bpy.context.scene.collection.objects.link(new_obj)
                        view_layer.objects.active = new_obj
                        colliders.objects.link(new_obj)

                        pivot_ref[new_obj] = new_obj.matrix_world.to_translation()

                        org_subdiv = obj.sx2.subdivisionlevel
                        if ('sxSubdivision' in obj.modifiers):
                            new_obj.sx2.subdivisionlevel = 1 if org_subdiv > 1 else org_subdiv
                            new_obj.modifiers['sxSubdivision'].levels = 1 if org_subdiv > 1 else org_subdiv

                        new_obj.sx2.smartseparate = obj.sx2.smartseparate
                        new_obj.sx2.generatelodmeshes = False
                        new_obj.sx2.generateemissionmeshes = False
                        new_obj.sx2.weldthreshold = 0.0
                        new_obj.sx2.decimation = 0.0
                        new_obj.parent = group
                        new_obj['group_id'] = obj.parent.name
                        new_hull_objs.append(new_obj)

                        # Clear existing modifier stacks
                        modifiers.apply_modifiers([new_obj, ])

                    if sxglobals.benchmark_cvx:
                        now = time.perf_counter()
                        print(f'SX Tools: Convex hull generation: hull objects created. Time: {now-then}')

                # Split mirrored collider source objects
                separated_objs = []
                for new_obj in new_cid_objs:
                    if (new_obj.sx2.preserveborders) or (not new_obj.sx2.separate_cids):
                        new_objs.append(new_obj)
                    else:
                        bpy.ops.object.select_all(action='DESELECT')
                        view_layer.objects.active = new_obj
                        sep_objs = self.smart_separate([new_obj, ], override=True, parent=False)
                        # print('Hull Mesh:', new_obj.name, 'Mergefragments:', new_obj.sx2.mergefragments)
                        # print('Separated:', [sep_obj.name for sep_obj in sep_objs])

                        if new_obj.sx2.mergefragments:
                            # Merge needlessly separated objects by analyzing their bbx centers
                            left, right, front, back, top, bottom, center_x, center_y, center_z = [], [], [], [], [], [], [], [], []

                            for i, sep_obj in enumerate(sep_objs):
                                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([sep_obj, ])
                                ref_pivot = pivot_ref[new_obj]
                                # print('New obj:', new_obj.name, ref_pivot)
                                # print('Sep obj:', sep_obj.name, (xmin + xmax * 0.5, ymin + ymax * 0.5, zmin + zmax * 0.5))

                                if new_obj.sx2.xmirror:
                                    if ((xmin + xmax) * 0.5 > ref_pivot[0]) and (abs(((xmin + xmax) * 0.5) - abs(ref_pivot[0])) > 0.01):
                                        # print('Adding', sep_obj.name, 'left')
                                        left.append(sep_obj)
                                    elif ((xmin + xmax) * 0.5 < ref_pivot[0]) and (abs(((xmin + xmax) * 0.5) - abs(ref_pivot[0])) > 0.01):
                                        # print('Adding', sep_obj.name, 'right')
                                        right.append(sep_obj)
                                    else:
                                        # print('Adding', sep_obj.name, 'center_x')
                                        center_x.append(sep_obj)

                                elif new_obj.sx2.ymirror:
                                    if ((ymin + ymax) * 0.5 > ref_pivot[1]) and (abs(((ymin + ymax) * 0.5) - abs(ref_pivot[1])) > 0.01):
                                        # print('Adding', sep_obj.name, 'front')
                                        front.append(sep_obj)
                                    elif ((ymin + ymax) * 0.5 < ref_pivot[1]) and (abs(((ymin + ymax) * 0.5) - abs(ref_pivot[1])) > 0.01):
                                        # print('Adding', sep_obj.name, 'back')
                                        back.append(sep_obj)
                                    else:
                                        # print('Adding', sep_obj.name, 'center_y')
                                        center_y.append(sep_obj)

                                elif new_obj.sx2.zmirror:
                                    if ((zmin + zmax) * 0.5 > ref_pivot[2]) and (abs(((zmin + zmax) * 0.5) - abs(ref_pivot[2])) > 0.01):
                                        # print('Adding', sep_obj.name, 'top')
                                        top.append(sep_obj)
                                    elif ((zmin + zmax) * 0.5 < ref_pivot[2]) and (abs(((zmin + zmax) * 0.5) - abs(ref_pivot[2])) > 0.01):
                                        # print('Adding', sep_obj.name, 'bottom')
                                        bottom.append(sep_obj)
                                    else:
                                        # print('Adding', sep_obj.name, 'center_z')
                                        center_z.append(sep_obj)

                            for i, bucket in enumerate([left, right, center_x, top, bottom, center_y, front, back, center_z]):
                                if len(bucket) > 1:
                                    bpy.ops.object.select_all(action='DESELECT')
                                    view_layer.objects.active = bucket[0]
                                    cleanup = []
                                    for bucket_object in bucket:
                                        cleanup.append(bucket_object.data)
                                        bucket_object.select_set(True)

                                    for bucket_object in bucket[1:]:
                                        sep_objs.remove(bucket_object)

                                    # print('Bucket:', i, bucket)
                                    bpy.ops.object.join()

                                    for cleanup_mesh in cleanup[1:]:
                                        bpy.data.meshes.remove(cleanup_mesh)

                        if sep_objs:
                            sep_objs = [sep_obj for sep_obj in sep_objs if sep_obj.name in view_layer.objects]

                        if sep_objs:
                            separated_objs += sep_objs

                if sxglobals.benchmark_cvx:
                    now = time.perf_counter()
                    print(f'SX Tools: Convex hull generation: fragment merging done. Time: {now-then}')

                for new_obj in new_hull_objs:
                    bpy.ops.object.select_all(action='DESELECT')
                    sep_objs = self.smart_separate([new_obj, ], parent=False)

                    if sep_objs:
                        separated_objs += sep_objs
                    else:
                        new_objs.append(new_obj)

                if separated_objs:
                    new_objs += separated_objs
                    

                bpy.ops.object.select_all(action='DESELECT')
                for new_obj in new_objs:
                    shrink_colliders(new_obj)

                    if new_obj.name not in bpy.context.scene.collection.objects:
                        bpy.context.scene.collection.objects.link(new_obj)

                    if new_obj.name not in bpy.context.view_layer.objects:
                        bpy.context.view_layer.objects.link(new_obj)

                    if new_obj.name not in colliders.objects:
                        colliders.objects.link(new_obj)

                    # Clear sx2layers
                    new_obj.sx2layers.clear()
                    new_obj.select_set(False)

                for obj in org_objs:
                    obj.select_set(True)

                view_layer.objects.active = active_obj
                if not bpy.app.background:
                    setup.update_sx2material(bpy.context)


    def generate_mesh_colliders(self, objs):
        org_objs = objs[:]
        new_objs = []
        active_obj = bpy.context.view_layer.objects.active
        scene = bpy.context.scene.sx2
        name_list = [(obj.name[:], obj.data.name[:]) for obj in org_objs]

        export_objects = utils.create_collection('ExportObjects')
        colliders = utils.create_collection('SXColliders')
        collision_source_objs = utils.create_collection('CollisionSourceObjects')

        if colliders.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(colliders)

        for j, obj in enumerate(org_objs):
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()

            new_obj.data.name = name_list[j][1] + '_collision'
            new_obj.name = name_list[j][0] + '_collision'

            bpy.context.scene.collection.objects.link(new_obj)
            collision_source_objs.objects.link(new_obj)

            new_obj.parent = bpy.context.view_layer.objects[obj.parent.name]
            new_obj.sx2.subdivisionlevel = scene.sourcesubdivision
            new_obj.sx2.collideroffsetfactor = obj.sx2.collideroffsetfactor

            new_objs.append(new_obj)

        modifiers.apply_modifiers(new_objs)

        bpy.ops.object.select_all(action='DESELECT')
        for new_obj in new_objs:
            new_obj.select_set(True)
        bpy.context.view_layer.objects.active = new_objs[0]

        if obj.sx2.collideroffset:
            offset = -1.0 * obj.sx2.collideroffsetfactor * utils.find_safe_mesh_offset(obj)
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.transform.shrink_fatten(value=offset)
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        bpy.ops.object.vhacd('EXEC_DEFAULT', remove_doubles=scene.removedoubles, apply_transforms='NONE', resolution=scene.voxelresolution, concavity=scene.maxconcavity, planeDownsampling=scene.planedownsampling, convexhullDownsampling=scene.hulldownsampling, alpha=scene.collalpha, beta=scene.collbeta, maxHulls=scene.maxhulls, pca=False, projectHullVertices=True, mode='VOXEL', maxNumVerticesPerCH=scene.maxvertsperhull, minVolumePerCH=scene.minvolumeperhull)

        for src_obj in collision_source_objs.objects:
            bpy.data.objects.remove(src_obj, do_unlink=True)

        bpy.ops.object.select_all(action='DESELECT')

        # grab hulls generated by vhacd
        for obj in bpy.context.view_layer.objects:
            if ('_collision_hull' in obj.name) and (obj.name not in colliders.objects):
                colliders.objects.link(obj)
                obj.select_set(True)

        # set collider pivots
        export.set_pivots(colliders.objects, pivotmode=1)

        # optimize hulls
        # for obj in colliders.objects:
        #     obj.modifiers.new(type='DECIMATE', name='hullDecimate')
        #     obj.modifiers['hullDecimate'].show_viewport = True
        #     obj.modifiers['hullDecimate'].show_expanded = False
        #     obj.modifiers['hullDecimate'].decimate_type = 'DISSOLVE'
        #     obj.modifiers['hullDecimate'].angle_limit = math.radians(20.0)
        #     obj.modifiers['hullDecimate'].use_dissolve_boundaries = True

        #     obj.modifiers.new(type='WELD', name='hullWeld')
        #     obj.modifiers['hullWeld'].show_viewport = True
        #     obj.modifiers['hullWeld'].show_expanded = False
        #     obj.modifiers['hullWeld'].merge_threshold = 0.05

        #     modifiers.apply_modifiers([obj, ])

        #     bpy.context.view_layer.objects.active = obj
        #     bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        #     bpy.ops.mesh.select_all(action='SELECT')
        #     bpy.ops.mesh.convex_hull()
        #     bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        #     bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in org_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active_obj


    def generate_emission_meshes(self, objs):
        export_objects = utils.create_collection('ExportObjects')
        sxglobals.refresh_in_progress = True
        offset = 0.001
        active_obj = bpy.context.view_layer.objects.active
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)

        emission_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                # Check if emission layer is empty
                if not layers.get_layer_mask(obj, obj.sx2layers['Emission'])[1]:
                    bpy.context.view_layer.objects.active = obj
                    emission_obj = obj.copy()
                    emission_obj.data = obj.data.copy()
                    bpy.context.collection.objects.link(emission_obj)
                    export_objects.objects.link(emission_obj)
                    emission_obj.name = obj.name + '_emission'
                    emission_obj.data.name = emission_obj.name + '_mesh'
                    emission_obj.sx2.smartseparate = True
                    emission_obj.sx2.generatehulls = False
                    emission_obj.sx2.generatelodmeshes = False
                    emission_obj.sx2.generateemissionmeshes = False
                    emission_objs.append(emission_obj)

                    modifiers.apply_modifiers([emission_obj, ])
                    tools.select_mask([emission_obj, ], emission_obj.sx2layers['Emission'])
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

                    mesh = emission_obj.data
                    bm = bmesh.new()
                    bm.from_mesh(mesh)

                    to_delete = []
                    # Offset selected faces along their normals
                    for face in bm.faces:
                        if face.select:
                            normal_offset = face.normal * offset
                            for vert in face.verts:
                                vert.co += normal_offset
                        else:
                            to_delete.append(face)

                    # Delete unselected faces
                    bmesh.ops.delete(bm, geom=to_delete, context='FACES')

                    bm.to_mesh(mesh)
                    bm.free()

        bpy.context.view_layer.objects.active = active_obj
        sxglobals.refresh_in_progress = False

        return emission_objs


    def remove_exports(self):
        if 'ExportObjects' in bpy.data.collections:
            export_objects = bpy.data.collections['ExportObjects'].objects
            print('Export Objects:', export_objects.keys())
            for obj in export_objects:
                mesh = obj.data if obj.type == 'MESH' else None
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh:
                    bpy.data.meshes.remove(mesh)

        if 'SXColliders' in bpy.data.collections:
            collider_objects = bpy.data.collections['SXColliders'].objects
            print('Collider Objects:', collider_objects.keys())
            for obj in collider_objects:
                mesh = obj.data if obj.type == 'MESH' else None
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh:
                    bpy.data.meshes.remove(mesh)

        if 'SourceObjects' in bpy.data.collections:
            source_objects = bpy.data.collections['SourceObjects'].objects
            print('Source Objects:', source_objects.keys())
            self.set_pivots(source_objects)
            tags = sxglobals.keywords
            for obj in source_objects:
                if obj.type == 'MESH':
                    name = obj.name[:]
                    dataname = obj.data.name[:]
                    for tag in tags:
                        name = name.replace(tag, '')
                        dataname = dataname.replace(tag, '')
                    obj.name = name
                    obj.data.name = dataname
                    modifiers.remove_modifiers([obj, ])
                    modifiers.add_modifiers([obj, ])
                else:
                    name = obj.name[:]
                    for tag in tags:
                        name = name.replace(tag, '')
                    obj.name = name

                obj.hide_viewport = False
                source_objects.unlink(obj)


    # Remove and recreate all export uvlayers
    def reset_uv_layers(self, objs):
        for obj in objs:
            reserved = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]]['reserved']
            to_delete = []
            for uvlayer in obj.data.uv_layers:
                if uvlayer.name not in reserved:
                    to_delete.append(uvlayer.name[:])
            if to_delete:
                for uvlayer_name in to_delete:
                    obj.data.uv_layers.remove(obj.data.uv_layers[uvlayer_name])

            for i in range(8):
                uvmap = 'UVSet' + str(i)
                if uvmap not in obj.data.uv_layers.keys():
                    obj.data.uv_layers.new(name=uvmap)


    # Generate 1-bit layer masks for color layers
    # so the faces can be re-colored in a game engine.
    # Brush tools may leave low alpha values that break
    # palettemasks, alpha_tolerance can be used to fix this.
    # The default setting accepts only fully opaque values.
    def generate_palette_masks(self, objs, alpha_tolerance=1.0):
        channels = {'R': 0, 'G': 1, 'B': 2, 'A': 3, 'U': 0, 'V': 1}

        for obj in objs:
            layers = utils.find_color_layers(obj, staticvertexcolors=False)
            uvmap, target_channel = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]]['uv_palette_masks']
            if uvmap not in obj.data.uv_layers.keys():
                obj.data.uv_layers.new(name=uvmap)

            for i, layer in enumerate(layers):
                for poly in obj.data.polygons:
                    for loop_idx in poly.loop_indices:
                        vertex_alpha = 1.0 if i == 0 else obj.data.color_attributes[layer.color_attribute].data[loop_idx].color[3]
                        if vertex_alpha >= alpha_tolerance:
                            if layer.paletted:
                                obj.data.uv_layers[uvmap].data[loop_idx].uv[channels[target_channel]] = layer.palette_index + 1
                            elif 'PBR' in layer.name:
                                obj.data.uv_layers[uvmap].data[loop_idx].uv[channels[target_channel]] = 7
                            else:
                                obj.data.uv_layers[uvmap].data[loop_idx].uv[channels[target_channel]] = 6


    def generate_uv_channels(self, objs):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        for obj in objs:
            for layer in obj.sx2layers:
                if (layer.layer_type == 'CMP') or (layer.layer_type == 'CID'):
                    pass
                elif layer.layer_type != 'COLOR':
                    channel_name = 'uv_' + layer.name.lower()
                    uvmap, target_channel = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]][channel_name]

                    # Take layer opacity into account
                    values = layers.get_luminances(obj, layer)
                    if layer.name == 'Occlusion':
                        values = convert.invert_values(values)
                        values = [value * layer.opacity for value in values]
                        values = convert.invert_values(values)
                    else:
                        values = [value * layer.opacity for value in values]

                    if (prefs.exportroughness == 'SMOOTH') and (layer.layer_type == 'RGH'):
                        values = convert.invert_values(values)
                    layers.set_uvs(obj, uvmap, values, target_channel)

                elif 'Gradient' in layer.name:
                    channel_name = 'uv_' + layer.name.lower()
                    uvmap, target_channel = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]][channel_name]

                    values, empty = layers.get_layer_mask(obj, layer)
                    if not empty:
                        values = [value * layer.opacity for value in values]

                    layers.set_uvs(obj, uvmap, values, target_channel)

                elif layer.name == 'Overlay':
                    channel_name = 'uv_' + layer.name.lower()
                    target0, target1 = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]][channel_name]
                    colors = layers.get_layer(obj, obj.sx2layers['Overlay'])
                    gray = generate.color_list(obj, (0.5, 0.5, 0.5, 1.0))
                    colors = tools.blend_values(colors, gray, 'ALPHA', obj.sx2layers['Overlay'].opacity)

                    values0 = generate.empty_list(obj, 2)
                    values1 = generate.empty_list(obj, 2)
                    count = len(values0)//2
                    for i in range(count):
                        [values0[(0+i*2)], values0[(1+i*2)], values1[(0+i*2)], values1[(1+i*2)]] = colors[(0+i*4):(4+i*4)]

                    layers.set_uvs(obj, target0, values0, None)
                    layers.set_uvs(obj, target1, values1, None)


    def create_sxcollection(self):
        sx_objs = utils.create_collection('SXObjects')

        for obj in bpy.data.objects:
            if (len(obj.sx2layers) > 0) and (obj.name not in sx_objs.objects) and (obj.type == 'MESH'):
                sx_objs.objects.link(obj)

        if sx_objs.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(sx_objs)

        return [obj for obj in sx_objs.all_objects]


    def group_objects(self, objs, origin=False):
        utils.mode_manager(objs, set_mode=True, mode_id='group_objects')
        bpy.context.view_layer.objects.active = objs[0]

        if origin:
            pivot = Vector((0.0, 0.0, 0.0))
        else:
            pivot = utils.find_root_pivot(objs)

        group = bpy.data.objects.new('empty', None)
        bpy.context.scene.collection.objects.link(group)
        group.empty_display_size = 0.5
        group.empty_display_type = 'PLAIN_AXES'
        group.location = pivot
        group.name = objs[0].name.split('_', 1)[0] + '_root'
        group.show_name = True
        group.sx2.exportready = False

        for obj in objs:
            if obj.parent is None:
                obj.parent = group
                obj.location.x -= group.location.x
                obj.location.y -= group.location.y
                obj.location.z -= group.location.z

        utils.mode_manager(objs, set_mode=False, mode_id='group_objects')


    # pivotmodes: 0 == no change, 1 == center of mass, 2 == center of bbox,
    # 3 == base of bbox, 4 == world origin, 5 == pivot of parent, 6 == used with submesh convex hulls,
    # force == set mirror axis to mirrorobj
    def set_pivots(self, objs, pivotmode=None, force=False):
        viewlayer = bpy.context.view_layer
        active = viewlayer.objects.active
        selected = viewlayer.objects.selected[:]
        modedict = {'OFF': 0, 'MASS': 1, 'BBOX': 2, 'ROOT': 3, 'ORG': 4, 'PAR': 5, 'CID': 6}

        for sel in viewlayer.objects.selected:
            sel.select_set(False)

        for obj in objs:
            mode = modedict[obj.sx2.pivotmode] if pivotmode is None else pivotmode
            viewlayer.objects.active = obj
            obj.select_set(True)

            # print(f'SX Tools: Setting pivot for {obj.name} to mode {mode})

            if sxglobals.benchmark_magic:
                then = time.perf_counter()

            if mode == 0:
                pivot_loc = obj.matrix_world.to_translation()
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ])
                if obj.sx2.xmirror and not (xmin < pivot_loc[0] < xmax):
                    pivot_loc[0] *= -1.0
                if obj.sx2.ymirror and not (ymin < pivot_loc[1] < ymax):
                    pivot_loc[1] *= -1.0
                if obj.sx2.zmirror and not (zmin < pivot_loc[2] < zmax):
                    pivot_loc[2] *= -1.0
                bpy.context.scene.cursor.location = pivot_loc
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            elif mode == 1:
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
                if force:
                    pivot_loc = obj.matrix_world.to_translation()
                    if obj.sx2.mirrorobject is not None:
                        mirror_pivot_loc = obj.sx2.mirrorobject.matrix_world.to_translation()
                    else:
                        mirror_pivot_loc = (0.0, 0.0, 0.0)
                    if obj.sx2.xmirror:
                        pivot_loc[0] = mirror_pivot_loc[0]
                    if obj.sx2.ymirror:
                        pivot_loc[1] = mirror_pivot_loc[1]
                    if obj.sx2.zmirror:
                        pivot_loc[2] = mirror_pivot_loc[2]
                    bpy.context.scene.cursor.location = pivot_loc
                    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            elif mode == 2:
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                if force:
                    pivot_loc = obj.matrix_world.to_translation()
                    if obj.sx2.mirrorobject is not None:
                        mirror_pivot_loc = obj.sx2.mirrorobject.matrix_world.to_translation()
                    else:
                        mirror_pivot_loc = (0.0, 0.0, 0.0)
                    if obj.sx2.xmirror:
                        pivot_loc[0] = mirror_pivot_loc[0]
                    if obj.sx2.ymirror:
                        pivot_loc[1] = mirror_pivot_loc[1]
                    if obj.sx2.zmirror:
                        pivot_loc[2] = mirror_pivot_loc[2]
                    bpy.context.scene.cursor.location = pivot_loc
                    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            elif mode == 3:
                bpy.context.scene.cursor.location = utils.find_root_pivot([obj, ])
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            elif mode == 4:
                bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            elif mode == 5:
                pivot_loc = obj.parent.matrix_world.to_translation()
                bpy.context.scene.cursor.location = pivot_loc
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            elif mode == 6:
                pivot_world = obj.matrix_world.to_translation()
                # pivot_world = obj.sx2.mirrorobject.matrix_world.to_translation() if obj.sx2.mirrorobject else  obj.matrix_world.to_translation()
                pivot_loc = obj.matrix_local.to_translation()
                pivot_ref = obj.sx2.mirrorobject.matrix_local.to_translation() if obj.sx2.mirrorobject else (0.0, 0.0, 0.0)
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ], mode='parent')
                if obj.sx2.xmirror and (((xmin + xmax) * 0.5) > pivot_ref[0]) and (pivot_loc[0] < pivot_ref[0]):
                    pivot_world[0] += -2 * (pivot_loc[0] - pivot_ref[0])
                    # pivot_loc[0] *= -1.0
                elif obj.sx2.xmirror and (((xmin + xmax) * 0.5) < pivot_ref[0]) and (pivot_loc[0] > pivot_ref[0]):
                    pivot_world[0] += -2 * (pivot_loc[0] - pivot_ref[0])
                    # pivot_loc[0] *= -1.0
                if obj.sx2.ymirror and (((ymin + ymax) * 0.5) > pivot_ref[1]) and (pivot_loc[1] < pivot_ref[1]):
                    pivot_world[1] += -2 * (pivot_loc[1] - pivot_ref[1])
                    # pivot_loc[1] *= -1.0
                elif obj.sx2.ymirror and (((ymin + ymax) * 0.5) < pivot_ref[1]) and (pivot_loc[1] > pivot_ref[1]):
                    pivot_world[1] += -2 * (pivot_loc[1] - pivot_ref[1])
                    # pivot_loc[1] *= -1.0
                if obj.sx2.zmirror and (((zmin + zmax) * 0.5) > pivot_ref[2]) and (pivot_loc[2] < pivot_ref[2]):
                    pivot_world[2] += -2 * (pivot_loc[2] - pivot_ref[2])
                    # pivot_loc[2] *= -1.0
                elif obj.sx2.zmirror and (((zmin + zmax) * 0.5) < pivot_ref[2]) and (pivot_loc[2] > pivot_ref[2]):
                    pivot_world[2] += -2 * (pivot_loc[2] - pivot_ref[2])
                    # pivot_loc[2] *= -1.0
                bpy.context.scene.cursor.location = pivot_world
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            else:
                pass

            if sxglobals.benchmark_magic:
                now = time.perf_counter()
                print(f'SX Tools: {obj.name} pivot set: {round(now-then, 4)} seconds')

            obj.select_set(False)

        for sel in selected:
            sel.select_set(True)
        viewlayer.objects.active = active


    def create_reserved_uvsets(self, objs):
        active = bpy.context.view_layer.objects.active
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            setCount = len(obj.data.uv_layers)
            if setCount == 0:
                reserved = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]]['reserved']
                for uvset in reserved:
                    obj.data.uv_layers.new(name=uvset)
            elif setCount > 6:
                base = obj.data.uv_layers[0]
                if base.name != 'UVSet0':
                    print(f'SX Tools Error: {obj.name} does not have enough free UV Set slots for the operation (2 free slots needed)')
            else:
                base = obj.data.uv_layers[0]
                if 'UVSet' not in base.name:
                    base.name = 'UVSet0'
                    if (setCount > 1) and (obj.data.uv_layers[1].name != 'UVSet1'):
                        obj.data.uv_layers[1].name = 'UVSet1'
                elif base.name == 'UVSet1':
                    obj.data.uv_layers.new(name='UVSet0')
                    for i in range(setCount):
                        uvName = obj.data.uv_layers[0].name
                        obj.data.uv_layers.active_index = 0
                        bpy.ops.mesh.uv_texture_add()
                        obj.data.uv_layers.active_index = 0
                        bpy.ops.mesh.uv_texture_remove()
                        obj.data.uv_layers[setCount].name = uvName
                else:
                    print(f'SX Tools Error: {obj.name} has an unknown UV Set configuration')

        bpy.context.view_layer.objects.active = active


    def bytes_to_floats(self, objs):
        active = bpy.context.view_layer.objects.active
        for obj in objs:
            if (obj.sx2layers) and (obj.type == 'MESH'):
                bpy.context.view_layer.objects.active = obj
                byte_colors = []
                for color_attribute in obj.data.color_attributes:
                    if color_attribute.data_type == 'BYTE_COLOR':
                        byte_colors.append(color_attribute.name)

                if byte_colors:
                    for byte_color in byte_colors:
                        # VertexColor0 was only used by SX Tools 1 for compositing, and is redundant
                        if byte_color == 'VertexColor0':
                            obj.data.color_attributes.remove(obj.data.color_attributes[byte_color])
                        else:
                            obj.data.color_attributes.active_color = obj.data.color_attributes[byte_color]
                            bpy.ops.geometry.color_attribute_convert(domain='CORNER', data_type='FLOAT_COLOR')

        bpy.context.view_layer.objects.active = active


    def revert_objects(self, objs):
        sxglobals.magic_in_progress = True
        modifiers.remove_modifiers(objs)
        prefs = bpy.context.preferences.addons['sxtools2'].preferences

        for obj in objs:
            if obj.sx2.category != 'DEFAULT':
                if obj.sx2.transmissionoverride:
                    clr_layers = ['Overlay', 'Occlusion', 'Metallic', 'Roughness']
                else:
                    clr_layers = ['Overlay', 'Occlusion', 'Metallic', 'Roughness', 'Transmission', 'Subsurface']
                    
                for layer_name in clr_layers:
                    if layer_name in obj.sx2layers.keys():
                        layers.clear_layers([obj, ], obj.sx2layers[layer_name])

        active = bpy.context.view_layer.objects.active
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
            bpy.ops.object.shade_flat()

        bpy.context.view_layer.objects.active = active
        sxglobals.magic_in_progress = False


    def zero_verts(self, objs):
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)

        for obj in objs:
            xmirror = obj.sx2.xmirror
            ymirror = obj.sx2.ymirror
            zmirror = obj.sx2.zmirror
            bm = bmesh.from_edit_mesh(obj.data)

            selectedVerts = [vert for vert in bm.verts if vert.select]
            for vert in selectedVerts:
                if xmirror:
                    vert.co.x = 0.0 - obj.location.x
                if ymirror:
                    vert.co.y = 0.0 - obj.location.y
                if zmirror:
                    vert.co.z = 0.0 - obj.location.z

            bmesh.update_edit_mesh(obj.data)
            bm.free()


    def validate_objects(self, objs):
        utils.mode_manager(objs, set_mode=True, mode_id='validate_objects')

        self.validate_sxtoolmaterial()

        ok0 = self.validate_categories(objs)
        ok1 = self.validate_palette_layers(objs)
        ok2 = self.validate_names(objs)
        ok3 = True  # self.validate_loose(objs)
        ok4 = True  # self.validate_uv_sets(objs)

        utils.mode_manager(objs, set_mode=False, mode_id='validate_objects')

        if ok0 and ok1 and ok2 and ok3 and ok4:
            print('SX Tools: Selected objects passed validation tests')
            return True
        else:
            print('SX Tools Error: Selected objects failed validation tests')
            print('Category validation: ', ok0)
            print('Palette layers validation: ', ok1)
            print('Names validation: ', ok2)
            print('Loose geometry validation: ', ok3)
            return False


    def validate_sxtoolmaterial(self):
        try:
            sxtoolmaterial = bpy.data.materials['SXToolMaterial'].node_tree
        except:
            print('SX Tools Error: Missing SXToolMaterial.\nPerforming SXToolMaterial reset.')
            setup.create_sxtoolmaterial()


    # Check for improper UV sets
    def validate_uv_sets(self, objs):
        for obj in objs:
            for i, uv_layer in enumerate(obj.data.uv_layers):
                if uv_layer.name != 'UVSet'+str(i):
                    message_box('Objects contain incorrectly named UV Sets!')
                    print(f'SX Tools Error: {obj.name} has incorrect UV Set naming')
                    return False

        return True


    # Check for loose geometry
    def validate_loose(self, objs):
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.select_loose()

        loose_objs = []
        for obj in objs:
            obj.update_from_editmode()
            mesh = obj.data
            selection = [None] * len(mesh.vertices)
            mesh.vertices.foreach_get('select', selection)

            if True in selection:
                loose_objs.append(obj.name)

        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        if loose_objs:
            new_line = '\n'
            message_box(f'Objects contain loose geometry:{new_line}{new_line.join(map(str, loose_objs))}')
            print(f'SX Tools Error: Loose geometry in {loose_objs}')
            return False

        return True


    # Check that objects are grouped
    def validate_parents(self, objs):
        for obj in objs:
            if obj.parent is None:
                message_box('Object is not in a group: ' + obj.name)
                print(f'SX Tools Error: {obj.name} is not in a group')
                return False

        return True


    # if paletted, fail if layer colorcount > 1
    def validate_palette_layers(self, objs):
        for obj in objs:
            if obj.sx2.category != 'DEFAULT':
                mesh = obj.data
                layers = utils.find_color_layers(obj, staticvertexcolors=int(obj.sx2.staticvertexcolors))
                for layer in layers:
                    if layer.paletted:
                        colors = []
                        vertex_colors = mesh.attributes[layer.color_attribute].data

                        for poly in mesh.polygons:
                            for loop_idx in poly.loop_indices:
                                colors.append(vertex_colors[loop_idx].color[:])

                        # if object is transparent, ignore alpha
                        if layers[0].opacity < 1.0:
                            quantized_colors = [(utils.round_stepped(color[0]), utils.round_stepped(color[1]), utils.round_stepped(color[2])) for color in colors]
                        else:
                            quantized_colors = [(utils.round_stepped(color[0]), utils.round_stepped(color[1]), utils.round_stepped(color[2]), utils.round_stepped(color[3])) for color in colors]
                        color_set = set(quantized_colors)

                        if layer == layers[0]:
                            if len(color_set) > 1:
                                print(f'SX Tools Error: Multiple colors in {obj.name} {layer.name}\nValues: {color_set}')
                                message_box('Multiple colors in ' + obj.name + ' ' + layer.name)
                                return False
                        else:
                            if len(color_set) > 2:
                                print(f'SX Tools Error: Multiple colors in {obj.name} {layer.name}\nValues: {color_set}')
                                message_box('Multiple colors in ' + obj.name + ' ' + layer.name)
                                return False
        return True


    # if paletted, check that emissive faces have color in static channel
    def validate_emissives(self, objs):
        pass


    def validate_names(self, objs):
        for obj in objs:
            if ('sxMirror' in obj.modifiers.keys()) and (obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror):
                for keyword in sxglobals.keywords:
                    if keyword in obj.name:
                        message_box(obj.name + '\ncontains the substring ' + keyword + '\nreserved for Smart Separate\nfunction of Mirror Modifier')
                        return False
        return True


    def validate_categories(self, objs):
        cmp_objs = []
        for obj in objs:
            if ('Composite' in obj.sx2layers.keys()) and (obj.sx2layers['Composite'].layer_type == 'CMP'):
                cmp_objs.append(obj)
        if cmp_objs:
            layers.del_layer(cmp_objs, 'Composite')

        for obj in objs:
            if obj.sx2.category != 'DEFAULT':
                category_data = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]]
                cat_layers = category_data['layers']
                obj_layers = [layer for layer in obj.sx2layers if layer.name != 'Collider IDs']
                if len(cat_layers) == len(obj_layers):
                    for cat_layer, layer in zip(cat_layers, obj_layers):
                        if (cat_layer[0] != layer.name) or (cat_layer[1] != layer.layer_type):
                            print(f'SX Tools Error: {obj.name} {layer.name} does not match the selected category')
                            message_box(f'{obj.name}: {layer.name} and {cat_layer[0]} do not match the selected category')
                            return False
                else:
                    print(f'SX Tools Error: {obj.name} layers do not match the selected category')
                    message_box(obj.name + ' layers do not match the selected category')
                    return False
        return True


    def __del__(self):
        print('SX Tools: Exiting export')


# ------------------------------------------------------------------------
#    Baking and Processing Functions
# ------------------------------------------------------------------------
class SXTOOLS2_magic(object):
    def __init__(self):
        return None


    # This is a project-specific batch operation.
    # These should be adapted to the needs of the game,
    # baking category-specific values to achieve
    # consistent project-wide looks.
    def process_objects(self, objs):
        if not sxglobals.refresh_in_progress:
            sxglobals.refresh_in_progress = True

        then = time.perf_counter()
        then2 = 0.0

        scene = bpy.context.scene.sx2
        viewlayer = bpy.context.view_layer
        org_obj_names = {}

        org_toolmode = scene.toolmode
        org_toolopacity = scene.toolopacity
        org_toolblend = scene.toolblend
        org_rampmode = scene.rampmode
        org_ramplist = scene.ramplist
        org_ramp = utils.get_ramp()
        org_dirangle = scene.dirAngle
        org_dirinclination = scene.dirInclination
        org_dircone = scene.dirCone

        utils.mode_manager(objs, set_mode=True, mode_id='process_objects')
        scene.toolopacity = 1.0
        scene.toolblend = 'ALPHA'

        for obj in objs:
            for layer in obj.sx2layers:
                layer.locked = False

        # Make sure export and source Collections exist
        export_objects = utils.create_collection('ExportObjects')
        source_objects = utils.create_collection('SourceObjects')

        # Make sure objects are in groups
        for obj in objs:
            if obj.parent is None:
                obj.hide_viewport = False
                # viewlayer.objects.active = obj
                export.group_objects([obj, ])

            if '_mesh' not in obj.data.name:
                obj.data.name = obj.name + '_mesh'

        # Remove empties from selected objects
        for sel in viewlayer.objects.selected:
            if sel.type != 'MESH':
                sel.select_set(False)

        # Prepare modifiers
        modifiers.remove_modifiers(objs)

        if sxglobals.benchmark_magic:
            now = time.perf_counter()
            print(f'SX Tools: Modifier stack removed: {round(now-then, 4)} seconds')
            then2 = now

        for obj in objs:
            obj.sx2.modifiervisibility = True
        modifiers.add_modifiers(objs)

        if sxglobals.benchmark_magic:
            now = time.perf_counter()
            print(f'SX Tools: Modifier stack added: {round(now-then2, 4)} seconds')
            then2 = now

        # Create high-poly bake meshes
        if scene.exportquality == 'HI':
            new_objs = []
            lod_objs = []
            sep_objs = []
            part_objs = []

            for obj in objs:
                if obj.sx2.smartseparate:
                    if obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror:
                        sep_objs.append(obj)

            if sep_objs:
                part_objs = export.smart_separate(sep_objs)
                for obj in part_objs:
                    if obj not in objs:
                        objs.append(obj)

            groups = utils.find_groups(objs)
            for group in groups:
                org_group = bpy.data.objects.new('empty', None)
                bpy.context.scene.collection.objects.link(org_group)
                org_group.empty_display_size = 2
                org_group.empty_display_type = 'PLAIN_AXES'
                org_group.location = group.location
                org_group.name = group.name + '_org'
                org_group.hide_viewport = True
                source_objects.objects.link(org_group)
                export_objects.objects.link(group)

            viewlayer.update()

            for obj in objs:
                if obj.name not in source_objects.objects:
                    source_objects.objects.link(obj)
                org_obj_names[obj] = [obj.name, obj.data.name][:]
                obj.data.name = obj.data.name + '_org'
                obj.name = obj.name + '_org'

                new_obj = obj.copy()
                new_obj.data = obj.data.copy()
                new_obj.name = org_obj_names[obj][0]
                new_obj.data.name = org_obj_names[obj][1]
                bpy.context.scene.collection.objects.link(new_obj)
                export_objects.objects.link(new_obj)

                if obj.sx2.generatelodmeshes:
                    lod_objs.append(new_obj)
                else:
                    new_objs.append(new_obj)

                obj.parent = viewlayer.objects[obj.parent.name + '_org']

            if lod_objs:
                new_obj_list = export.generate_lods(lod_objs)
                for new_obj in new_obj_list:
                    new_objs.append(new_obj)

            objs = new_objs

        # Set subdivision to 1 for the duration of the processing
        subdivision_list = [obj.sx2.subdivisionlevel for obj in objs]
        for obj in objs:
            if ('sxSubdivision' in obj.modifiers) and (obj.sx2.subdivisionlevel > 1):
                obj.modifiers['sxSubdivision'].levels = 1

        if sxglobals.benchmark_magic:
            now = time.perf_counter()
            print(f'SX Tools: Setup 1: {round(now-then2, 4)} seconds')
            then2 = now

        # Place pivots
        export.set_pivots(objs, force=True)

        if sxglobals.benchmark_magic:
            now = time.perf_counter()
            print(f'SX Tools: Setup 2: {round(now-then2, 4)} seconds')
            then2 = now

        # Fix transforms
        utils.clear_parent_inverse_matrix(objs)

        if sxglobals.benchmark_magic:
            now = time.perf_counter()
            print(f'SX Tools: Setup 3: {round(now-then2, 4)} seconds')
            then2 = now

        for obj in objs:
            obj.select_set(False)

        viewlayer.objects.active = objs[0]

        # Begin category-specific compositing operations
        # Hide modifiers for performance
        for obj in objs:
            if obj.sx2.category == '':
                obj.sx2.category == 'DEFAULT'

        for obj in viewlayer.objects:
            if obj.type == 'MESH' and obj.hide_viewport is False:
                obj.hide_viewport = True

        # Mandatory to update visibility?
        # viewlayer.update()

        if sxglobals.benchmark_magic:
            now = time.perf_counter()
            print(f'SX Tools: Mesh setup and viewlayer update: {round(now-then2, 4)} seconds')
            then2 = now

        category_list = list(sxglobals.category_dict.keys())
        categories = [category.replace(" ", "_").upper() for category in category_list]
        for category in categories:
            category_objs = [obj for obj in objs if obj.sx2.category == category]

            if category_objs:
                group_list = utils.find_groups(category_objs)
                for group in group_list:
                    createLODs = False
                    group_objs = utils.find_children(group, category_objs, recursive=True, child_type='MESH')
                    viewlayer.objects.active = group_objs[0]
                    for obj in group_objs:
                        if obj.sx2.generatelodmeshes:
                            createLODs = True
                        obj.hide_viewport = False
                        obj.select_set(True)

                    if category == 'DEFAULT':
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_default(group_objs)
                    if category == 'STATIC':
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_static(group_objs)
                    elif category == 'PALETTED':
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_paletted(group_objs)
                    elif (category == 'VEHICLES') or (category == 'NPCVEHICLES'):
                        for obj in group_objs:
                            if ('wheel' in obj.name) or ('tire' in obj.name):
                                scene.occlusionblend = 0.0
                            else:
                                scene.occlusionblend = 0.5
                            if obj.name.endswith('_roof') or obj.name.endswith('_frame') or obj.name.endswith('_dash') or obj.name.endswith('_hood') or ('bumper' in obj.name):
                                obj.modifiers['sxDecimate2'].use_symmetry = True
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        if (scene.exportquality == 'HI') and (createLODs is True):
                            for i in range(3):
                                lod_objs = []
                                for obj in group_objs:
                                    if '_LOD'+str(i) in obj.name:
                                        lod_objs.append(obj)
                                        # obj.select_set(True)
                                        obj.hide_viewport = False
                                    else:
                                        # obj.select_set(False)
                                        obj.hide_viewport = True
                                viewlayer.objects.active = lod_objs[0]
                                self.process_vehicles(lod_objs)
                        else:
                            self.process_vehicles(group_objs)
                    elif category == 'BUILDINGS':
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_buildings(group_objs)
                    elif category == 'TREES':
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_trees(group_objs)
                    elif category == 'CHARACTERS':
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_characters(group_objs)
                    elif category == 'ROADTILES':
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_roadtiles(group_objs)
                    else:
                        if scene.exportquality == 'HI':
                            modifiers.apply_modifiers(group_objs)
                        self.process_default(group_objs)

                    for obj in group_objs:
                        obj.select_set(False)
                        obj.hide_viewport = True

                if sxglobals.benchmark_magic:
                    now = time.perf_counter()
                    print(f'SX Tools: {category} / {len(group_list)} groups duration: {round(now-then2, 4)} seconds')
                    then2 = now

        for obj in viewlayer.objects:
            if (scene.exportquality == 'HI') and ('_org' in obj.name):
                obj.hide_viewport = True
            elif obj.type == 'MESH':
                obj.hide_viewport = False

        # Restore subdivision level
        for i, obj in enumerate(objs):
            if 'sxSubdivision' in obj.modifiers:
                obj.modifiers['sxSubdivision'].levels = subdivision_list[i]
        # viewlayer.update()

        # LOD mesh generation for low-detail
        if scene.exportquality == 'LO':
            non_lod_objs = []
            for obj in objs:
                if obj.sx2.generatelodmeshes:
                    if '_LOD' not in obj.name:
                        non_lod_objs.append(obj)
                else:
                    obj.select_set(True)

            if non_lod_objs:
                lod_objs = export.generate_lods(non_lod_objs)
                bpy.ops.object.select_all(action='DESELECT')
                for obj in non_lod_objs:
                    obj.select_set(True)
                for obj in lod_objs:
                    obj.select_set(True)

        if scene.exportquality == 'HI':
            for obj in objs:
                obj.select_set(True)
                # obj.show_viewport = True
                obj.modifiers.new(type='WEIGHTED_NORMAL', name='sxWeightedNormal')
                obj.modifiers['sxWeightedNormal'].mode = 'FACE_AREA_WITH_ANGLE'
                obj.modifiers['sxWeightedNormal'].weight = 50
                obj.modifiers['sxWeightedNormal'].keep_sharp = True

            # self.apply_modifiers(objs)

        # Restore tool settings
        if not bpy.app.background:
            scene.toolmode = org_toolmode
            scene.toolopacity = org_toolopacity
            scene.toolblend = org_toolblend
            scene.rampmode = org_rampmode
            scene.ramplist = org_ramplist
            load_ramp(self, bpy.context, org_ramp)
            scene.dirAngle = org_dirangle
            scene.dirInclination = org_dirinclination
            scene.dirCone = org_dircone

        now = time.perf_counter()
        print(f'SX Tools: Magic process duration: {round(now-then, 4)} seconds')

        utils.mode_manager(objs, set_mode=False, mode_id='process_objects')
        sxglobals.refresh_in_progress = False


    def apply_palette_overrides(self, objs, clear=False):
        for obj in objs:
            if obj.sx2.metallicoverride or obj.sx2.roughnessoverride:
                color_layers = utils.find_color_layers(obj, staticvertexcolors=int(obj.sx2.staticvertexcolors))
                paletted_layers = [layer for layer in color_layers if layer.paletted]

            if obj.sx2.roughnessoverride:
                if clear:
                    layers.clear_layers([obj, ], obj.sx2layers['Roughness'])

                if 'Roughness' not in obj.sx2layers.keys():
                    layers.add_layer([obj, ], name='Roughness', layer_type='RGH')

                for layer in paletted_layers:
                    value = getattr(obj.sx2, 'roughness'+str(layer.palette_index))
                    value *= layer.opacity
                    color = (value, value, value, 1.0)
                    values = generate.color_list(obj, color, layer)
                    base = layers.get_layer(obj, obj.sx2layers['Roughness'])
                    colors = tools.blend_values(values, base, 'ALPHA', 1.0)
                    layers.set_layer(obj, colors, obj.sx2layers['Roughness'])

            if obj.sx2.metallicoverride:
                if clear:
                    layers.clear_layers([obj, ], obj.sx2layers['Metallic'])

                if 'Metallic' not in obj.sx2layers.keys():
                    layers.add_layer([obj, ], name='Metallic', layer_type='MET')

                for layer in paletted_layers:
                    value = getattr(obj.sx2, 'metallic'+str(layer.palette_index))
                    value *= layer.opacity
                    color = (value, value, value, 1.0)
                    values = generate.color_list(obj, color, layer)
                    base = layers.get_layer(obj, obj.sx2layers['Metallic'])
                    colors = tools.blend_values(values, base, 'ALPHA', 1.0)
                    layers.set_layer(obj, colors, obj.sx2layers['Metallic'])


    def apply_occlusion(self, objs, masklayername=None, blend=0.5, rays=50, groundplane=True, distance=10.0):
        for obj in objs:
            layer = obj.sx2layers['Occlusion']
            masklayer = obj.sx2layers[masklayername] if masklayername else None
            colors = generate.occlusion_list(obj, rays, blend, distance, groundplane)
            if masklayer:
                mask = generate.color_list(obj, (1.0, 1.0, 1.0, 1.0), masklayer)
                colors = tools.blend_values(mask, colors, 'ALPHA', 1.0)
            layers.set_layer(obj, colors, layer)

            blur_colors = generate.blur_list(obj, layer, masklayer)

            colors = tools.blend_values(blur_colors, colors, 'ALPHA', 0.3)
            if masklayer:
                mask = generate.color_list(obj, (1.0, 1.0, 1.0, 1.0), masklayer)
                colors = tools.blend_values(mask, colors, 'ALPHA', 1.0)
            layers.set_layer(obj, colors, layer)

            obj.sx2layers['Occlusion'].opacity = obj.sx2.mat_occlusion


    def apply_curvature_overlay(self, objs, convex=True, concave=True, noise=0.0):
        scene = bpy.context.scene.sx2
        layer = objs[0].sx2layers['Overlay']
        scene.toolmode = 'CRV'
        scene.toolopacity = 1.0
        scene.normalizeconvex = convex
        scene.normalizeconcave = concave

        tools.apply_tool(objs, layer)

        if noise > 0.0:
            scene.noisemono = False
            scene.toolmode = 'NSE'
            scene.toolopacity = noise
            scene.toolblend = 'MUL'

            tools.apply_tool(objs, layer)

        for obj in objs:
            obj.sx2layers['Overlay'].blend_mode = 'OVR'
            obj.sx2layers['Overlay'].opacity = obj.sx2.mat_overlay

        scene.toolopacity = 1.0


    def process_default(self, objs):
        self.apply_palette_overrides(objs)


    def process_static(self, objs):
        obj = objs[0]

        self.apply_palette_overrides(objs, clear=True)
        self.apply_occlusion(objs)
        self.apply_curvature_overlay(objs, convex=True, concave=True, noise=0.01)
        # Emissives are smooth
        tools.apply_tool(objs, obj.sx2layers['Roughness'], masklayer=obj.sx2layers['Emission'], color=(0.0, 0.0, 0.0, 1.0))


    def process_roadtiles(self, objs):
        scene = bpy.context.scene.sx2
        self.apply_palette_overrides(objs, clear=False)
        self.apply_occlusion(objs, blend=0.0, groundplane=False, distance=50.0)
        self.apply_curvature_overlay(objs, convex=False, concave=False)

        # Apply Gradient1 to roughness and metallic
        for obj in objs:
            obj.sx2layers['Gradient1'].opacity = 1.0
            colors = layers.get_layer(obj, obj.sx2layers['Gradient1'])
            if colors is not None:
                scene.toolmode = 'COL'
                scene.toolopacity = 1.0
                scene.toolblend = 'ALPHA'
                tools.apply_tool(objs, obj.sx2layers['Metallic'], masklayer=obj.sx2layers['Gradient1'], color=(1.0, 1.0, 1.0, 1.0))
                tools.apply_tool(objs, obj.sx2layers['Roughness'], masklayer=obj.sx2layers['Gradient1'], color=(0.0, 0.0, 0.0, 1.0))

        # Emissive road paint
        for obj in objs:
            colors = layers.get_layer(obj, obj.sx2layers['Layer 2 - Paint'])
            if colors is not None:
                layers.set_layer(obj, colors, obj.sx2layers['Emission'])

        # Emissives are smooth
        # tools.apply_tool(objs, obj.sx2layers['Roughness'], masklayer=obj.sx2layers['Emission'], color=(0.0, 0.0, 0.0, 1.0))

        # Tone down Gradient1 contribution
        for obj in objs:
            obj.sx2layers['Gradient1'].opacity = 0.15


    def process_characters(self, objs):
        obj = objs[0]

        self.apply_palette_overrides(objs)
        self.apply_occlusion(objs, blend=0.2, distance=1.0)
        self.apply_curvature_overlay(objs)

        # Emissives are smooth
        tools.apply_tool(objs, obj.sx2layers['Roughness'], masklayer=obj.sx2layers['Emission'], color=(0.0, 0.0, 0.0, 1.0))


    def process_paletted(self, objs):
        scene = bpy.context.scene.sx2
        obj = objs[0]

        self.apply_palette_overrides(objs, clear=True)
        self.apply_occlusion(objs, masklayername='Emission')
        self.apply_curvature_overlay(objs, convex=True, concave=True, noise=0.01)

        material = 'Iron'
        palette = [
            bpy.context.scene.sx2materials[material].color0,
            bpy.context.scene.sx2materials[material].color1,
            bpy.context.scene.sx2materials[material].color2]

        for obj in objs:
            colors = layers.get_layer(obj, obj.sx2layers['Roughness'])
            # Static colors layer is rough
            rgh_value = (obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness)
            colors1 = generate.color_list(obj, rgh_value, utils.find_color_layers(obj, 5))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Combine with roughness from PBR material
            colors1 = generate.color_list(obj, palette[2], utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Noise for variance
            colors1 = generate.noise_list(obj, 0.01, True)
            colors = tools.blend_values(colors1, colors, 'OVR', 1.0)
            # Combine roughness base mask with custom curvature gradient
            scene.normalizeconvex = True
            scene.normalizeconcave = True
            scene.ramplist = 'CURVATUREROUGHNESS'
            colors1 = generate.curvature_list(obj, objs)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ADD', 0.1)
            # Combine previous mix with directional dust
            scene.ramplist = 'DIRECTIONALDUST'
            scene.dirAngle = 0.0
            scene.dirInclination = 90.0
            scene.dirCone = 30
            colors1 = generate.direction_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ADD', 0.2)
            # Emissives are smooth
            color = (0.0, 0.0, 0.0, 1.0)
            colors1 = generate.color_list(obj, color, obj.sx2layers['Emission'])
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Write roughness
            layer = obj.sx2layers['Roughness']
            layers.set_layer(obj, colors, layer)

            # Mix metallic with occlusion (dirt in crevices)
            colors = generate.color_list(obj, color=palette[1], masklayer=utils.find_color_layers(obj, 6))
            colors1 = layers.get_layer(obj, obj.sx2layers['Occlusion'], single_as_alpha=True)
            if colors is not None:
                colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
                layers.set_layer(obj, colors, obj.sx2layers['Metallic'])


    def process_vehicles(self, objs):
        scene = bpy.context.scene.sx2

        self.apply_palette_overrides(objs, clear=True)
        self.apply_occlusion(objs, masklayername='Emission')
        self.apply_curvature_overlay(objs, convex=True, concave=False, noise=0.01)

        material = 'Iron'
        palette = [
            bpy.context.scene.sx2materials[material].color0,
            bpy.context.scene.sx2materials[material].color1,
            bpy.context.scene.sx2materials[material].color2]

        for obj in objs:
            colors = layers.get_layer(obj, obj.sx2layers['Roughness'])
            # Static colors layer is rough
            rgh_value = (obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness)
            colors1 = generate.color_list(obj, rgh_value, utils.find_color_layers(obj, 5))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Combine with roughness from PBR material
            colors1 = generate.color_list(obj, palette[2], utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Noise for variance
            colors1 = generate.noise_list(obj, 0.01, True)
            colors = tools.blend_values(colors1, colors, 'OVR', 1.0)
            # Combine roughness base mask with custom curvature gradient
            scene.normalizeconvex = True
            scene.normalizeconcave = False
            scene.ramplist = 'CURVATUREROUGHNESS'
            colors1 = generate.curvature_list(obj, objs)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ADD', 0.1)
            # Combine previous mix with directional dust
            scene.ramplist = 'DIRECTIONALDUST'
            scene.dirAngle = 0.0
            scene.dirInclination = 40.0
            scene.dirCone = 30
            colors1 = generate.direction_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ADD', 0.2)
            # Emissives are smooth
            colors1 = generate.color_list(obj, (0.0, 0.0, 0.0, 1.0), obj.sx2layers['Emission'])
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Write roughness
            layers.set_layer(obj, colors, obj.sx2layers['Roughness'])

            # Apply PBR metal based on layer7
            # noise = 0.01
            # mono = True
            scene.toolmode = 'COL'
            scene.toolopacity = 1.0
            scene.toolblend = 'ALPHA'

            # Mix metallic with occlusion (dirt in crevices)
            colors = generate.color_list(obj, color=palette[1], masklayer=utils.find_color_layers(obj, 6))
            colors1 = layers.get_layer(obj, obj.sx2layers['Occlusion'])  # , single_as_alpha=True)
            if colors is not None:
                colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
                layers.set_layer(obj, colors, obj.sx2layers['Metallic'])


    def process_buildings(self, objs):
        scene = bpy.context.scene.sx2
        obj = objs[0]

        objs_windows = []
        objs_tiles = []
        for obj in objs:
            if 'window' in obj.name:
                objs_windows.append(obj)
                obj.hide_viewport = True
            else:
                objs_tiles.append(obj)

        objs = objs_tiles[:]

        self.apply_palette_overrides(objs, clear=True)
        self.apply_occlusion(objs, masklayername=utils.find_color_layers(obj, 6).name, blend=0.0, groundplane=False, distance=0.5)

        # Apply curvature with luma remapping to overlay, clear windows
        scene.normalizeconvex = False
        scene.normalizeconcave = False
        scene.ramplist = 'WEARANDTEAR'
        for obj in objs:
            colors = generate.curvature_list(obj, objs)
            values = layers.get_luminances(obj, colors=colors)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            color = (0.5, 0.5, 0.5, 1.0)
            colors1 = generate.color_list(obj, color=color, masklayer=utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            layers.set_layer(obj, colors, obj.sx2layers['Overlay'])
            obj.sx2layers['Overlay'].blend_mode = 'OVR'
            obj.sx2layers['Overlay'].opacity = 0.6

        material = 'Silver'
        palette = [
            bpy.context.scene.sx2materials[material].color0,
            bpy.context.scene.sx2materials[material].color1,
            bpy.context.scene.sx2materials[material].color2]

        for obj in objs:
            colors = layers.get_layer(obj, obj.sx2layers['Roughness'])
            # Static colors layer is rough
            rgh_value = (obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness)
            colors1 = generate.color_list(obj, rgh_value, utils.find_color_layers(obj, 5))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Combine with roughness from PBR material
            colors1 = generate.color_list(obj, palette[2], utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Combine roughness base mask with custom curvature gradient
            scene.normalizeconvex = False
            scene.normalizeconcave = False
            scene.ramplist = 'CURVATUREROUGHNESS'
            colors1 = generate.curvature_list(obj, objs)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'MUL', 0.2)
            # Combine previous mix with directional dust
            scene.ramplist = 'DIRECTIONALDUST'
            scene.dirAngle = 0.0
            scene.dirInclination = 90.0
            scene.dirCone = 30
            colors1 = generate.direction_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ADD', 0.4)
            # Emissives are smooth
            color = (0.0, 0.0, 0.0, 1.0)
            colors1 = generate.color_list(obj, color, obj.sx2layers['Emission'])
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Write roughness
            layer = obj.sx2layers['Roughness']
            layers.set_layer(obj, colors, layer)

            # Apply PBR metal based on layer7
            colors = generate.color_list(obj, color=palette[0], masklayer=utils.find_color_layers(obj, 6))
            if colors is not None:
                layers.set_layer(obj, colors, obj.sx2layers['Metallic'])

            # Windows are smooth and emissive
            colors = generate.color_list(obj, color=(0.0, 0.0, 0.0, 1.0), masklayer=utils.find_color_layers(obj, 6))
            base = layers.get_layer(obj, obj.sx2layers['Roughness'])
            colors = tools.blend_values(colors, base, 'ALPHA', 1.0)
            if colors is not None:
                layers.set_layer(obj, colors, obj.sx2layers['Roughness'])
            emission_base = generate.color_list(obj, color=(1.0, 1.0, 1.0, 1.0), masklayer=utils.find_color_layers(obj, 6))
            if emission_base is not None:
                layers.set_layer(obj, emission_base, obj.sx2layers['Emission'])
                emission = generate.emission_list(obj, 200)
                layers.set_layer(obj, emission, obj.sx2layers['Emission'])

        for obj in objs_windows:
            obj.hide_viewport = False


    def process_trees(self, objs):
        scene = bpy.context.scene.sx2
        obj = objs[0]

        self.apply_palette_overrides(objs, clear=True)
        self.apply_occlusion(objs)
        self.apply_curvature_overlay(objs, convex=True, concave=False, noise=0.03)

        # Apply thickness to transmission
        layers.clear_layers(objs, obj.sx2layers['Transmission'])
        scene.ramplist = 'FOLIAGERAMP'
        color = (0.0, 0.0, 0.0, 1.0)

        for obj in objs:
            colors = generate.thickness_list(obj, 250, masklayer=None)
            values = layers.get_luminances(obj, colors=colors)
            colors = generate.luminance_remap_list(obj, values=values)
            # Mask trunks out
            colors1 = generate.color_list(obj, color=color, masklayer=utils.find_color_layers(obj, 3))
            colors2 = generate.color_list(obj, color=color, masklayer=utils.find_color_layers(obj, 4))
            colors1 = tools.blend_values(colors2, colors1, 'ALPHA', 1.0)
            colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
            layers.set_layer(obj, colors, obj.sx2layers['Transmission'])

            # Get roughness base
            colors = layers.get_layer(obj, obj.sx2layers['Roughness'])
            # Static colors layer is rough
            rgh_value = (obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness, obj.sx2.static_roughness)
            colors1 = generate.color_list(obj, rgh_value, utils.find_color_layers(obj, 5))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            colors1 = generate.noise_list(obj, 0.01, True)
            colors = tools.blend_values(colors1, colors, 'OVR', 1.0)
            # Combine previous mix with directional dust
            scene.ramplist = 'DIRECTIONALDUST'
            scene.dirAngle = 0.0
            scene.dirInclination = 90.0
            scene.dirCone = 30
            colors1 = generate.direction_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ADD', 0.2)
            # Write roughness
            layer = obj.sx2layers['Roughness']
            layers.set_layer(obj, colors, layer)


    def __del__(self):
        print('SX Tools: Exiting magic')


# ------------------------------------------------------------------------
#    Scene Setup
# ------------------------------------------------------------------------
class SXTOOLS2_setup(object):
    def __init__(self):
        return None


    def create_tiler(self):

        def connect_nodes(output, input):
            nodetree.links.new(output, input)

        nodetree = bpy.data.node_groups.new(type='GeometryNodeTree', name='sx_tiler')
        nodetree.is_modifier = True
        group_in = nodetree.nodes.new(type='NodeGroupInput')
        group_in.name = 'group_input'
        group_in.location = (-800, 0)
        group_out = nodetree.nodes.new(type='NodeGroupOutput')
        group_out.name = 'group_output'
        group_out.location = (1200, 0)

        # create tiler inputs and outputs
        axis_switches = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']

        nodetree.interface.new_socket(in_out='INPUT', name='Geometry', socket_type='NodeSocketGeometry')
        nodetree.interface.new_socket(in_out='INPUT', name='Offset', socket_type='NodeSocketFloat')
        nodetree.interface.new_socket(in_out='OUTPUT', name='Geometry', socket_type='NodeSocketGeometry')
        for axis in axis_switches:
            nodetree.interface.new_socket(in_out='INPUT', name=axis, socket_type='NodeSocketBool')

        bbx = nodetree.nodes.new(type='GeometryNodeBoundBox')
        bbx.name = 'bbx'
        bbx.location = (-600, 400)

        flip = nodetree.nodes.new(type='GeometryNodeFlipFaces')
        flip.name = 'flip'
        flip.location = (-400, 0)

        # offset multiplier for mirror objects
        offset = nodetree.nodes.new(type='ShaderNodeValue')
        offset.name = 'offset'
        offset.location = (-600, 300)
        offset.outputs[0].default_value = 4.0

        # join all mirror copies prior to output
        join = nodetree.nodes.new(type='GeometryNodeJoinGeometry')
        join.name = 'join'
        join.location = (1000, 0)

        # link base mesh to bbx and flip faces
        connect_nodes(group_in.outputs['Geometry'], nodetree.nodes['bbx'].inputs['Geometry'])
        connect_nodes(group_in.outputs['Geometry'], nodetree.nodes['flip'].inputs['Mesh'])

        for i in range(2):
            # bbx offset multipliers for mirror objects
            multiply = nodetree.nodes.new(type='ShaderNodeVectorMath')
            multiply.name = 'multiply'+str(i)
            multiply.location = (-400, 400-100*i)
            multiply.operation = 'MULTIPLY'

            # bbx min and max separators
            separate = nodetree.nodes.new(type='ShaderNodeSeparateXYZ')
            separate.name = 'separate'+str(i)
            separate.location = (0, 400-100*i)

            # link offset to multipliers
            connect_nodes(offset.outputs['Value'], multiply.inputs[1])

            # link bbx bounds to multipliers
            connect_nodes(bbx.outputs[i+1], multiply.inputs[0])

        bbx_subtract = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bbx_subtract.name = 'bbx_subtract'
        bbx_subtract.location = (-200, 400)
        bbx_subtract.operation = 'SUBTRACT'

        bbx_add = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bbx_add.name = 'bbx_add'
        bbx_add.location = (-200, 300)
        bbx_add.operation = 'ADD'

        # combine node for offset
        combine_offset = nodetree.nodes.new(type='ShaderNodeCombineXYZ')
        combine_offset.name = 'combine_offset'
        combine_offset.location = (-600, 100)

        # expose offset, connect to bbx offsets
        connect_nodes(group_in.outputs['Offset'], combine_offset.inputs['X'])
        connect_nodes(group_in.outputs['Offset'], combine_offset.inputs['Y'])
        connect_nodes(group_in.outputs['Offset'], combine_offset.inputs['Z'])

        connect_nodes(combine_offset.outputs['Vector'], bbx_subtract.inputs[1])
        connect_nodes(combine_offset.outputs['Vector'], bbx_add.inputs[1])

        # link multipliers to bbx offsets
        connect_nodes(nodetree.nodes['multiply0'].outputs['Vector'], nodetree.nodes['bbx_subtract'].inputs[0])
        connect_nodes(nodetree.nodes['multiply1'].outputs['Vector'], nodetree.nodes['bbx_add'].inputs[0])

        # bbx to separate min and max
        connect_nodes(nodetree.nodes['bbx_subtract'].outputs['Vector'], nodetree.nodes['separate0'].inputs[0])
        connect_nodes(nodetree.nodes['bbx_add'].outputs['Vector'], nodetree.nodes['separate1'].inputs[0])

        for i in range(6):
            # combine nodes for mirror offsets
            combine = nodetree.nodes.new(type='ShaderNodeCombineXYZ')
            combine.name = 'combine'+str(i)
            combine.location = (400, -100*i)

            # mirror axis enable switches
            switch = nodetree.nodes.new(type="GeometryNodeSwitch")
            switch.name = 'switch'+str(i)
            switch.location = (600, -100*i)

            # transforms for mirror copies
            transform = nodetree.nodes.new(type='GeometryNodeTransform')
            transform.name = 'transform'+str(i)
            transform.location = (800, -100*i)

            # link combine to translation
            connect_nodes(combine.outputs["Vector"], transform.inputs[1])

            # expose axis enable switches
            connect_nodes(group_in.outputs[axis_switches[i]], switch.inputs['Switch'])

            # link flipped mesh to switch input
            connect_nodes(flip.outputs['Mesh'], switch.inputs['True'])

            # link switch mesh output to transform
            connect_nodes(switch.outputs['Output'], transform.inputs['Geometry'])

            # link mirror mesh output to join node
            connect_nodes(transform.outputs['Geometry'], join.inputs[0])

        # link group in as first in join node for working auto-smooth
        connect_nodes(group_in.outputs['Geometry'], join.inputs[0])

        # link combined geometry to group output
        connect_nodes(nodetree.nodes['join'].outputs['Geometry'], group_out.inputs['Geometry'])

        # min max pairs to combiners in -x, x, -y, y, -z, z order
        connect_nodes(nodetree.nodes['separate0'].outputs['X'], nodetree.nodes['combine0'].inputs['X'])
        connect_nodes(nodetree.nodes['separate1'].outputs['X'], nodetree.nodes['combine1'].inputs['X'])
        connect_nodes(nodetree.nodes['separate0'].outputs['Y'], nodetree.nodes['combine2'].inputs['Y'])
        connect_nodes(nodetree.nodes['separate1'].outputs['Y'], nodetree.nodes['combine3'].inputs['Y'])
        connect_nodes(nodetree.nodes['separate0'].outputs['Z'], nodetree.nodes['combine4'].inputs['Z'])
        connect_nodes(nodetree.nodes['separate1'].outputs['Z'], nodetree.nodes['combine5'].inputs['Z'])

        nodetree.nodes['transform0'].inputs['Scale'].default_value[0] = -3.0
        nodetree.nodes['transform1'].inputs['Scale'].default_value[0] = -3.0
        nodetree.nodes['transform2'].inputs['Scale'].default_value[1] = -3.0
        nodetree.nodes['transform3'].inputs['Scale'].default_value[1] = -3.0
        nodetree.nodes['transform4'].inputs['Scale'].default_value[2] = -3.0
        nodetree.nodes['transform5'].inputs['Scale'].default_value[2] = -3.0


    def create_occlusion_network(self, raycount):
        def ray_randomizer(count):
            hemisphere = [None] * count
            random.seed(sxglobals.randomseed)

            for i in range(count):
                r = math.sqrt(random.random())
                theta = 2 * math.pi * random.random()
                hemisphere[i] = (r, theta, 0)

            sorted_hemisphere = sorted(hemisphere, key=lambda x: x[1], reverse=True)
            return sorted_hemisphere


        def connect_nodes(output, input):
            nodetree.links.new(output, input)


        nodetree = bpy.data.node_groups.new(type='GeometryNodeTree', name='sx_ao')
        nodetree.is_modifier = True

        # expose bias and ground plane inputs
        group_in = nodetree.nodes.new(type='NodeGroupInput')
        group_in.name = 'group_input'
        group_in.location = (-1000, 0)


        geometry = nodetree.interface.new_socket(in_out='INPUT', name='Geometry', socket_type='NodeSocketGeometry')
        bias = nodetree.interface.new_socket(in_out='INPUT', name='Ray Bias', socket_type='NodeSocketFloat')
        bias.min_value = 0
        bias.max_value = 1
        bias.default_value = 0.001

        ground_plane = nodetree.interface.new_socket(in_out='INPUT', name='Ground Plane', socket_type='NodeSocketBool')
        ground_plane.default_value = False

        ground_offset = nodetree.interface.new_socket(in_out='INPUT', name='Ground Plane Offset', socket_type='NodeSocketFloat')
        ground_offset.default_value = 0

        # expose group color output
        group_out = nodetree.nodes.new(type='NodeGroupOutput')
        group_out.name = 'group_output'
        group_out.location = (2000, 0)

        geometry_out = nodetree.interface.new_socket(in_out='OUTPUT', name='Geometry', socket_type='NodeSocketGeometry')
        color_out = nodetree.interface.new_socket(in_out='OUTPUT', name='Color Output', socket_type='NodeSocketColor')
        color_out.attribute_domain = 'POINT'
        color_out.default_attribute_name = 'Occlusion'
        color_out.default_value = (1, 1, 1, 1)

        # vertex inputs
        index = nodetree.nodes.new(type='GeometryNodeInputIndex')
        index.name = 'index'
        index.location = (-1000, 200)
        index.hide = True

        normal = nodetree.nodes.new(type='GeometryNodeInputNormal')
        normal.name = 'normal'
        normal.location = (-1000, 150)
        normal.hide = True

        position = nodetree.nodes.new(type='GeometryNodeInputPosition')
        position.name = 'position'
        position.location = (-1000, 100)
        position.hide = True

        eval_normal = nodetree.nodes.new(type='GeometryNodeFieldAtIndex')
        eval_normal.name = 'eval_normal'
        eval_normal.location = (-800, 200)
        eval_normal.data_type = 'FLOAT_VECTOR'
        eval_normal.domain = 'POINT'
        eval_normal.hide = True

        eval_position = nodetree.nodes.new(type='GeometryNodeFieldAtIndex')
        eval_position.name = 'eval_position'
        eval_position.location = (-800, 150)
        eval_position.data_type = 'FLOAT_VECTOR'
        eval_position.domain = 'POINT'
        eval_position.hide = True

        bias_normal = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bias_normal.name = 'bias_normal'
        bias_normal.location = (-600, 200)
        bias_normal.operation = 'MULTIPLY'
        bias_normal.hide = True

        bias_pos = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bias_pos.name = 'bias_pos'
        bias_pos.location = (-400, 200)
        bias_pos.operation = 'ADD'
        bias_pos.hide = True

        connect_nodes(group_in.outputs['Geometry'], group_out.inputs['Geometry'])

        connect_nodes(index.outputs['Index'], eval_normal.inputs['Index'])
        connect_nodes(index.outputs['Index'], eval_position.inputs['Index'])
        connect_nodes(normal.outputs['Normal'], eval_normal.inputs['Value'])
        connect_nodes(position.outputs['Position'], eval_position.inputs['Value'])

        connect_nodes(eval_normal.outputs['Value'], bias_normal.inputs[0])
        connect_nodes(group_in.outputs['Ray Bias'], bias_normal.inputs[1])

        connect_nodes(bias_normal.outputs['Vector'], bias_pos.inputs[0])
        connect_nodes(eval_position.outputs['Value'], bias_pos.inputs[1])


        # optional ground plane
        bbx = nodetree.nodes.new(type='GeometryNodeBoundBox')
        bbx.name = 'bbx'
        bbx.location = (-800, -200)
        bbx.hide = True

        bbx_min_separate = nodetree.nodes.new(type='ShaderNodeSeparateXYZ')
        bbx_min_separate.name = 'bbx_min_separate'
        bbx_min_separate.location = (-600, -200)
        bbx_min_separate.hide = True

        ground_bias = nodetree.nodes.new(type='ShaderNodeMath')
        ground_bias.name = 'ground_bias'
        ground_bias.location = (-400, -150)
        ground_bias.operation = 'SUBTRACT'
        ground_bias.inputs[1].default_value = 0.001
        ground_bias.hide = True

        ground_offset_add = nodetree.nodes.new(type='ShaderNodeMath')
        ground_offset_add.name = 'add_hits'
        ground_offset_add.location = (-400, -200)
        ground_offset_add.operation = 'ADD'
        ground_offset_add.inputs[1].default_value = 0
        ground_offset_add.hide = True

        bbx_min_combine = nodetree.nodes.new(type='ShaderNodeCombineXYZ')
        bbx_min_combine.name = 'bbx_min_combine'
        bbx_min_combine.location = (-200, -200)
        bbx_min_combine.hide = True

        bbx_multiply = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bbx_multiply.name = 'bbx_multiply'
        bbx_multiply.location = (-600, -250)
        bbx_multiply.operation = 'MULTIPLY'
        bbx_multiply.inputs[1].default_value = (10, 10, 10)
        bbx_multiply.hide = True

        ground_grid = nodetree.nodes.new(type='GeometryNodeMeshGrid')
        ground_grid.name = 'ground_grid'
        ground_grid.location = (-200, -150)
        ground_grid.inputs[2].default_value = 2
        ground_grid.inputs[3].default_value = 2
        ground_grid.hide = True

        ground_transform = nodetree.nodes.new(type='GeometryNodeTransform')
        ground_transform.name = 'ground_transform'
        ground_transform.location = (0, -200)
        ground_transform.hide = True

        join = nodetree.nodes.new(type='GeometryNodeJoinGeometry')
        join.name = 'join'
        join.location = (200, -150)
        join.hide = True

        ground_switch = nodetree.nodes.new(type='GeometryNodeSwitch')
        ground_switch.name = 'ground_switch'
        ground_switch.location = (400, -100)
        ground_switch.input_type = 'GEOMETRY'
        ground_switch.hide = True

        connect_nodes(group_in.outputs['Geometry'], bbx.inputs['Geometry'])
        connect_nodes(bbx.outputs['Min'], bbx_min_separate.inputs[0])
        connect_nodes(group_in.outputs['Ground Plane Offset'], ground_bias.inputs[0])
        connect_nodes(ground_bias.outputs[0], ground_offset_add.inputs[0])
        connect_nodes(bbx_min_separate.outputs['Z'], ground_offset_add.inputs[1])
        connect_nodes(ground_offset_add.outputs[0], bbx_min_combine.inputs['Z'])
        connect_nodes(bbx.outputs['Max'], bbx_multiply.inputs[0])
        connect_nodes(bbx_min_combine.outputs[0], ground_transform.inputs['Translation'])
        connect_nodes(bbx_multiply.outputs[0], ground_transform.inputs['Scale'])
        connect_nodes(ground_grid.outputs['Mesh'], ground_transform.inputs['Geometry'])
        connect_nodes(group_in.outputs['Geometry'], join.inputs[0])
        connect_nodes(ground_transform.outputs['Geometry'], join.inputs[0])
        connect_nodes(group_in.outputs['Ground Plane'], ground_switch.inputs['Switch'])
        connect_nodes(group_in.outputs['Geometry'], ground_switch.inputs['False'])
        connect_nodes(join.outputs['Geometry'], ground_switch.inputs['True'])

        # create raycasts with a loop
        hemisphere = ray_randomizer(raycount)
        previous = None
        for i in range(raycount):
            random_rot = nodetree.nodes.new(type='ShaderNodeVectorRotate')
            random_rot.name = 'random_rot'
            random_rot.location = (600, i * 100 + 400)
            random_rot.rotation_type = 'EULER_XYZ'
            random_rot.inputs[4].default_value = hemisphere[i]
            random_rot.hide = True

            raycast = nodetree.nodes.new(type='GeometryNodeRaycast')
            raycast.name = 'raycast'
            raycast.location = (800, i * 100 + 400)
            raycast.inputs['Ray Length'].default_value = 10
            raycast.hide = True

            if previous is not None:
                add_hits = nodetree.nodes.new(type='ShaderNodeMath')
                add_hits.name = 'add_hits'
                add_hits.location = (1000, i * 100 + 400)
                add_hits.operation = 'ADD'
                add_hits.hide = True
                connect_nodes(raycast.outputs[0], add_hits.inputs[0])
                connect_nodes(previous.outputs[0], add_hits.inputs[1])
                previous = add_hits
            else:
                previous = raycast

            connect_nodes(eval_normal.outputs['Value'], random_rot.inputs['Vector'])
            connect_nodes(bias_pos.outputs['Vector'], random_rot.inputs['Center'])
            connect_nodes(ground_switch.outputs['Output'], raycast.inputs['Target Geometry'])
            connect_nodes(random_rot.outputs[0], raycast.inputs['Ray Direction'])
            connect_nodes(bias_pos.outputs['Vector'], raycast.inputs['Source Position'])


        # normalize hit results
        div_hits = nodetree.nodes.new(type='ShaderNodeMath')
        div_hits.name = 'add_hits'
        div_hits.location = (1400, 300)
        div_hits.operation = 'DIVIDE'
        div_hits.inputs[1].default_value = raycount

        color_mix = nodetree.nodes.new(type='ShaderNodeMix')
        color_mix.name = 'color_mix'
        color_mix.data_type = 'RGBA'
        color_mix.location = (1600, 300)
        color_mix.inputs[6].default_value = (1, 1, 1, 1)
        color_mix.inputs[7].default_value = (0, 0, 0, 1)

        connect_nodes(previous.outputs[0], div_hits.inputs[0])
        connect_nodes(div_hits.outputs[0], color_mix.inputs[0])
        connect_nodes(color_mix.outputs[2], group_out.inputs['Color Output'])


    def create_sxtoolmaterial(self):
        sxmaterial = bpy.data.materials.new(name='SXToolMaterial')
        sxmaterial.use_nodes = True

        sxmaterial.node_tree.nodes.remove(sxmaterial.node_tree.nodes['Principled BSDF'])
        sxmaterial.node_tree.nodes['Material Output'].location = (-200, 0)

        # Gradient tool color ramp
        ramp = sxmaterial.node_tree.nodes.new(type='ShaderNodeValToRGB')
        ramp.name = 'Color Ramp'
        ramp.label = 'Color Ramp'
        ramp.location = (-1000, 0)


    def create_sx2material(self, objs):

        def connect_nodes(output, input):
            sxmaterial.node_tree.links.new(output, input)

        blend_mode_dict = {'ALPHA': 'MIX', 'OVR': 'OVERLAY', 'MUL': 'MULTIPLY', 'ADD': 'ADD', 'SUB': 'SUBTRACT'}
        scene = bpy.context.scene.sx2

        for obj in objs:
            if obj.sx2layers:
                sxmaterial = bpy.data.materials.new(name='SX2Material_'+obj.name)
                sxmaterial.use_nodes = True
                sxmaterial.node_tree.nodes['Principled BSDF'].inputs[26].default_value = [0.0, 0.0, 0.0, 1.0]
                sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = [0.0, 0.0, 0.0, 1.0]
                sxmaterial.node_tree.nodes['Principled BSDF'].location = (1000, 0)
                sxmaterial.node_tree.nodes['Material Output'].location = (1300, 0)

                if (obj.sx2.shadingmode == 'FULL') or (obj.sx2.shadingmode == 'XRAY'):
                    sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Anisotropic'].default_value = obj.sx2.mat_anisotropic
                    sxmaterial.node_tree.nodes['Principled BSDF'].inputs[18].default_value = obj.sx2.mat_clearcoat
                    prev_color = None
                    prev_alpha = None
                    rgba_mats = ['SSS', 'EMI']
                    alpha_mats = ['OCC', 'MET', 'RGH', 'TRN']
                    color_layers = []
                    material_layers = []
                    if len(sxglobals.layer_stack_dict.get(obj.name, {})) > 0:
                        color_stack = [sxglobals.layer_stack_dict[obj.name][key] for key in sorted(sxglobals.layer_stack_dict[obj.name].keys())]
                        for layer in color_stack:
                            if layer[2] == 'COLOR':
                                color_layers.append(layer)
                            else:
                                material_layers.append(layer)

                    # Create layer inputs
                    paletted_shading = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                    paletted_shading.name = 'Paletted Shading'
                    paletted_shading.label = 'Paletted Shading'
                    paletted_shading.attribute_name = 'sx2.palettedshading'
                    paletted_shading.attribute_type = 'OBJECT'
                    paletted_shading.location = (-2000, 0)

                    for i in range(math.ceil(len(color_layers)/3)):
                        input_vector = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                        input_vector.name = 'Input Vector ' + str(i)
                        input_vector.label = 'Input Vector ' + str(i)
                        input_vector.attribute_name = 'sx2.layer_opacity_list_' + str(i)
                        input_vector.attribute_type = 'OBJECT'
                        input_vector.location = (-2000, i*200+400)

                        input_splitter = sxmaterial.node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
                        input_splitter.name = 'Input Splitter ' + str(i)
                        input_splitter.label = 'Input Splitter ' + str(i)
                        input_splitter.location = (-1800, i*200+400)

                        connect_nodes(input_vector.outputs['Vector'], input_splitter.inputs['Vector'])

                    # Set shader blend method
                    if ((len(color_layers) > 0) and (obj.sx2layers[color_layers[0][0]].opacity < 1.0)) or (obj.sx2.shadingmode == 'XRAY'):
                        if obj.sx2.backfaceculling:
                            sxmaterial.blend_method = 'BLEND'
                        else:
                            sxmaterial.blend_method = 'HASHED'
                        sxmaterial.use_backface_culling = obj.sx2.backfaceculling
                        sxmaterial.show_transparent_back = not obj.sx2.backfaceculling
                    else:
                        sxmaterial.blend_method = 'OPAQUE'
                        sxmaterial.use_backface_culling = True
                        sxmaterial.show_transparent_back = False

                    # Create color layer groups
                    splitter_index = 0
                    count = 0
                    for i in range(len(color_layers)):
                        if obj.sx2layers[color_layers[i][0]].visibility:
                            source_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                            source_color.name = 'LayerColor' + str(i)
                            source_color.label = 'LayerColor' + str(i)
                            source_color.layer_name = color_layers[i][1]
                            source_color.location = (-1000, i*400+400)

                            layer_palette = sxmaterial.node_tree.nodes.new(type="ShaderNodeRGB")
                            layer_palette.name = 'PaletteColor' + color_layers[i][0]
                            layer_palette.label = 'PaletteColor' + color_layers[i][0]
                            layer_palette.outputs[0].default_value = getattr(scene, 'newpalette'+str(obj.sx2layers[color_layers[i][0]].palette_index))
                            layer_palette.location = (-1000, i*400+250)

                            palette_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                            palette_blend.inputs[0].default_value = 0
                            palette_blend.use_clamp = True
                            palette_blend.location = (-800, i*400+250)

                            connect_nodes(source_color.outputs['Color'], palette_blend.inputs['Color1'])
                            connect_nodes(layer_palette.outputs['Color'], palette_blend.inputs['Color2'])

                            if obj.sx2layers[color_layers[i][0]].paletted:
                                connect_nodes(paletted_shading.outputs[2], palette_blend.inputs[0])

                            opacity_and_alpha = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                            opacity_and_alpha.name = 'Opacity ' + str(i)
                            opacity_and_alpha.label = 'Opacity ' + str(i)
                            opacity_and_alpha.operation = 'MULTIPLY'
                            opacity_and_alpha.use_clamp = True
                            opacity_and_alpha.inputs[0].default_value = 0
                            opacity_and_alpha.location = (-800, i*400+400)

                            layer_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                            layer_blend.name = 'Mix ' + str(i)
                            layer_blend.label = 'Mix ' + str(i)
                            layer_blend.inputs[0].default_value = 1
                            layer_blend.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                            layer_blend.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                            layer_blend.blend_type = blend_mode_dict[obj.sx2layers[color_layers[i][0]].blend_mode]
                            layer_blend.use_clamp = True
                            layer_blend.location = (-600, i*400+400)

                            alpha_accumulation = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                            alpha_accumulation.name = 'Accumulation ' + str(i)
                            alpha_accumulation.label = 'Accumulation ' + str(i)
                            alpha_accumulation.operation = 'ADD'
                            alpha_accumulation.use_clamp = True
                            alpha_accumulation.inputs[0].default_value = 0.0
                            alpha_accumulation.inputs[1].default_value = 0.0
                            alpha_accumulation.location = (-600, i*400+250)

                            # Group connections
                            connect_nodes(palette_blend.outputs['Color'], layer_blend.inputs['Color2'])
                            connect_nodes(sxmaterial.node_tree.nodes['Input Splitter ' + str(splitter_index)].outputs[(i) % 3], opacity_and_alpha.inputs[0])
                            connect_nodes(source_color.outputs['Alpha'], opacity_and_alpha.inputs[1])
                            connect_nodes(opacity_and_alpha.outputs[0], layer_blend.inputs[0])

                            if (layer_blend.blend_type != 'MULTIPLY') and (layer_blend.blend_type != 'OVERLAY'):
                                connect_nodes(opacity_and_alpha.outputs[0], alpha_accumulation.inputs[1])

                            if prev_color is not None:
                                connect_nodes(prev_color, layer_blend.inputs['Color1'])

                            if prev_alpha is not None:
                                connect_nodes(prev_alpha, alpha_accumulation.inputs[0])

                            prev_color = layer_blend.outputs['Color']
                            prev_alpha = alpha_accumulation.outputs[0]

                        count += 1
                        if count == 3:
                            splitter_index += 1
                            count = 0

                    # Create material inputs
                    for i in range(math.ceil(len(material_layers)/3)):
                        input_vector = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                        input_vector.name = 'Material Input Vector ' + str(i)
                        input_vector.label = 'Material Input Vector ' + str(i)
                        input_vector.attribute_name = 'sx2.material_opacity_list_' + str(i)
                        input_vector.attribute_type = 'OBJECT'
                        input_vector.location = (-1600, -i*200-400)

                        input_splitter = sxmaterial.node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
                        input_splitter.name = 'Material Input Splitter ' + str(i)
                        input_splitter.label = 'Material Input Splitter ' + str(i)
                        input_splitter.location = (-1400, -i*200-400)

                        connect_nodes(input_vector.outputs['Vector'], input_splitter.inputs['Vector'])

                    # Create material layers
                    # Single-channel material values are packed into 'Alpha Materials' color layer:
                    # R: Occlusion, G: Metallic, B: Roughness, A: Transmission
                    splitter_index = 0
                    count = 0
                    source_alphas = None

                    for i in range(len(material_layers)):
                        if obj.sx2layers[material_layers[i][0]].visibility:
                            if (material_layers[i][2] in alpha_mats) and (source_alphas is None):
                                source_mats = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                                source_mats.name = 'Alpha Materials'
                                source_mats.label = 'Alpha Materials'
                                source_mats.layer_name = 'Alpha Materials'
                                source_mats.location = (-400, -400)

                                source_alphas = sxmaterial.node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
                                source_alphas.name = 'Alpha Splitter'
                                source_alphas.label = 'Alpha Splitter'
                                source_alphas.location = (0, -400)

                                connect_nodes(source_mats.outputs['Color'], source_alphas.inputs['Vector'])

                            elif (material_layers[i][2] not in alpha_mats):
                                name = material_layers[i][0]

                                source_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                                source_color.name = name
                                source_color.label = name
                                source_color.layer_name = material_layers[i][1]
                                source_color.location = (0, -i*400-800)

                                source_mask = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                                source_mask.inputs[0].default_value = 0
                                source_mask.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                                source_mask.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                                source_mask.use_clamp = True
                                source_mask.location = (200, -i*400-800)

                                source_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                                source_blend.inputs[0].default_value = 0
                                source_blend.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                                source_blend.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                                source_blend.use_clamp = True
                                source_blend.location = (200, -i*400-950)

                                material_palette = sxmaterial.node_tree.nodes.new(type="ShaderNodeRGB")
                                material_palette.name = 'PaletteColor' + name
                                material_palette.label = 'PaletteColor' + name
                                material_palette.outputs[0].default_value = getattr(scene, 'newpalette'+str(obj.sx2layers[material_layers[i][0]].palette_index))
                                material_palette.location = (0, -i*400-950)

                                palette_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                                palette_blend.inputs[0].default_value = 0
                                palette_blend.use_clamp = True
                                palette_blend.location = (400, -i*400-800)

                                connect_nodes(source_color.outputs['Alpha'], source_mask.inputs[0])
                                connect_nodes(source_color.outputs['Color'], source_mask.inputs['Color2'])
                                connect_nodes(source_color.outputs['Alpha'], source_blend.inputs[0])
                                connect_nodes(material_palette.outputs['Color'], source_blend.inputs['Color2'])
                                connect_nodes(source_mask.outputs['Color'], palette_blend.inputs['Color1'])
                                connect_nodes(source_blend.outputs['Color'], palette_blend.inputs['Color2'])

                                if obj.sx2layers[material_layers[i][0]].paletted:
                                    connect_nodes(paletted_shading.outputs[2], palette_blend.inputs[0])

                            layer_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                            layer_blend.name = material_layers[i][0] + ' Mix'
                            layer_blend.label = material_layers[i][0] + ' Mix'
                            layer_blend.blend_type = 'MIX'
                            layer_blend.inputs[0].default_value = 1
                            layer_blend.inputs[1].default_value = obj.sx2layers[material_layers[i][0]].default_color
                            layer_blend.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                            layer_blend.use_clamp = True
                            layer_blend.location = (400, -i*400-400)

                            # Group connections
                            connect_nodes(sxmaterial.node_tree.nodes['Material Input Splitter ' + str(splitter_index)].outputs[i % 3], layer_blend.inputs[0])

                            count += 1
                            if count == 3:
                                splitter_index += 1
                                count = 0

                            if (material_layers[i][2] == 'OCC'):
                                layer_blend.blend_type = 'MULTIPLY'
                                if prev_color is not None:
                                    connect_nodes(prev_color, layer_blend.inputs['Color1'])
                                    prev_color = layer_blend.outputs['Color']

                                output = source_alphas.outputs['X']

                            elif (material_layers[i][2] == 'MET'):
                                output = source_alphas.outputs['Y']

                            elif (material_layers[i][2] == 'RGH'):
                                output = source_alphas.outputs['Z']

                            elif (material_layers[i][2] == 'TRN'):
                                output = source_mats.outputs['Alpha']

                            else:
                                output = palette_blend.outputs['Color']

                            connect_nodes(output, layer_blend.inputs[2])

                            output = layer_blend.outputs[0]
                            if material_layers[i][2] == 'MET':
                                connect_nodes(output, sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Metallic'])

                            if material_layers[i][2] == 'RGH':
                                connect_nodes(output, sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Roughness'])

                            if material_layers[i][2] == 'TRN':
                                connect_nodes(output, sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Transmission Weight'])

                            if material_layers[i][2] == 'SSS':
                                connect_nodes(output, sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Subsurface'])
                                connect_nodes(palette_blend.outputs['Color'], sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Subsurface Color'])

                            if material_layers[i][2] == 'EMI':
                                sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 10
                                connect_nodes(output, sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission Color'])

                    if prev_color is not None:
                        connect_nodes(prev_color,  sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color'])

                    if obj.sx2.shadingmode == 'XRAY':
                        sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 0.5
                    else:
                        if prev_alpha is not None:
                            connect_nodes(prev_alpha, sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Alpha'])

                elif (obj.sx2.shadingmode == 'DEBUG') or (obj.sx2.shadingmode == 'CHANNEL'):
                    sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Specular IOR Level'].default_value = 0.0
                    sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 1
                    alpha_mats = ['OCC', 'MET', 'RGH', 'TRN']
                    layer = obj.sx2layers[obj.sx2.selectedlayer]

                    # Create debug source
                    base_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                    base_color.name = 'DebugSource'
                    base_color.label = 'DebugSource'
                    base_color.layer_name = layer.color_attribute
                    base_color.location = (-1000, 0)

                    layer_opacity = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                    layer_opacity.name = 'Layer Opacity'
                    layer_opacity.label = 'Layer Opacity'
                    layer_opacity.attribute_name = 'sx2layers["' + layer.name + '"].opacity'
                    layer_opacity.attribute_type = 'OBJECT'
                    layer_opacity.location = (-1000, -400)

                    debug_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                    debug_blend.name = 'Debug Mix'
                    debug_blend.label = 'Debug Mix'
                    debug_blend.inputs[0].default_value = 1
                    debug_blend.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                    debug_blend.inputs[2].default_value = [1.0, 1.0, 1.0, 1.0]
                    debug_blend.blend_type = 'MIX'
                    debug_blend.use_clamp = True
                    debug_blend.location = (-600, 0)

                    if (base_color.layer_name == 'Alpha Materials') and (layer.layer_type != 'TRN'):
                        source_alphas = sxmaterial.node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
                        source_alphas.location = (-800, 0)

                        connect_nodes(base_color.outputs['Color'], source_alphas.inputs['Vector'])

                    if layer.layer_type not in alpha_mats:
                        opacity_and_alpha = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                        opacity_and_alpha.name = 'Opacity and Alpha'
                        opacity_and_alpha.label = 'Opacity and Alpha'
                        opacity_and_alpha.operation = 'MULTIPLY'
                        opacity_and_alpha.use_clamp = True
                        opacity_and_alpha.inputs[0].default_value = 1
                        opacity_and_alpha.inputs[1].default_value = 1
                        opacity_and_alpha.location = (-800, -400)

                        connect_nodes(layer_opacity.outputs[2], opacity_and_alpha.inputs[0])
                        if obj.sx2.shadingmode == 'DEBUG':
                            connect_nodes(base_color.outputs['Alpha'], opacity_and_alpha.inputs[1])
                        connect_nodes(opacity_and_alpha.outputs[0], debug_blend.inputs[0])

                    else:
                        connect_nodes(layer_opacity.outputs[2], debug_blend.inputs[0])

                    if layer.layer_type not in alpha_mats:
                        if obj.sx2.shadingmode == 'DEBUG':
                            output = base_color.outputs['Color']
                        else:
                            rgb_split = sxmaterial.node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
                            rgb_split.name = 'RBG Split'
                            rgb_split.label = 'RGB Split'
                            rgb_split.location = (-800, 150)
                            connect_nodes(base_color.outputs['Color'], rgb_split.inputs['Vector'])
                            if obj.sx2.debugmode == 'R':
                                output = rgb_split.outputs['X']
                            elif obj.sx2.debugmode == 'G':
                                output = rgb_split.outputs['Y']
                            elif obj.sx2.debugmode == 'B':
                                output = rgb_split.outputs['Z']
                            else:
                                output = base_color.outputs['Alpha']
                    elif layer.layer_type == 'OCC':
                        output = source_alphas.outputs['X']
                    elif layer.layer_type == 'MET':
                        output = source_alphas.outputs['Y']
                    elif layer.layer_type == 'RGH':
                        output = source_alphas.outputs['Z']
                    elif layer.layer_type == 'TRN':
                        output = base_color.outputs['Alpha']

                    connect_nodes(output, debug_blend.inputs['Color2'])
                    connect_nodes(debug_blend.outputs['Color'], sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission Color'])


    def update_sx2material(self, context):
        if (not sxglobals.magic_in_progress) and (not bpy.app.background):
            if 'SXToolMaterial' not in bpy.data.materials:
                setup.create_sxtoolmaterial()

            # Get all objects in the scene with sx2 data
            sx2_objs = []
            for obj in context.view_layer.objects:
                if (obj is not None) and (obj.type == 'MESH') and ('sx2layers' in obj.keys()):
                    sx2_objs.append(obj)

            sx2_objs = list(set(sx2_objs))

            # Clear sx2material_dict
            sxglobals.sx2material_dict.clear()

            # Store unique layer stacks in sx2material_dict
            for obj in sx2_objs:
                utils.sort_stack_indices(obj)
                obj.data.materials.clear()
                mat_id = utils.get_mat_id(obj)
                if mat_id not in sxglobals.sx2material_dict.keys():
                    sxglobals.sx2material_dict[mat_id] = (len(sxglobals.sx2material_dict), obj.name)

            # Get a list of all unique materials in sx2 objects
            # and remove them
            sx_mats = [mat for mat in bpy.data.materials if 'SX2Material_' in mat.name]

            if sx_mats:
                for mat in sx_mats:
                    bpy.data.materials.remove(mat, do_unlink=True)

            # bpy.context.view_layer.update()

            # Get reference objects for material update
            ref_objs = []
            for value in sxglobals.sx2material_dict.values():
                obj = context.view_layer.objects.get(value[1])
                if obj:
                    ref_objs.append(obj)

            # Create materials for group reference objects
            setup.create_sx2material(ref_objs)

            # Assign materials based on matching layer stacks
            for obj in sx2_objs:
                mat_id = utils.get_mat_id(obj)
                if (len(obj.sx2layers) > 0) and (mat_id in sxglobals.sx2material_dict):
                    obj.active_material = bpy.data.materials[utils.find_sx2material_name(obj)]


    def reset_scene(self):
        sxglobals.refresh_in_progress = True
        sxglobals.magic_in_progress = True
        sxglobals.layer_update_in_progress = True

        objs = bpy.context.view_layer.objects
        scene = bpy.context.scene

        # Clear layers and UV sets
        for obj in objs:
            remove_layer_list = []
            remove_uv_list = []
            if (obj.type == 'MESH') and ('sx2layers' in obj.keys()) and (len(obj.sx2layers) > 0):
                for layer in obj.sx2layers:
                    remove_layer_list.append(layer.name)

                for layer_name in remove_layer_list:
                    layers.del_layer([obj, ], layer_name)

                for uv_layer in obj.data.uv_layers:
                    if 'UVSet' in uv_layer.name:
                        remove_uv_list.append(uv_layer.name)

                for uv_layer_name in remove_uv_list:
                    obj.data.uv_layers.remove(obj.data.uv_layers[uv_layer_name])

            # Reset object properties
            obj.sx2.selectedlayer = 0
            obj.sx2.layercount = 0
            obj.sx2.palettedshading = 0.0
            obj.sx2.shadingmode = 'FULL'
            obj.sx2.category = 'DEFAULT'

        # Reset scene properties
        scene.sx2.toolmode = 'COL'
        scene.sx2.toolopacity = 1.0
        scene.sx2.toolblend = 'ALPHA'

        # Clear SX2 Materials
        sx_mats = [mat for mat in bpy.data.materials if 'SX2Material_' in mat.name]

        if sx_mats:
            for mat in sx_mats:
                bpy.data.materials.remove(mat, do_unlink=True)

        if 'SXToolMaterial' in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials['SXToolMaterial'], do_unlink=True)

        # Clear globals
        sxglobals.libraries_status = False
        sxglobals.prev_selection = []
        sxglobals.prev_component_selection = []
        sxglobals.copy_buffer.clear()
        sxglobals.ramp_dict.clear()
        sxglobals.category_dict.clear()
        sxglobals.preset_lookup.clear()
        sxglobals.layer_stack_dict.clear()
        sxglobals.sx2material_dict.clear()
        sxglobals.master_palette_list = []
        sxglobals.material_list = []
        sxglobals.prev_mode = 'OBJECT'
        sxglobals.curvature_update = False
        sxglobals.hsl_update = False
        sxglobals.mat_update = False
        sxglobals.mode = 'OBJECT'
        sxglobals.mode_id = None

        sxglobals.layer_update_in_progress = False
        sxglobals.magic_in_progress = False
        sxglobals.refresh_in_progress = False

        sxglobals.selection_modal_status = False
        sxglobals.key_modal_status = False


    def __del__(self):
        print('SX Tools: Exiting setup')


# ------------------------------------------------------------------------
#    Update Functions
# ------------------------------------------------------------------------
def start_modal():
    if (not sxglobals.selection_modal_status) and (len(bpy.data.objects) > 0):
        bpy.ops.sx2.selectionmonitor('EXEC_DEFAULT')
        sxglobals.selection_modal_status = True

    if not sxglobals.key_modal_status:
        bpy.ops.sx2.keymonitor('EXEC_DEFAULT')
        sxglobals.key_modal_status = True


# Return value eliminates duplicates
def mesh_selection_validator(self, context):
    mesh_objs = []
    for obj in context.view_layer.objects.selected:
        if obj is None:
            pass
        elif (obj.type == 'MESH') and (obj.hide_viewport is False):
            mesh_objs.append(obj)
        elif obj.type == 'ARMATURE':
            mesh_objs += utils.find_children(obj, recursive=True, child_type='MESH')

    return list(set(mesh_objs))


def layer_validator(self, context):
    objs = mesh_selection_validator(self, context)
    if len(objs) > 1:
        ref_layers = objs[0].sx2layers.keys()
        return all(obj.sx2layers.keys() == ref_layers for obj in objs)
    return True


def refresh_swatches(self, context):
    if not bpy.app.background:
        # if not sxglobals.refresh_in_progress:
        #     sxglobals.refresh_in_progress = True

        scene = context.scene.sx2
        objs = mesh_selection_validator(self, context)

        if objs:
            obj = objs[0]
            mode = obj.sx2.shadingmode
            utils.mode_manager(objs, set_mode=True, mode_id='refresh_swatches')

            if obj.sx2layers:
                layer = obj.sx2layers[obj.sx2.selectedlayer]
                colors = utils.find_colors_by_frequency(objs, layer.name, 8, alphavalues=(mode == 'ALPHA'))

                # Refresh SX Tools UI to latest selection
                # 1) Update layer HSL elements
                hue, sat, lightness = 0, 0, 0
                for color in colors:
                    h, s, l = convert.rgb_to_hsl(color)
                    hue = h if h > hue else hue
                    sat = s if s > sat else sat
                    lightness = l if l > lightness else lightness

                sxglobals.hsl_update = True
                obj.sx2.huevalue = hue
                obj.sx2.saturationvalue = sat
                obj.sx2.lightnessvalue = lightness
                sxglobals.hsl_update = False

                # 2) Update layer palette elements
                for i, pcol in enumerate(colors):
                    palettecolor = (pcol[0], pcol[1], pcol[2], 1.0)
                    setattr(scene, 'layerpalette' + str(i + 1), palettecolor)

                # 3) update tool palettes
                if scene.toolmode == 'COL':
                    update_fillcolor(self, context)
                # elif (scene.toolmode == 'MAT'):
                #     layers.material_layers_to_swatches(objs)

            utils.mode_manager(objs, set_mode=False, mode_id='refresh_swatches')
            # sxglobals.refresh_in_progress = False


def message_box(message='', title='SX Tools', icon='INFO'):
    messageLines = message.splitlines()


    def draw(self, context):
        for line in messageLines:
            self.layout.label(text=line)

    if not bpy.app.background:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def update_paletted_layers(self, context, index):
    objs = mesh_selection_validator(self, context)
    scene = context.scene.sx2
    for obj in objs:
        # If update_paletted_layers is called during object category change, materials may not exist
        mat_name = utils.find_sx2material_name(obj)
        if (mat_name is not None) and ((obj.sx2.shadingmode == 'FULL') or (obj.sx2.shadingmode == 'XRAY')):
            for layer in obj.sx2layers:
                if layer.paletted:
                    color = getattr(scene, 'newpalette'+str(layer.palette_index))
                    bpy.data.materials[mat_name].node_tree.nodes['PaletteColor' + str(layer.name)].outputs[0].default_value = color[:]


def update_material_layer(self, context, index):
    if not sxglobals.mat_update:
        sxglobals.mat_update = True

        scene = context.scene.sx2
        objs = mesh_selection_validator(self, context)
        utils.mode_manager(objs, set_mode=True, mode_id='update_material_layer')
        layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
        layer_ids = [layer.name, 'Metallic', 'Roughness']
        pbr_values = [scene.newmaterial0, scene.newmaterial1, scene.newmaterial2]
        scene.toolopacity = 1.0
        scene.toolblend = 'ALPHA'

        if scene.enablelimit:
            hsl = convert.rgb_to_hsl(pbr_values[index])
            if scene.limitmode == 'MET':
                minl = float(170.0/255.0)
                if hsl[2] < minl:
                    rgb = convert.hsl_to_rgb((hsl[0], hsl[1], minl))
                    pbr_values[index] = (rgb[0], rgb[1], rgb[2], 1.0)
            else:
                minl = float(50.0/255.0)
                maxl = float(240.0/255.0)
                if hsl[2] > maxl:
                    rgb = convert.hsl_to_rgb((hsl[0], hsl[1], maxl))
                    pbr_values[index] = (rgb[0], rgb[1], rgb[2], 1.0)
                elif hsl[2] < minl:
                    rgb = convert.hsl_to_rgb((hsl[0], hsl[2], minl))
                    pbr_values[index] = (rgb[0], rgb[1], rgb[2], 1.0)

        added_layers = False
        selected_layer = objs[0].sx2.selectedlayer
        for obj in objs:
            if 'Metallic' not in obj.sx2layers.keys():
                layers.add_layer([obj, ], name='Metallic', layer_type='MET')
                added_layers = True
            if 'Roughness' not in obj.sx2layers.keys():
                layers.add_layer([obj, ], name='Roughness', layer_type='RGH')
                added_layers = True
        if added_layers:
            objs[0].sx2.selectedlayer = selected_layer

        mask = objs[0].sx2layers[layer_ids[0]]  # if objs[0].sx2layers[layer_ids[0]].locked else None
        if sxglobals.mode == 'EDIT':
            modecolor = utils.find_colors_by_frequency(objs, layer_ids[index], numcolors=1, masklayer=mask)[0]
        else:
            modecolor = utils.find_colors_by_frequency(objs, layer_ids[index], numcolors=1, masklayer=mask)[0]

        if not utils.color_compare(modecolor, pbr_values[index]):
            tools.apply_tool(objs, objs[0].sx2layers[layer_ids[index]], masklayer=mask, color=pbr_values[index])
            setattr(scene, 'newmaterial' + str(index), pbr_values[index])

        utils.mode_manager(objs, set_mode=False, mode_id='update_material_layer')
        sxglobals.mat_update = False
        refresh_swatches(self, context)


# With multi-object selection, this requires identical layersets
def update_selected_layer(self, context):
    if (not sxglobals.magic_in_progress) and (not bpy.app.background):
        objs = mesh_selection_validator(self, context)
        for obj in objs:
            if obj.sx2layers:
                obj.data.attributes.active_color = obj.data.attributes[obj.sx2layers[obj.sx2.selectedlayer].color_attribute]

        areas = bpy.context.workspace.screens[0].areas
        shading_types = ['MATERIAL', 'RENDERED']  # 'WIREFRAME' 'SOLID' 'MATERIAL' 'RENDERED'
        for area in areas:
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    if space.shading.type not in shading_types:
                        space.shading.type = 'MATERIAL'

        if not ((objs[0].sx2.shadingmode == 'FULL') or (objs[0].sx2.shadingmode == 'XRAY')):
            setup.update_sx2material(context)

        if 'SXToolMaterial' not in bpy.data.materials:
            setup.create_sxtoolmaterial()

        refresh_swatches(self, context)


def update_modifiers(self, context, prop):
    update_obj_props(self, context, prop)
    if not sxglobals.refresh_in_progress:
        sxglobals.refresh_in_progress = True

        objs = mesh_selection_validator(self, context)
        if objs:
            if prop == 'modifiervisibility':
                for obj in objs:
                    if 'sxMirror' in obj.modifiers:
                        obj.modifiers['sxMirror'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxSubdivision' in obj.modifiers:
                        if (obj.sx2.subdivisionlevel == 0):
                            obj.modifiers['sxSubdivision'].show_viewport = False
                        else:
                            obj.modifiers['sxSubdivision'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxDecimate' in obj.modifiers:
                        if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                            obj.modifiers['sxDecimate'].show_viewport = False
                        else:
                            obj.modifiers['sxDecimate'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxDecimate2' in obj.modifiers:
                        if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                            obj.modifiers['sxDecimate2'].show_viewport = False
                        else:
                            obj.modifiers['sxDecimate2'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxBevel' in obj.modifiers:
                        if obj.sx2.bevelsegments == 0:
                            obj.modifiers['sxBevel'].show_viewport = False
                        else:
                            obj.modifiers['sxBevel'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxWeld' in obj.modifiers:
                        if (obj.sx2.weldthreshold == 0.0):
                            obj.modifiers['sxWeld'].show_viewport = False
                        else:
                            obj.modifiers['sxWeld'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxWeightedNormal' in obj.modifiers:
                        if not obj.sx2.weightednormals:
                            obj.modifiers['sxWeightedNormal'].show_viewport = False
                        else:
                            obj.modifiers['sxWeightedNormal'].show_viewport = obj.sx2.modifiervisibility

            elif (prop == 'xmirror') or (prop == 'ymirror') or (prop == 'zmirror') or (prop == 'mirrorobject'):
                for obj in objs:
                    if 'sxMirror' in obj.modifiers:
                        obj.modifiers['sxMirror'].use_axis[0] = obj.sx2.xmirror
                        obj.modifiers['sxMirror'].use_axis[1] = obj.sx2.ymirror
                        obj.modifiers['sxMirror'].use_axis[2] = obj.sx2.zmirror

                        if obj.sx2.mirrorobject is not None:
                            obj.modifiers['sxMirror'].mirror_object = obj.sx2.mirrorobject
                        else:
                            obj.modifiers['sxMirror'].mirror_object = None

                        if obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror or (obj.sx2.mirrorobject is not None):
                            obj.modifiers['sxMirror'].show_viewport = True
                        else:
                            obj.modifiers['sxMirror'].show_viewport = False

            elif (prop == 'tiling') or ('tile_' in prop):
                for obj in objs:
                    if 'sxTiler' not in obj.modifiers:
                        modifiers.remove_modifiers([obj, ])
                        modifiers.add_modifiers([obj, ])
                    elif obj.modifiers['sxTiler'].node_group != bpy.data.node_groups['sx_tiler']:
                        obj.modifiers['sxTiler'].node_group = bpy.data.node_groups['sx_tiler']

                    obj.modifiers['sxTiler'].show_viewport = obj.sx2.tile_preview

                    if obj.sx2.tiling:
                        utils.round_tiling_verts([obj, ])
                        tiler = obj.modifiers['sxTiler']
                        tiler['Socket_1'] = obj.sx2.tile_offset
                        tiler['Socket_3'] = obj.sx2.tile_neg_x
                        tiler['Socket_4'] = obj.sx2.tile_pos_x
                        tiler['Socket_5'] = obj.sx2.tile_neg_y
                        tiler['Socket_6'] = obj.sx2.tile_pos_y
                        tiler['Socket_7'] = obj.sx2.tile_neg_z
                        tiler['Socket_8'] = obj.sx2.tile_pos_z

            elif prop == 'hardmode':
                for obj in objs:
                    if 'sxWeightedNormal' in obj.modifiers:
                        if obj.sx2.hardmode == 'SMOOTH':
                            obj.modifiers['sxWeightedNormal'].keep_sharp = False
                        else:
                            obj.modifiers['sxWeightedNormal'].keep_sharp = True

            elif prop == 'subdivisionlevel':
                for obj in objs:
                    if 'sxSubdivision' in obj.modifiers:
                        obj.modifiers['sxSubdivision'].levels = obj.sx2.subdivisionlevel
                        if obj.sx2.subdivisionlevel == 0:
                            obj.modifiers['sxSubdivision'].show_viewport = False
                        else:
                            obj.modifiers['sxSubdivision'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxDecimate' in obj.modifiers:
                        if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                            obj.modifiers['sxDecimate'].show_viewport = False
                        else:
                            obj.modifiers['sxDecimate'].show_viewport = obj.sx2.modifiervisibility
                    if 'sxDecimate2' in obj.modifiers:
                        if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                            obj.modifiers['sxDecimate2'].show_viewport = False
                        else:
                            obj.modifiers['sxDecimate2'].show_viewport = obj.sx2.modifiervisibility

            elif (prop == 'bevelwidth') or (prop == 'bevelsegments') or (prop == 'beveltype'):
                for obj in objs:
                    if 'sxBevel' in obj.modifiers:
                        if obj.sx2.bevelsegments == 0:
                            obj.modifiers['sxBevel'].show_viewport = False
                        else:
                            obj.modifiers['sxBevel'].show_viewport = obj.sx2.modifiervisibility
                        obj.modifiers['sxBevel'].width = obj.sx2.bevelwidth
                        obj.modifiers['sxBevel'].width_pct = obj.sx2.bevelwidth
                        obj.modifiers['sxBevel'].segments = obj.sx2.bevelsegments
                        obj.modifiers['sxBevel'].offset_type = obj.sx2.beveltype

            elif prop == 'weldthreshold':
                for obj in objs:
                    if 'sxWeld' in obj.modifiers:
                        obj.modifiers['sxWeld'].merge_threshold = obj.sx2.weldthreshold
                        if obj.sx2.weldthreshold == 0:
                            obj.modifiers['sxWeld'].show_viewport = False
                        elif (obj.sx2.weldthreshold > 0) and obj.sx2.modifiervisibility:
                            obj.modifiers['sxWeld'].show_viewport = True

            elif prop == 'decimation':
                for obj in objs:
                    if 'sxDecimate' in obj.modifiers:
                        obj.modifiers['sxDecimate'].angle_limit = math.radians(obj.sx2.decimation)
                        if obj.sx2.decimation == 0.0:
                            obj.modifiers['sxDecimate'].show_viewport = False
                            obj.modifiers['sxDecimate2'].show_viewport = False
                        elif (obj.sx2.decimation > 0) and obj.sx2.modifiervisibility:
                            obj.modifiers['sxDecimate'].show_viewport = True
                            obj.modifiers['sxDecimate2'].show_viewport = True

            elif prop == 'smoothangle':
                for obj in objs:
                    if 'sxSmoothNormals' in obj.modifiers:
                        obj.modifiers["sxSmoothNormals"]["Input_1"] = math.radians(obj.sx2.smoothangle)

            elif prop == 'weightednormals':
                for obj in objs:
                    if 'sxWeightedNormal' in obj.modifiers:
                        obj.modifiers['sxWeightedNormal'].show_viewport = obj.sx2.weightednormals

        sxglobals.refresh_in_progress = False


def adjust_hsl(self, context, hslmode):
    if not sxglobals.hsl_update:
        objs = mesh_selection_validator(self, context)

        if objs:
            hslvalues = [objs[0].sx2.huevalue, objs[0].sx2.saturationvalue, objs[0].sx2.lightnessvalue]
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            tools.apply_hsl(objs, layer, hslmode, hslvalues[hslmode])
            refresh_swatches(self, context)


def dict_lister(self, context, data_dict):
    items = data_dict.keys()
    enumItems = []
    for item in items:
        sxglobals.preset_lookup[item.replace(" ", "_").upper()] = item
        enumItem = (item.replace(" ", "_").upper(), item, '')
        enumItems.append(enumItem)
    return enumItems


def ext_category_lister(self, context, category):
    items = getattr(context.scene, category)
    categories = []
    for item in items:
        categoryEnum = item.category.replace(" ", "").upper()
        if categoryEnum not in sxglobals.preset_lookup:
            sxglobals.preset_lookup[categoryEnum] = item.category
        enumItem = (categoryEnum, item.category, '')
        categories.append(enumItem)
    enumItems = list(set(categories))
    return enumItems


def load_category(self, context):
    objs = mesh_selection_validator(self, context)
    if objs:
        active = context.view_layer.objects.active
        category_data = sxglobals.category_dict[sxglobals.preset_lookup[objs[0].sx2.category]]
        sxglobals.magic_in_progress = True

        for obj in objs:
            obj.select_set(False)

        for obj in objs:
            obj.select_set(True)
            context.view_layer.objects.active = obj

            # Move Collider ID layer out of the way
            for layer in obj.sx2layers:
                if layer.layer_type == 'CID':
                    new_index = len(obj.sx2layers) - 1
                    layer.index = utils.insert_layer_at_index(obj, layer, new_index)

            utils.sort_stack_indices(obj)

            for i in range(len(category_data['layers'])):
                layer = utils.find_layer_by_stack_index(obj, i)
                if layer is None:
                    layer_dict = layers.add_layer([obj, ], name=category_data["layers"][i][0], layer_type=category_data["layers"][i][1])
                    layer = layer_dict[obj]
                else:
                    layer.name = category_data["layers"][i][0]
                    layer.layer_type = category_data["layers"][i][1]
                
                layer.blend_mode = category_data["layers"][i][2]
                layer.opacity = category_data["layers"][i][3]
                if category_data["layers"][i][4] == "None":
                    layer.paletted = False
                else:
                    layer.paletted = True
                    layer.palette_index = category_data["layers"][i][4]

            obj.sx2.staticvertexcolors = str(category_data['staticvertexcolors'])
            obj.sx2.metallicoverride = bool(category_data['metallic_override'])
            obj.sx2.metallic0 = category_data['palette_metallic'][0]
            obj.sx2.metallic1 = category_data['palette_metallic'][1]
            obj.sx2.metallic2 = category_data['palette_metallic'][2]
            obj.sx2.metallic3 = category_data['palette_metallic'][3]
            obj.sx2.metallic4 = category_data['palette_metallic'][4]
            obj.sx2.roughnessoverride = bool(category_data['roughness_override'])
            obj.sx2.roughness0 = category_data['palette_roughness'][0]
            obj.sx2.roughness1 = category_data['palette_roughness'][1]
            obj.sx2.roughness2 = category_data['palette_roughness'][2]
            obj.sx2.roughness3 = category_data['palette_roughness'][3]
            obj.sx2.roughness4 = category_data['palette_roughness'][4]
            if category_data['overlay_opacity'] < 1.0:
                obj.sx2.materialoverride = True
                obj.sx2.mat_overlay = category_data['overlay_opacity']
            obj.sx2.selectedlayer = 1

            utils.sort_stack_indices(obj)
            context.view_layer.objects.active = None
            obj.select_set(False)

        for obj in objs:
            obj.select_set(True)

        # export.create_reserved_uvsets(objs)

        sxglobals.magic_in_progress = False
        context.view_layer.objects.active = active


def load_ramp(self, context, ramp_dict=None):
    if ramp_dict is None:
        rampName = sxglobals.preset_lookup[bpy.context.scene.sx2.ramplist]
        ramp_dict = sxglobals.ramp_dict[rampName]

    ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp'].color_ramp
    ramp.color_mode = ramp_dict['mode']
    ramp.interpolation = ramp_dict['interpolation']
    ramp.hue_interpolation = ramp_dict['hue_interpolation']

    rampLength = len(ramp.elements)
    for i in range(rampLength-1):
        ramp.elements.remove(ramp.elements[0])

    for i, tempElement in enumerate(ramp_dict['elements']):
        if i == 0:
            ramp.elements[0].position = tempElement[0]
            ramp.elements[0].color = [tempElement[1][0], tempElement[1][1], tempElement[1][2], tempElement[1][3]]
        else:
            newElement = ramp.elements.new(tempElement[0])
            newElement.color = [tempElement[1][0], tempElement[1][1], tempElement[1][2], tempElement[1][3]]


def load_libraries(self, context):
    status1 = files.load_file('palettes')
    status2 = files.load_file('materials')
    status3 = files.load_file('gradients')
    status4 = files.load_file('categories')

    sxglobals.libraries_status = (status1 and status2 and status3 and status4)


# Fillcolor is automatically converted to grayscale on specific material layers
def update_fillcolor(self, context):
    scene = context.scene.sx2
    objs = mesh_selection_validator(self, context)
    layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

    gray_types = ['MET', 'RGH', 'OCC', 'TRN']
    if (layer.layer_type in gray_types) or (objs[0].sx2.shadingmode == 'CHANNEL'):
        color = scene.fillcolor[:]
        hsl = convert.rgb_to_hsl(color)
        hsl = [0.0, 0.0, hsl[2]]
        rgb = convert.hsl_to_rgb(hsl)
        new_color = (rgb[0], rgb[1], rgb[2], 1.0)
        if color != new_color:
            scene.fillcolor = new_color


def expand_element(self, context, element):
    if not getattr(context.scene.sx2, element):
        setattr(context.scene.sx2, element, True)

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def update_curvature_selection(self, context):
    if not sxglobals.curvature_update:
        sxglobals.curvature_update = True

        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        objs = mesh_selection_validator(self, context)
        sel_mode = context.tool_settings.mesh_select_mode[:]
        limitvalue = context.scene.sx2.curvaturelimit
        tolerance = context.scene.sx2.curvaturetolerance
        scene = context.scene.sx2
        normalize_convex = scene.normalizeconvex
        normalize_concave = scene.normalizeconcave
        scene.normalizeconvex = True
        scene.normalizeconcave = True
        context.tool_settings.mesh_select_mode = (True, False, False)

        utils.clear_component_selection(objs)

        for obj in objs:
            vert_curv_dict = generate.curvature_list(obj, objs, returndict=True)
            mesh = obj.data

            for vert in mesh.vertices:
                vert.select = abs(vert_curv_dict[vert.index] - limitvalue) < tolerance

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        context.tool_settings.mesh_select_mode = sel_mode
        scene.normalizeconvex = normalize_convex
        scene.normalizeconcave = normalize_concave
        sxglobals.curvature_update = False


def update_obj_props(self, context, prop):
    if not sxglobals.refresh_in_progress:
        sxglobals.refresh_in_progress = True

        objs = mesh_selection_validator(self, context)
        if objs:
            value = getattr(objs[0].sx2, prop)
            for obj in objs:
                if getattr(obj.sx2, prop) != value:
                    setattr(obj.sx2, prop, value)

        mat_upd_props = ['shadingmode', 'debugmode', 'backfaceculling', 'mat_specular', 'mat_anisotropic', 'mat_clearcoat']
        if prop in mat_upd_props:
            setup.update_sx2material(context)
            refresh_swatches(self, context)

        elif prop == 'mat_occlusion':
            objs[0].sx2layers['Occlusion'].opacity = objs[0].sx2.mat_occlusion

        elif prop == 'mat_overlay':
            objs[0].sx2layers['Overlay'].opacity = objs[0].sx2.mat_overlay

        elif prop == 'selectedlayer':
            update_selected_layer(self, context)

        elif prop == 'staticvertexcolors':
            for obj in objs:
                obj['staticVertexColors'] = int(objs[0].sx2.staticvertexcolors)

        elif prop == 'category':
            utils.mode_manager(objs, set_mode=True, mode_id='load_category')
            load_category(self, context)
            utils.mode_manager(objs, set_mode=False, mode_id='load_category')

        sxglobals.refresh_in_progress = False


def update_layer_props(self, context, prop):
    if (not sxglobals.magic_in_progress) and (not sxglobals.layer_update_in_progress):
        sxglobals.layer_update_in_progress = True
        objs = mesh_selection_validator(self, context)
        for obj in objs:
            if getattr(obj.sx2layers[self.name], prop) != getattr(objs[0].sx2layers[self.name], prop):
                setattr(obj.sx2layers[self.name], prop, getattr(objs[0].sx2layers[self.name], prop))

        mat_props = ['paletted', 'palette_index', 'visibility', 'blend_mode']
        if prop in mat_props:
            setup.update_sx2material(context)

        elif prop == 'opacity':
            # Pack props into vectors to bypass prop count limits
            for obj in objs:
                if obj.sx2layers:
                    color_stack = []
                    material_stack = []
                    for layer in obj.sx2layers:
                        if (layer.layer_type == 'COLOR'):
                            color_stack.append((layer.index, layer.opacity))
                        elif (layer.layer_type != 'CMP'):
                            material_stack.append((layer.index, layer.opacity))

                    color_stack.sort(key=lambda y: y[0])
                    material_stack.sort(key=lambda y: y[0])
                    opacity_list = [value[1] for value in color_stack]
                    mat_list = [value[1] for value in material_stack]

                    vectors = []
                    for i in range(0, len(opacity_list), 3):
                        vectors.append(opacity_list[i:i+3])

                    for i, opacity_vector in enumerate(vectors):
                        while len(opacity_vector) < 3:
                            opacity_vector.append(1.0)
                        opacity_vector = tuple(opacity_vector)
                        setattr(obj.sx2, 'layer_opacity_list_' + str(i), opacity_vector)

                    vectors = []
                    for i in range(0, len(mat_list), 3):
                        vectors.append(mat_list[i:i+3])

                    for i, opacity_vector in enumerate(vectors):
                        while len(opacity_vector) < 3:
                            opacity_vector.append(1.0)
                        setattr(obj.sx2, 'material_opacity_list_' + str(i), opacity_vector)

                    # Create materials if blend method change is needed
                    bg_layer = utils.find_color_layers(obj, 0)
                    if (bg_layer.opacity < 1.0) and (obj.active_material.blend_method == 'OPAQUE'):
                        setup.update_sx2material(context)
                    elif (bg_layer.opacity == 1.0) and (obj.active_material.blend_method == 'BLEND'):
                        setup.update_sx2material(context)

        sxglobals.layer_update_in_progress = False


def export_validator(self, context):
    if self.exportready:
        root_group = self.id_data
        objs = mesh_selection_validator(self, context)
        for obj in objs:
            obj.select_set(False)

        objs = utils.find_children(root_group, context.view_layer.objects)
        for obj in objs:
            obj.select_set(True)

        bpy.context.view_layer.update()
        export.validate_objects(objs)
        root_group.select_set(True)

        export.create_sxcollection()


@persistent
def load_post_handler(dummy):
    prefs = bpy.context.preferences.addons['sxtools2'].preferences

    sxglobals.layer_stack_dict.clear()
    sxglobals.sx2material_dict.clear()
    sxglobals.libraries_status = False
    sxglobals.selection_modal_status = False
    sxglobals.key_modal_status = False

    load_libraries(dummy, bpy.context)

    if bpy.data.scenes['Scene'].sx2.rampmode == '':
        bpy.data.scenes['Scene'].sx2.rampmode = 'X'

    setup.update_sx2material(bpy.context)
    # Convert legacy byte color attributes to floats
    if prefs.layerdatatype == 'FLOAT':
        export.bytes_to_floats(bpy.data.objects)

    for obj in bpy.data.objects:
        if len(obj.sx2.keys()) > 0:
            if obj.sx2.hardmode == '':
                obj.sx2.hardmode = 'SHARP'


# Update revision IDs and save in asset catalogue
@persistent
def save_pre_handler(dummy):
    for obj in bpy.data.objects:
        if (len(obj.sx2.keys()) > 0):
            obj['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))
            if ('revision' not in obj.keys()):
                obj['revision'] = 1
            else:
                revision = obj['revision']
                obj['revision'] = revision + 1


@persistent
def save_post_handler(dummy):
    catalogue_dict = {}
    revision = 1
    prefs = bpy.context.preferences.addons['sxtools2'].preferences

    objs = []
    for obj in bpy.data.objects:
        if len(obj.sx2.keys()) > 0:
            objs.append(obj)
            if obj['revision'] > revision:
                revision = obj['revision']

    cost = modifiers.calculate_triangles(objs)

    if len(prefs.cataloguepath) > 0:
        try:
            with open(prefs.cataloguepath, 'r') as input:
                catalogue_dict = json.load(input)
                input.close()

            file_path = bpy.data.filepath
            asset_path = os.path.split(prefs.cataloguepath)[0]
            # prefix = os.path.commonpath([asset_path, file_path])
            # file_rel_path = os.path.relpath(file_path, asset_path)

            for category in catalogue_dict:
                for key in catalogue_dict[category]:
                    key_path = key.replace('//', os.path.sep)
                    if os.path.samefile(file_path, os.path.join(asset_path, key_path)):
                        catalogue_dict[category][key]['revision'] = str(revision)
                        catalogue_dict[category][key]['cost'] = cost

            with open(prefs.cataloguepath, 'w') as output:
                json.dump(catalogue_dict, output, indent=4)
                output.close()

        except (ValueError, IOError) as error:
            message_box('Failed to update file revision in Asset Catalogue file.', 'SX Tools Error', 'ERROR')
            return False, None


# ------------------------------------------------------------------------
#    Settings and properties
# ------------------------------------------------------------------------
class SXTOOLS2_objectprops(bpy.types.PropertyGroup):

    layer_set: bpy.props.EnumProperty(
        name='Layer Set Index',
        description='Select active layer set',
        items=[
            ('0', '0', ''),
            ('1', '1', ''),
            ('2', '2', ''),
            ('3', '3', ''),
            ('4', '4', '')],
        default='0',
        update=lambda self, context: update_obj_props(self, context, 'layer_set'))

    selectedlayer: bpy.props.IntProperty(
        name='Selected Layer',
        min=0,
        default=0,
        update=lambda self, context: update_obj_props(self, context, 'selectedlayer'))

    shadingmode: bpy.props.EnumProperty(
        name='Shading Mode',
        description='Full: Composited shading with all channels enabled\nDebug: Display selected layer only, with alpha as black\nAlpha: Display layer alpha in grayscale',
        items=[
            ('FULL', 'Full', ''),
            ('XRAY', 'X-Ray', ''),
            ('DEBUG', 'Debug', ''),
            ('CHANNEL', 'Channel', '')],
        default='FULL',
        update=lambda self, context: update_obj_props(self, context, 'shadingmode'))

    debugmode: bpy.props.EnumProperty(
        name='Debug Isolation Mode',
        description='View R/G/B/A channels in isolation',
        items=[
            ('R', 'Red', ''),
            ('G', 'Green', ''),
            ('B', 'Blue', ''),
            ('A', 'Alpha', '')],
        default='A',
        update=lambda self, context: update_obj_props(self, context, 'debugmode'))

    palettedshading: bpy.props.FloatProperty(
        name='Paletted Shading',
        description='Enable palette in SXMaterial',
        min=0.0,
        max=1.0,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'palettedshading'))

    # Running index for unique layer names
    layercount: bpy.props.IntProperty(
        name='Layer Count',
        min=0,
        default=0,
        update=lambda self, context: update_obj_props(self, context, 'layercount'))

    huevalue: bpy.props.FloatProperty(
        name='Hue',
        description='The mode hue in the selection',
        min=0.0,
        max=1.0,
        default=0.0,
        update=lambda self, context: adjust_hsl(self, context, 0))

    saturationvalue: bpy.props.FloatProperty(
        name='Saturation',
        description='The max saturation in the selection',
        min=0.0,
        max=1.0,
        default=0.0,
        update=lambda self, context: adjust_hsl(self, context, 1))

    lightnessvalue: bpy.props.FloatProperty(
        name='Lightness',
        description='The max lightness in the selection',
        min=0.0,
        max=1.0,
        default=0.0,
        update=lambda self, context: adjust_hsl(self, context, 2))

    backfaceculling: bpy.props.BoolProperty(
        name='Backface Culling',
        description='Show backfaces of transparent object',
        default=True,
        update=lambda self, context: update_obj_props(self, context, 'backfaceculling'))

    # Modifier Module Properties ------------------

    modifiervisibility: bpy.props.BoolProperty(
        name='Modifier Visibility',
        default=True,
        update=lambda self, context: update_modifiers(self, context, 'modifiervisibility'))

    smoothangle: bpy.props.FloatProperty(
        name='Normal Smoothing Angle',
        min=0.0,
        max=180.0,
        default=180.0,
        update=lambda self, context: update_obj_props(self, context, 'smoothangle'))

    xmirror: bpy.props.BoolProperty(
        name='X-Axis',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'xmirror'))

    ymirror: bpy.props.BoolProperty(
        name='Y-Axis',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'ymirror'))

    zmirror: bpy.props.BoolProperty(
        name='Z-Axis',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'zmirror'))

    smartseparate: bpy.props.BoolProperty(
        name='Smart Separate',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'smartseparate'))

    mirrorobject: bpy.props.PointerProperty(
        name='Mirror Object',
        type=bpy.types.Object,
        update=lambda self, context: update_modifiers(self, context, 'mirrorobject'))

    hardmode: bpy.props.EnumProperty(
        name='Max Crease Mode',
        description='Mode for processing edges with maximum crease',
        items=[
            ('SMOOTH', 'Smooth', ''),
            ('SHARP', 'Sharp', '')],
        default='SHARP',
        update=lambda self, context: update_modifiers(self, context, 'hardmode'))

    subdivisionlevel: bpy.props.IntProperty(
        name='Subdivision Level',
        min=0,
        max=6,
        default=1,
        update=lambda self, context: update_modifiers(self, context, 'subdivisionlevel'))

    bevelwidth: bpy.props.FloatProperty(
        name='Bevel Width',
        min=0.0,
        max=100.0,
        default=0.05,
        update=lambda self, context: update_modifiers(self, context, 'bevelwidth'))

    bevelsegments: bpy.props.IntProperty(
        name='Bevel Segments',
        min=0,
        max=10,
        default=2,
        update=lambda self, context: update_modifiers(self, context, 'bevelsegments'))

    beveltype: bpy.props.EnumProperty(
        name='Bevel Type',
        description='Bevel offset mode',
        items=[
            ('OFFSET', 'Offset', ''),
            ('WIDTH', 'Width', ''),
            ('DEPTH', 'Depth', ''),
            ('PERCENT', 'Percent', ''),
            ('ABSOLUTE', 'Absolute', '')],
        default='WIDTH',
        update=lambda self, context: update_modifiers(self, context, 'beveltype'))

    weldthreshold: bpy.props.FloatProperty(
        name='Weld Threshold',
        min=0.0,
        max=10.0,
        default=0.0,
        precision=3,
        update=lambda self, context: update_modifiers(self, context, 'weldthreshold'))

    decimation: bpy.props.FloatProperty(
        name='Decimation',
        min=0.0,
        max=10.0,
        default=0.0,
        update=lambda self, context: update_modifiers(self, context, 'decimation'))

    tiling: bpy.props.BoolProperty(
        name='Tiling Object',
        description='Creates duplicates during occlusion rendering to fix edge values',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_pos_x: bpy.props.BoolProperty(
        name='Tile X',
        description='Select required mirror directions',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tile_pos_x'))

    tile_neg_x: bpy.props.BoolProperty(
        name='Tile -X',
        description='Select required mirror directions',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tile_neg_x'))

    tile_pos_y: bpy.props.BoolProperty(
        name='Tile Y',
        description='Select required mirror directions',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tile_pos_y'))

    tile_neg_y: bpy.props.BoolProperty(
        name='Tile -Y',
        description='Select required mirror directions',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tile_neg_y'))

    tile_pos_z: bpy.props.BoolProperty(
        name='Tile Z',
        description='Select required mirror directions',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tile_pos_z'))

    tile_neg_z: bpy.props.BoolProperty(
        name='Tile -Z',
        description='Select required mirror directions',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tile_neg_z'))

    tile_offset: bpy.props.FloatProperty(
        name='Tile Offset',
        description='Adjust offset of mirror copies',
        default=0.0,
        update=lambda self, context: update_modifiers(self, context, 'tile_offset'))

    tile_preview: bpy.props.BoolProperty(
        name='Tiling Preview',
        description='Preview tiling in viewport',
        default=False,
        update=lambda self, context: update_modifiers(self, context, 'tile_preview'))

    weightednormals: bpy.props.BoolProperty(
        name='Weighted Normals',
        description='Enables Weighted Normals modifier on the object',
        default=True,
        update=lambda self, context: update_modifiers(self, context, 'weightednormals'))


    # Export Module Properties -------------------------

    category: bpy.props.EnumProperty(
        name='Category Presets',
        description='Select object category\nRenames layers to match',
        items=lambda self, context: dict_lister(self, context, sxglobals.category_dict),
        update=lambda self, context: update_obj_props(self, context, 'category'))

    staticvertexcolors: bpy.props.EnumProperty(
        name='Vertex Color Processing Mode',
        description='Choose how to export vertex colors to a game engine\nStatic bakes all color layers to VertexColor0\nPaletted leaves overlays in alpha channels',
        items=[
            ('1', 'Static', ''),
            ('0', 'Paletted', '')],
        default='1',
        update=lambda self, context: update_obj_props(self, context, 'staticvertexcolors'))

    metallicoverride: bpy.props.BoolProperty(
        name='Palette Metallic Override',
        description='Apply metallic per palette color',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'metallicoverride'))

    metallic0: bpy.props.FloatProperty(
        name='Palette Color 0 Base Metallic',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'metallic0'))

    metallic1: bpy.props.FloatProperty(
        name='Palette Color 1 Base Metallic',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'metallic1'))

    metallic2: bpy.props.FloatProperty(
        name='Palette Color 2 Base Metallic',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'metallic2'))

    metallic3: bpy.props.FloatProperty(
        name='Palette Color 3 Base Metallic',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'metallic3'))

    metallic4: bpy.props.FloatProperty(
        name='Palette Color 4 Base Metallic',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'metallic4'))

    static_metallic: bpy.props.FloatProperty(
        name='Static Layer Base Metallic',
        description='Metallic value applied to category-defined single static color layer',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'static_metallic'))

    roughnessoverride: bpy.props.BoolProperty(
        name='Palette Roughness Override',
        description='Apply roughness per palette color',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'roughnessoverride'))

    roughness0: bpy.props.FloatProperty(
        name='Palette Color 0 Base Roughness',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'roughness0'))

    roughness1: bpy.props.FloatProperty(
        name='Palette Color 1 Base Roughness',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'roughness1'))

    roughness2: bpy.props.FloatProperty(
        name='Palette Color 2 Base Roughness',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'roughness2'))

    roughness3: bpy.props.FloatProperty(
        name='Palette Color 3 Base Roughness',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'roughness3'))

    roughness4: bpy.props.FloatProperty(
        name='Palette Color 4 Base Roughness',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'roughness4'))

    static_roughness: bpy.props.FloatProperty(
        name='Static Layer Base Roughness',
        description='Roughness value applied to category-defined single static color layer',
        min=0.0,
        max=1.0,
        precision=2,
        default=1.0,
        update=lambda self, context: update_obj_props(self, context, 'static_roughness'))

    materialoverride: bpy.props.BoolProperty(
        name='Material Property Override',
        description='Apply custom shading properties\nNOTE: Specular, Anisotropic and Clearcoat\nare only exported as custom properties',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'materialoverride'))

    mat_occlusion: bpy.props.FloatProperty(
        name='Occlusion Opacity',
        min=0.0,
        max=1.0,
        precision=2,
        default=1.0,
        update=lambda self, context: update_obj_props(self, context, 'mat_occlusion'))

    mat_overlay: bpy.props.FloatProperty(
        name='Overlay Opacity',
        min=0.0,
        max=1.0,
        precision=2,
        default=1.0,
        update=lambda self, context: update_obj_props(self, context, 'mat_overlay'))

    mat_specular: bpy.props.FloatProperty(
        name='Material Specular',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.5,
        update=lambda self, context: update_obj_props(self, context, 'mat_specular'))

    mat_anisotropic: bpy.props.FloatProperty(
        name='Material Anisotropic',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'mat_anisotropic'))

    mat_clearcoat: bpy.props.FloatProperty(
        name='Material Clearcoat',
        min=0.0,
        max=1.0,
        precision=2,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'mat_clearcoat'))

    transmissionoverride: bpy.props.BoolProperty(
        name='Override Transmission',
        description='Retain manually applied Transmission / SSS',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'transmissionoverride'))

    generatelodmeshes: bpy.props.BoolProperty(
        name='Generate LOD Meshes',
        description='NOTE: Subdivision and Bevel settings affect the end result',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'generatelodmeshes'))

    lod1_bevels: bpy.props.IntProperty(
        name='LOD1 Bevel Segments',
        description='Bevel segment count for LOD1',
        min=0,
        max=10,
        default=0,
        update=lambda self, context: update_obj_props(self, context, 'lod1_bevels'))

    lod2_bevels: bpy.props.IntProperty(
        name='LOD2 Bevel Segments',
        description='Bevel segment count for LOD2',
        min=0,
        max=10,
        default=0,
        update=lambda self, context: update_obj_props(self, context, 'lod2_bevels'))

    lod1_subdiv: bpy.props.IntProperty(
        name='LOD1 Subdiv Level',
        description='Subdiv level for LOD1',
        min=0,
        max=4,
        default=1,
        update=lambda self, context: update_obj_props(self, context, 'lod1_subdiv'))

    lod2_subdiv: bpy.props.IntProperty(
        name='LOD2 Subdiv Level',
        description='Subdiv level for LOD2',
        min=0,
        max=4,
        default=0,
        update=lambda self, context: update_obj_props(self, context, 'lod2_subdiv'))

    pivotmode: bpy.props.EnumProperty(
        name='Pivot Mode',
        description='Auto pivot placement mode',
        items=[
            ('OFF', 'No Change', ''),
            ('MASS', 'Center of Mass', ''),
            ('BBOX', 'Bbox Center', ''),
            ('ROOT', 'Bbox Base', ''),
            ('ORG', 'Origin', ''),
            ('PAR', 'Parent', ''),
            ('CID', 'Collision Hull', '')],
        default='OFF',
        update=lambda self, context: update_obj_props(self, context, 'pivotmode'))

    generatehulls: bpy.props.BoolProperty(
        name='Generate Convex Hulls',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'generatehulls'))

    use_cids: bpy.props.BoolProperty(
        name='Use Collider ID',
        description='Add a Collider ID layer and paint unique face colors for each convex submesh.\nFaces with (0.0, 0.0, 0.0, 0.0) color are discarded.',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'use_cids'))

    separate_cids: bpy.props.BoolProperty(
        name='Separate Collider ID Parts',
        description='Separate loose parts even if they have the same Collider ID.\nOften useful for mirrored objects.',
        default=True,
        update=lambda self, context: update_obj_props(self, context, 'separate_cids'))

    mergefragments: bpy.props.BoolProperty(
        name='Merge fragments',
        description='Merge hull fragments by pivot. Useful for CIDs that are not continuous.\n(Meaning faces with same collider color that are separate islands.)',
        default=True,
        update=lambda self, context: update_obj_props(self, context, 'mergefragments'))

    preserveborders: bpy.props.BoolProperty(
        name='Preserve mirrored ID borders',
        description='Useful when a mirrored mesh creates convex hulls that lose concavity over mirror axes.\nEnable to split mirrored same-color ID regions to separate convex hulls.\nSeparate CID parts setting is ignored.',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'preserveborders'))

    hulltrimax: bpy.props.IntProperty(
        name='Max Hull Triangles',
        description='Max number of triangles allowed in the convex hull',
        min=4,
        max=1000,
        default=250,
        update=lambda self, context: update_obj_props(self, context, 'hulltrimax'))

    hulltrifactor: bpy.props.FloatProperty(
        name='Low Bound Factor',
        description='Sets requested low bound factor for generated hull.',
        min=0.0,
        max=1.0,
        step=0.1,
        precision=2,
        default=0.7,
        update=lambda self, context: update_obj_props(self, context, 'hulltrifactor'))

    collideroffset: bpy.props.BoolProperty(
        name='Collision Mesh Auto-Offset',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'collideroffset'))

    collideroffsetfactor: bpy.props.FloatProperty(
        name='Auto-Offset Factor',
        description='Requested shrink distance in world units, default shrink 0.01 is 1cm.\nRequested distance is capped by internally calculated max safe shrink to prevent mesh inversion.',
        min=0.0,
        max=1.0,
        step=0.01,
        precision=3,
        default=0.01,
        update=lambda self, context: update_obj_props(self, context, 'collideroffsetfactor'))

    generateemissionmeshes: bpy.props.BoolProperty(
        name='Generate Emission Meshes',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'generateemissionmeshes'))

    exportready: bpy.props.BoolProperty(
        name='Export Ready',
        description='Mark the group ready for batch export',
        default=False,
        update=export_validator)



    # SX2Material Properties ---------------------------

    layer_opacity_list_0: bpy.props.FloatVectorProperty(
        name='Layer Opacity List',
        description='Collects layer opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    layer_opacity_list_1: bpy.props.FloatVectorProperty(
        name='Layer Opacity List',
        description='Collects layer opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    layer_opacity_list_2: bpy.props.FloatVectorProperty(
        name='Layer Opacity List',
        description='Collects layer opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    layer_opacity_list_3: bpy.props.FloatVectorProperty(
        name='Layer Opacity List',
        description='Collects layer opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    layer_opacity_list_4: bpy.props.FloatVectorProperty(
        name='Layer Opacity List',
        description='Collects layer opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    layer_opacity_list_5: bpy.props.FloatVectorProperty(
        name='Layer Opacity List',
        description='Collects layer opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    layer_opacity_list_6: bpy.props.FloatVectorProperty(
        name='Layer Opacity List',
        description='Collects layer opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    material_opacity_list_0: bpy.props.FloatVectorProperty(
        name='Material Opacity List',
        description='Collects material opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)

    material_opacity_list_1: bpy.props.FloatVectorProperty(
        name='Material Opacity List',
        description='Collects material opacity values for SX2Material',
        default=(1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=3)


class SXTOOLS2_sceneprops(bpy.types.PropertyGroup):

    layerpalette1: bpy.props.FloatVectorProperty(
        name='Layer Palette 1',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    layerpalette2: bpy.props.FloatVectorProperty(
        name='Layer Palette 2',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    layerpalette3: bpy.props.FloatVectorProperty(
        name='Layer Palette 3',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    layerpalette4: bpy.props.FloatVectorProperty(
        name='Layer Palette 4',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    layerpalette5: bpy.props.FloatVectorProperty(
        name='Layer Palette 5',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    layerpalette6: bpy.props.FloatVectorProperty(
        name='Layer Palette 6',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    layerpalette7: bpy.props.FloatVectorProperty(
        name='Layer Palette 7',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    layerpalette8: bpy.props.FloatVectorProperty(
        name='Layer Palette 8',
        description='Color from the selected layer\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    toolmode: bpy.props.EnumProperty(
        name='Tool Mode',
        description='Select tool',
        items=[
            ('COL', 'Color', ''),
            ('GRD', 'Gradient', ''),
            ('PAL', 'Palette', ''),
            ('MAT', 'Material', ''),
            ('NSE', 'Noise', ''),
            ('CRV', 'Curvature', ''),
            ('OCC', 'Ambient Occlusion', ''),
            ('THK', 'Mesh Thickness', ''),
            ('DIR', 'Directional', ''),
            ('LUM', 'Luminance Remap', ''),
            ('BLR', 'Blur', ''),
            ('EMI', 'Emission', '')],
        default='COL')
        # update=lambda self, context: expand_element(self, context, 'expandfill'))

    toolopacity: bpy.props.FloatProperty(
        name='Fill Opacity',
        description='Blends fill with existing layer',
        min=0.0,
        max=1.0,
        default=1.0)

    toolblend: bpy.props.EnumProperty(
        name='Fill Blend',
        description='Fill blend mode',
        items=[
            ('ALPHA', 'Alpha', ''),
            ('ADD', 'Add', ''),
            ('MUL', 'Multiply', ''),
            ('SUB', 'Subtract', ''),
            ('OVR', 'Overlay', ''),
            ('REP', 'Replace', '')],
        default='ALPHA')

    toolinvert: bpy.props.BoolProperty(
        name='Invert',
        description='Invert Fill Value',
        default=False)

    fillpalette1: bpy.props.FloatVectorProperty(
        name='Recent Color 1',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillpalette2: bpy.props.FloatVectorProperty(
        name='Recent Color 2',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillpalette3: bpy.props.FloatVectorProperty(
        name='Recent Color 3',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillpalette4: bpy.props.FloatVectorProperty(
        name='Recent Color 4',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillpalette5: bpy.props.FloatVectorProperty(
        name='Recent Color 5',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillpalette6: bpy.props.FloatVectorProperty(
        name='Recent Color 6',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillpalette7: bpy.props.FloatVectorProperty(
        name='Recent Color 7',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillpalette8: bpy.props.FloatVectorProperty(
        name='Recent Color 8',
        description='Recent colors\nDrag and drop to Fill Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))

    fillcolor: bpy.props.FloatVectorProperty(
        name='Fill Color',
        description='This color is applied to\nthe selected objects or components',
        subtype='COLOR_GAMMA',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        update=update_fillcolor)

    noiseamplitude: bpy.props.FloatProperty(
        name='Amplitude',
        description='Random per-vertex noise amplitude',
        min=0.0,
        max=1.0,
        default=0.5)

    noiseoffset: bpy.props.FloatProperty(
        name='Offset',
        description='Random per-vertex noise offset',
        min=0.0,
        max=1.0,
        default=0.5)

    noisemono: bpy.props.BoolProperty(
        name='Monochrome',
        description='Uncheck to randomize all noise channels separately',
        default=False)

    rampmode: bpy.props.EnumProperty(
        name='Ramp Mode',
        description='X/Y/Z: Axis-aligned gradient',
        items=[
            ('X', 'X-Axis', ''),
            ('Y', 'Y-Axis', ''),
            ('Z', 'Z-Axis', '')],
        default='X')

    rampbbox: bpy.props.BoolProperty(
        name='Global Bbox',
        description='Use the combined bounding volume of\nall selected objects or components.',
        default=True)

    rampalpha: bpy.props.BoolProperty(
        name='Overwrite Alpha',
        description='Check to flood fill entire selection\nUncheck to retain current alpha mask',
        default=True)

    rampnoise: bpy.props.FloatProperty(
        name='Noise',
        description='Random per-vertex noise',
        min=0.0,
        max=1.0,
        default=0.0)

    rampmono: bpy.props.BoolProperty(
        name='Monochrome',
        description='Uncheck to randomize all noise channels separately',
        default=False)

    ramplist: bpy.props.EnumProperty(
        name='Ramp Presets',
        description='Load stored ramp presets',
        items=lambda self, context: dict_lister(self, context, sxglobals.ramp_dict),
        update=lambda self, context: load_ramp(self, context, None))

    mergemode: bpy.props.EnumProperty(
        name='Merge Mode',
        items=[
            ('UP', 'Up', ''),
            ('DOWN', 'Down', '')],
        default='UP')

    occlusionblend: bpy.props.FloatProperty(
        name='Occlusion Blend',
        description='Blend between self-occlusion and\nthe contribution of all objects in the scene',
        min=0.0,
        max=1.0,
        default=0.5)

    occlusionrays: bpy.props.IntProperty(
        name='Ray Count',
        description='Increase ray count to reduce noise',
        min=1,
        max=1000,
        step=50,
        default=250)

    occlusiondistance: bpy.props.FloatProperty(
        name='Ray Distance',
        description='How far a ray can travel without\nhitting anything before being a miss',
        min=0.0,
        max=100.0,
        default=10.0)

    occlusiongroundplane: bpy.props.BoolProperty(
        name='Ground Plane',
        description='Enable temporary ground plane for occlusion (height -0.5)',
        default=True)

    occlusiongroundheight: bpy.props.FloatProperty(
        name='Ground Plane Height',
        description='Ground Z coordinate in world space',
        default=-0.5)

    dirInclination: bpy.props.FloatProperty(
        name='Inclination',
        min=-90.0,
        max=90.0,
        default=0.0)

    dirAngle: bpy.props.FloatProperty(
        name='Angle',
        min=-360.0,
        max=360.0,
        default=0.0)

    dirCone: bpy.props.IntProperty(
        name='Spread',
        description='Softens the result with multiple samples',
        min=0,
        max=360,
        default=0)

    normalizeconvex: bpy.props.BoolProperty(
        name='Normalize Curvature',
        description='Normalize convex values',
        default=False)

    normalizeconcave: bpy.props.BoolProperty(
        name='Normalize Curvature',
        description='Normalize concave values',
        default=False)

    curvaturelimit: bpy.props.FloatProperty(
        name='Curvature Limit',
        min=0.0,
        max=1.0,
        default=0.5,
        update=update_curvature_selection)

    curvaturetolerance: bpy.props.FloatProperty(
        name='Curvature Tolerance',
        min=0.0,
        max=1.0,
        default=0.1,
        update=update_curvature_selection)

    palettecategories: bpy.props.EnumProperty(
        name='Category',
        description='Choose palette category',
        items=lambda self, context: ext_category_lister(self, context, 'sx2palettes'))

    materialcategories: bpy.props.EnumProperty(
        name='Category',
        description='Choose material category',
        items=lambda self, context: ext_category_lister(self, context, 'sx2materials'))

    mergemode: bpy.props.EnumProperty(
        name='Merge Mode',
        items=[
            ('UP', 'Up', ''),
            ('DOWN', 'Down', '')],
        default='UP')

    shift: bpy.props.BoolProperty(
        name='Shift',
        description='Keyboard input',
        default=False)

    alt: bpy.props.BoolProperty(
        name='Alt',
        description='Keyboard input',
        default=False)

    ctrl: bpy.props.BoolProperty(
        name='Ctrl',
        description='Keyboard input',
        default=False)

    enablelimit: bpy.props.BoolProperty(
        name='PBR Limit',
        description='Limit diffuse color values to PBR range',
        default=False)

    limitmode: bpy.props.EnumProperty(
        name='PBR Limit Mode',
        description='Limit diffuse values to Metallic or Non-Metallic range',
        items=[
            ('MET', 'Metallic', ''),
            ('NONMET', 'Non-Metallic', '')],
        default='MET')

    expandlayer: bpy.props.BoolProperty(
        name='Expand Layer Controls',
        default=False)
    
    expandfill: bpy.props.BoolProperty(
        name='Expand Fill',
        default=False)

    expandpal: bpy.props.BoolProperty(
        name='Expand Add Palette',
        default=False)

    newpalettename: bpy.props.StringProperty(
        name='Palette Name',
        description='New Palette Name',
        default='',
        maxlen=64)

    newpalette0: bpy.props.FloatVectorProperty(
        name='New Palette Color 0',
        description='New Palette Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_paletted_layers(self, context, 0))

    newpalette1: bpy.props.FloatVectorProperty(
        name='New Palette Color 1',
        description='New Palette Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_paletted_layers(self, context, 1))

    newpalette2: bpy.props.FloatVectorProperty(
        name='New Palette Color 2',
        description='New Palette Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_paletted_layers(self, context, 2))

    newpalette3: bpy.props.FloatVectorProperty(
        name='New Palette Color 3',
        description='New Palette Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_paletted_layers(self, context, 3))

    newpalette4: bpy.props.FloatVectorProperty(
        name='New Palette Color 4',
        description='New Palette Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_paletted_layers(self, context, 4))

    expandmat: bpy.props.BoolProperty(
        name='Expand Add Material',
        default=False)

    newmaterialname: bpy.props.StringProperty(
        name='Material Name',
        description='New Material Name',
        default='',
        maxlen=64)

    newmaterial0: bpy.props.FloatVectorProperty(
        name='Layer 7',
        description='Diffuse Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_material_layer(self, context, 0))

    newmaterial1: bpy.props.FloatVectorProperty(
        name='Metallic',
        description='Metallic Value',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_material_layer(self, context, 1))

    newmaterial2: bpy.props.FloatVectorProperty(
        name='Roughness',
        description='Roughness Value',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=lambda self, context: update_material_layer(self, context, 2))

    # Modifier Module properties -------------------------------

    creasemode: bpy.props.EnumProperty(
        name='Edge Weight Mode',
        description='Display weight tools or modifier settings',
        items=[
            ('CRS', 'Crease', ''),
            ('BEV', 'Bevel', ''),
            ('SDS', 'Modifiers', '')],
        default='CRS',
        update=lambda self, context: expand_element(self, context, 'expandcrease'))

    autocrease: bpy.props.BoolProperty(
        name='Auto Hard-Crease Bevel Edges',
        default=True)

    expandcrease: bpy.props.BoolProperty(
        name='Expand Crease',
        default=False)

    expandmirror: bpy.props.BoolProperty(
        name='Expand Mirror',
        default=False)

    expandbevel: bpy.props.BoolProperty(
        name='Expand Bevel',
        default=False)

    # Export Module properties -------------------------------

    expandexport: bpy.props.BoolProperty(
        name='Expand Export',
        default=False)

    expanddebug: bpy.props.BoolProperty(
        name='Expand Debug',
        default=False)

    expandcolliders: bpy.props.BoolProperty(
        name='Expand Colliders',
        default=False)

    expandbatchexport: bpy.props.BoolProperty(
        name='Expand Batch Export',
        default=False)

    exportmode: bpy.props.EnumProperty(
        name='Export Mode',
        description='Display magic processing, misc utils, or export settings',
        items=[
            ('MAGIC', 'Magic', ''),
            ('UTILS', 'Utilities', ''),
            ('EXPORT', 'Export', '')],
        default='MAGIC',
        update=lambda self, context: expand_element(self, context, 'expandexport'))

    exporttype: bpy.props.EnumProperty(
        name='Export Type',
        description='Export color data as vertex colors or palette atlas textures',
        items=[
            ('VERTEX', 'Vertex Colors', ''),
            ('ATLAS', 'Palette Atlas', '')],
        default='VERTEX',
        update=lambda self, context: expand_element(self, context, 'exporttype'))

    exportquality: bpy.props.EnumProperty(
        name='Export Quality',
        description='Low Detail mode uses base mesh for baking\nHigh Detail mode bakes after applying modifiers but disables decimation',
        items=[
            ('LO', 'Low Detail', ''),
            ('HI', 'High Detail', '')],
        default='LO')

    exportfolder: bpy.props.StringProperty(
        name='Export Folder',
        description='Folder to export FBX files to',
        default='',
        maxlen=1024,
        subtype='DIR_PATH')

    exportcolliders: bpy.props.BoolProperty(
        name='Export Mesh Colliders',
        default=False)


class SXTOOLS2_layerprops(bpy.types.PropertyGroup):
    # name: from PropertyGroup

    layer_set: bpy.props.IntProperty(
        name='Layer Set Index',
        min=0,
        default=0)

    index: bpy.props.IntProperty(
        name='Layer Index',
        min=0,
        max=100,
        default=100)

    layer_type: bpy.props.EnumProperty(
        name='Layer Type',
        items=[
            ('COLOR', 'Color', ''),
            ('OCC', 'Occlusion', ''),
            ('MET', 'Metallic', ''),
            ('RGH', 'Roughness', ''),
            ('TRN', 'Transmission', ''),
            ('SSS', 'Subsurface', ''),
            ('EMI', 'Emission', ''),
            ('CMP', 'Composite', ''),
            ('CID', 'Collider IDs', '')],
        default='COLOR')

    default_color: bpy.props.FloatVectorProperty(
        name='Default Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 0.0))

    visibility: bpy.props.BoolProperty(
        name='Layer Visibility',
        default=True,
        update=lambda self, context: update_layer_props(self, context, 'visibility'))

    opacity: bpy.props.FloatProperty(
        name='Layer Opacity',
        min=0.0,
        max=1.0,
        default=1.0,
        update=lambda self, context: update_layer_props(self, context, 'opacity'))

    blend_mode: bpy.props.EnumProperty(
        name='Layer Blend Mode',
        items=[
            ('ALPHA', 'Alpha', ''),
            ('ADD', 'Add', ''),
            ('MUL', 'Multiply', ''),
            ('SUB', 'Subtract', ''),
            ('OVR', 'Overlay', '')],
        default='ALPHA',
        update=lambda self, context: update_layer_props(self, context, 'blend_mode'))

    color_attribute: bpy.props.StringProperty(
        name='Vertex Color Layer',
        description='Maps a list item to a vertex color layer',
        default='')

    locked: bpy.props.BoolProperty(
        name='Lock Layer Opacity',
        default=False)

    paletted: bpy.props.BoolProperty(
        name='Paletted Layer',
        default=False,
        update=lambda self, context: update_layer_props(self, context, 'paletted'))

    palette_index: bpy.props.IntProperty(
        name='Palette Color Index',
        min=0,
        max=4,
        default=0,
        update=lambda self, context: update_layer_props(self, context, 'palette_index'))

    variants: bpy.props.BoolProperty(
        name='Variant Layer Toggle',
        default=False)

    variant_count: bpy.props.IntProperty(
        name='Variant Layer Count',
        min=0,
        default=0)

    variant_index: bpy.props.IntProperty(
        name='Variant Layer Index',
        min=0,
        default=0)


class SXTOOLS2_masterpalette(bpy.types.PropertyGroup):
    category: bpy.props.StringProperty(
        name='Category',
        description='Palette Category',
        default='')

    color0: bpy.props.FloatVectorProperty(
        name='Palette Color 0',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))

    color1: bpy.props.FloatVectorProperty(
        name='Palette Color 1',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))

    color2: bpy.props.FloatVectorProperty(
        name='Palette Color 2',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))

    color3: bpy.props.FloatVectorProperty(
        name='Palette Color 3',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))

    color4: bpy.props.FloatVectorProperty(
        name='Palette Color 4',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))


class SXTOOLS2_material(bpy.types.PropertyGroup):
    category: bpy.props.StringProperty(
        name='Category',
        description='Material Category',
        default='')

    color0: bpy.props.FloatVectorProperty(
        name='Material Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))

    color1: bpy.props.FloatVectorProperty(
        name='Material Metallic',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))

    color2: bpy.props.FloatVectorProperty(
        name='Material Roughness',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0))


class SXTOOLS2_rampcolor(bpy.types.PropertyGroup):
    # name: from PropertyGroup

    index: bpy.props.IntProperty(
        name='Element Index',
        min=0,
        max=100,
        default=0)

    position: bpy.props.FloatProperty(
        name='Element Position',
        min=0.0,
        max=1.0,
        default=0.0)

    color: bpy.props.FloatVectorProperty(
        name='Element Color',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0))


# ------------------------------------------------------------------------
#    UI Panel and Add-On Preferences
# ------------------------------------------------------------------------
class SXTOOLS2_PT_panel(bpy.types.Panel):

    bl_idname = 'SXTOOLS2_PT_panel'
    bl_label = 'SX Tools 2'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SX Tools 2'


    def draw(self, context):
        prefs = context.preferences.addons['sxtools2'].preferences
        objs = mesh_selection_validator(self, context)
        layout = self.layout
        gray_channels = ['OCC', 'MET', 'RGH', 'TRN']
        ray_modes = ['OCC', 'THK', 'EMI']
        valid_stacks = layer_validator(self, context)

        if not sxglobals.libraries_status:
            col = layout.column()
            col.label(text='Libraries not loaded')
            col.label(text='Check Add-on Preferences')
            if len(prefs.libraryfolder) > 0:
                col.operator('sx2.loadlibraries', text='Reload Libraries')

        elif objs:
            obj = objs[0]
            mode = obj.mode
            sx2 = obj.sx2
            scene = context.scene.sx2

            if not obj.sx2layers:
                layer = None
            else:
                layer = obj.sx2layers[sx2.selectedlayer]

            # Layer Controls -----------------------------------------------
            row_shading = layout.row()
            row_shading.operator("wm.url_open", text='', icon='URL').url = 'https://secretexit.notion.site/SX-Tools-2-for-Blender-Documentation-1681c68851fb4d018d1f9ec762e5aec9'

            if layer is None:
                row_shading.label(text='No layers')

            elif not valid_stacks:
                row_shading.label(text='Mismatching layer sets')

            else:
                row_shading.prop(sx2, 'shadingmode', expand=True)

                layer_header, layer_panel = layout.panel_prop(scene, 'expandlayer')
                split_basics = layer_header.split(factor=0.3)
                split_basics.prop(layer, 'blend_mode', text='')
                split_basics.prop(layer, 'opacity', slider=True, text='Layer Opacity')
                split_basics.enabled = bool(obj.sx2.shadingmode != 'XRAY')

                if layer_panel:
                    if obj.mode == 'OBJECT':
                        hue_text = 'Layer Hue'
                        saturation_text = 'Layer Saturation'
                        lightness_text = 'Layer Lightness'
                    else:
                        hue_text = 'Selection Hue'
                        saturation_text = 'Selection Saturation'
                        lightness_text = 'Selection Lightness'

                    if (obj.sx2.shadingmode != 'CHANNEL'):
                        col_hsl = layer_panel.column(align=True)
                        row_hue = col_hsl.row(align=True)
                        row_hue.prop(sx2, 'huevalue', slider=True, text=hue_text)
                        row_sat = col_hsl.row(align=True)
                        row_sat.prop(sx2, 'saturationvalue', slider=True, text=saturation_text)
                        row_lightness = col_hsl.row(align=True)
                        row_lightness.prop(sx2, 'lightnessvalue', slider=True, text=lightness_text)

                    if ((layer == utils.find_color_layers(obj, 0)) and (layer.opacity < 1.0)) or (obj.sx2.shadingmode == 'XRAY'):
                        col_culling = layer_panel.column(align=True)
                        col_culling.prop(sx2, 'backfaceculling', toggle=True)

                    if (obj.sx2.shadingmode == 'CHANNEL'):
                        row_isolation = layer_panel.row(align=True)
                        row_isolation.prop(sx2, 'debugmode', expand=True)

                    if layer.layer_type in gray_channels:
                        row_hue.enabled = False
                        row_sat.enabled = False

                row_palette = layout.row(align=True)
                for i in range(8):
                    row_palette.prop(scene, 'layerpalette' + str(i+1), text='')

            if (layer is None) or valid_stacks:
                split_list = layout.split(factor=0.9, align=True)
                split_list.template_list('SXTOOLS2_UL_layerlist', 'sx2.layerlist', obj, 'sx2layers', sx2, 'selectedlayer', type='DEFAULT', sort_reverse=True)
                col_list = split_list.column(align=True)
                col_list.operator('sx2.add_layer', text='', icon='ADD')
                col_list.operator('sx2.del_layer', text='', icon='REMOVE')
                col_list.separator()
                col_list.operator('sx2.layer_up', text='', icon='TRIA_UP')
                col_list.operator('sx2.layer_down', text='', icon='TRIA_DOWN')
                col_list.separator()
                col_list.operator('sx2.layer_props', text='', icon='OPTIONS')

            # Layer Copy Paste Merge ---------------------------------------
            if valid_stacks:
                copy_text = 'Copy'
                clr_text = 'Clear'
                paste_text = 'Paste'
                sel_text = 'Select Mask'

                if scene.shift:
                    clr_text = 'Reset All'
                    paste_text = 'Paste Alpha'
                    sel_text = 'Select Inverse'
                elif scene.alt:
                    paste_text = 'Paste'
                elif scene.ctrl:
                    clr_text = 'Flatten'
                    if obj.sx2.shadingmode == 'CHANNEL':
                        sel_text = 'Select Value'
                    else:
                        sel_text = 'Select Color'
                    paste_text = 'Paste Into Alpha'

                col_misc = layout.column(align=True)
                row_misc1 = col_misc.row(align=True)
                row_misc1.operator('sx2.mergeup')
                row_misc1.operator('sx2.copylayer', text=copy_text)
                row_misc1.operator('sx2.clear', text=clr_text)
                row_misc2 = col_misc.row(align=True)
                row_misc2.operator('sx2.mergedown')
                row_misc2.operator('sx2.pastelayer', text=paste_text)
                row_misc2.operator('sx2.selmask', text=sel_text)

                # Fill Tools --------------------------------------------------------
                box_fill = layout.box()
                row_fill = box_fill.row()
                row_fill.prop(
                    scene, 'expandfill',
                    icon='DOWNARROW_HLT' if scene.expandfill else 'RIGHTARROW',
                    icon_only=True, emboss=False)
                row_fill.prop(scene, 'toolmode', text='')
                if scene.toolmode != 'PAL' and scene.toolmode != 'MAT':
                    if (obj.sx2.shadingmode == 'CHANNEL') and (layer.layer_type not in gray_channels):
                        channels = {'R': 'Red', 'G': 'Green', 'B': 'Blue', 'A': 'Alpha'}
                        apply_text = 'Apply to ' + channels[obj.sx2.debugmode]
                    else:
                        apply_text = 'Apply'
                    row_fill.operator('sx2.applytool', text=apply_text)

                # Color Tool --------------------------------------------------------
                if scene.toolmode == 'COL':
                    split1_fill = box_fill.split(factor=0.33)
                    if ((layer is None) or (layer.layer_type == 'COLOR') or (layer.layer_type == 'EMI') or (layer.layer_type == 'SSS')) and (obj.sx2.shadingmode != 'CHANNEL'):
                        fill_text = 'Fill Color'
                    else:
                        fill_text = 'Fill Value'
                    split1_fill.label(text=fill_text)
                    split2_fill = split1_fill.split(factor=0.8)
                    split2_fill.prop(scene, 'fillcolor', text='')
                    split2_fill.operator('sx2.layerinfo', text='', icon='INFO')

                    if scene.expandfill:
                        if (layer is None) or (layer.layer_type == 'COLOR') or (layer.layer_type == 'EMI'):
                            row_fpalette = box_fill.row(align=True)
                            for i in range(8):
                                row_fpalette.prop(scene, 'fillpalette' + str(i+1), text='')

                # Occlusion Tool ---------------------------------------------------
                elif scene.toolmode in ray_modes:
                    if scene.expandfill:
                        col_fill = box_fill.column(align=True)
                        col_fill.prop(scene, 'occlusionrays', slider=True, text='Ray Count')
                        if scene.toolmode == 'OCC':
                            col_fill.prop(scene, 'occlusionblend', slider=True, text='Local/Global Mix')
                            col_fill.prop(scene, 'occlusiondistance', slider=True, text='Ray Distance')
                            if scene.occlusiongroundplane:
                                col_fill.prop(scene, 'occlusiongroundheight', text='Ground Plane Height')
                            row_ground = col_fill.row(align=False)
                            row_ground.prop(scene, 'occlusiongroundplane', text='Ground Plane')
                            row_tiling = col_fill.row(align=False)
                            row_tiling.prop(sx2, 'tiling', text='Seamless Tiling')
                            if 'sxTiler' in obj.modifiers:
                                preview_icon_dict = {True: 'RESTRICT_VIEW_OFF', False: 'RESTRICT_VIEW_ON'}
                                row_tiling.prop(sx2, 'tile_preview', text='Preview', toggle=True, icon=preview_icon_dict[obj.sx2.tile_preview])
                            row_tiling1 = col_fill.row(align=False)
                            row_tiling1.prop(sx2, 'tile_offset', text='Tile Offset')
                            row_tiling2 = col_fill.row(align=True)
                            row_tiling2.prop(sx2, 'tile_pos_x', text='+X', toggle=True)
                            row_tiling2.prop(sx2, 'tile_pos_y', text='+Y', toggle=True)
                            row_tiling2.prop(sx2, 'tile_pos_z', text='+Z', toggle=True)
                            row_tiling3 = col_fill.row(align=True)
                            row_tiling3.prop(sx2, 'tile_neg_x', text='-X', toggle=True)
                            row_tiling3.prop(sx2, 'tile_neg_y', text='-Y', toggle=True)
                            row_tiling3.prop(sx2, 'tile_neg_z', text='-Z', toggle=True)
                            if scene.occlusionblend == 0:
                                row_ground.enabled = False
                            if not sx2.tiling:
                                row_tiling1.enabled = False
                                row_tiling2.enabled = False
                                row_tiling3.enabled = False

                # Directional Tool ---------------------------------------------------
                elif scene.toolmode == 'DIR':
                    if scene.expandfill:
                        col_fill = box_fill.column(align=True)
                        col_fill.prop(scene, 'dirInclination', slider=True, text='Inclination')
                        col_fill.prop(scene, 'dirAngle', slider=True, text='Angle')
                        col_fill.prop(scene, 'dirCone', slider=True, text='Spread')

                # Curvature Tool ---------------------------------------------------
                elif scene.toolmode == 'CRV':
                    if scene.expandfill:
                        col_fill = box_fill.column(align=True)
                        row_normalize = col_fill.row(align=True)
                        row_normalize.prop(scene, 'normalizeconvex', text='Normalize Convex', toggle=True)
                        row_normalize.prop(scene, 'normalizeconcave', text='Normalize Concave', toggle=True)
                        col_fill.prop(scene, 'toolinvert', text='Invert')
                        row_tiling = col_fill.row(align=False)
                        row_tiling.prop(sx2, 'tiling', text='Seamless Tiling')
                        if 'sxTiler' in obj.modifiers:
                            preview_icon_dict = {True: 'RESTRICT_VIEW_OFF', False: 'RESTRICT_VIEW_ON'}
                            row_tiling.prop(sx2, 'tile_preview', text='Preview', toggle=True, icon=preview_icon_dict[obj.sx2.tile_preview])
                        row_tiling2 = col_fill.row(align=True)
                        row_tiling2.prop(sx2, 'tile_pos_x', text='+X', toggle=True)
                        row_tiling2.prop(sx2, 'tile_pos_y', text='+Y', toggle=True)
                        row_tiling2.prop(sx2, 'tile_pos_z', text='+Z', toggle=True)
                        row_tiling3 = col_fill.row(align=True)
                        row_tiling3.prop(sx2, 'tile_neg_x', text='-X', toggle=True)
                        row_tiling3.prop(sx2, 'tile_neg_y', text='-Y', toggle=True)
                        row_tiling3.prop(sx2, 'tile_neg_z', text='-Z', toggle=True)

                # Noise Tool -------------------------------------------------------
                elif scene.toolmode == 'NSE':
                    if scene.expandfill:
                        col_nse = box_fill.column(align=True)
                        col_nse.prop(scene, 'noiseamplitude', slider=True)
                        col_nse.prop(scene, 'noiseoffset', slider=True)
                        col_nse.prop(scene, 'noisemono', text='Monochromatic')

                # Gradient Tool ---------------------------------------------------
                elif scene.toolmode == 'GRD':
                    split1_fill = box_fill.split(factor=0.33)
                    split1_fill.label(text='Fill Mode')
                    split1_fill.prop(scene, 'rampmode', text='')

                    if scene.expandfill:
                        row3_fill = box_fill.row(align=True)
                        row3_fill.prop(scene, 'ramplist', text='')
                        row3_fill.operator('sx2.addramp', text='', icon='ADD')
                        row3_fill.operator('sx2.delramp', text='', icon='REMOVE')
                        box_fill.template_color_ramp(bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp'], 'color_ramp', expand=True)
                        if mode == 'OBJECT':
                            box_fill.prop(scene, 'rampbbox', text='Use Combined Bounding Box')

                # Luminance Remap Tool -----------------------------------------------
                elif scene.toolmode == 'LUM':
                    if scene.expandfill:
                        row3_fill = box_fill.row(align=True)
                        row3_fill.prop(scene, 'ramplist', text='')
                        row3_fill.operator('sx2.addramp', text='', icon='ADD')
                        row3_fill.operator('sx2.delramp', text='', icon='REMOVE')
                        box_fill.template_color_ramp(bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp'], 'color_ramp', expand=True)


                # Master Palettes -----------------------------------------------
                elif scene.toolmode == 'PAL':
                    palettes = context.scene.sx2palettes
                    col_preview = box_fill.column(align=True)
                    col_preview.prop(sx2, 'palettedshading', slider=True)
                    col_preview.operator('sx2.applypalette', text='Apply Palette')
                    row_newpalette = box_fill.row(align=True)
                    row_newpalette.operator('sx2.layerstopalette', text='', icon='EYEDROPPER')
                    for i in range(5):
                        row_newpalette.prop(scene, 'newpalette'+str(i), text='')
                    row_newpalette.operator('sx2.addpalette', text='', icon='ADD')

                    if scene.expandfill:
                        row_lib = box_fill.row()
                        row_lib.prop(
                            scene, 'expandpal',
                            icon='DOWNARROW_HLT' if scene.expandpal else 'RIGHTARROW',
                            icon_only=True, emboss=False)
                        row_lib.label(text='Library')
                        if scene.expandpal:
                            category = scene.palettecategories
                            split_category = box_fill.split(factor=0.33)
                            split_category.label(text='Category:')
                            row_category = split_category.row(align=True)
                            row_category.prop(scene, 'palettecategories', text='')
                            row_category.operator('sx2.addpalettecategory', text='', icon='ADD')
                            row_category.operator('sx2.delpalettecategory', text='', icon='REMOVE')
                            for palette in palettes:
                                name = palette.name
                                if palette.category.replace(" ", "").upper() == category:
                                    row_mpalette = box_fill.row(align=True)
                                    split_mpalette = row_mpalette.split(factor=0.33)
                                    split_mpalette.label(text=name)
                                    split2_mpalette = split_mpalette.split()
                                    row2_mpalette = split2_mpalette.row(align=True)
                                    for i in range(5):
                                        row2_mpalette.prop(palette, 'color'+str(i), text='')
                                    if not scene.shift:
                                        mp_button = split2_mpalette.operator('sx2.previewpalette', text='Preview')
                                        mp_button.label = name
                                    else:
                                        mp_button = split2_mpalette.operator('sx2.delpalette', text='Delete')
                                        mp_button.label = name

                # PBR Materials -------------------------------------------------
                elif scene.toolmode == 'MAT':
                    if (len(obj.sx2layers) == 0) or (obj.sx2layers[sx2.selectedlayer].layer_type != 'COLOR'):
                        if scene.expandfill:
                            box_fill.label(text='Select a Color layer to apply PBR materials')
                    else:
                        materials = context.scene.sx2materials

                        row_newmaterial = box_fill.row(align=True)
                        for i in range(3):
                            row_newmaterial.prop(scene, 'newmaterial'+str(i), text='')
                        row_newmaterial.operator('sx2.addmaterial', text='', icon='ADD')

                        row_limit = box_fill.row(align=True)
                        row_limit.prop(scene, 'enablelimit', text='Limit to PBR range')
                        row_limit.prop(scene, 'limitmode', text='')

                        if scene.expandfill:
                            row_lib = box_fill.row()
                            row_lib.prop(
                                scene, 'expandmat',
                                icon='DOWNARROW_HLT' if scene.expandmat else 'RIGHTARROW',
                                icon_only=True, emboss=False)
                            row_lib.label(text='Library')
                            if scene.expandmat:
                                category = scene.materialcategories
                                split_category = box_fill.split(factor=0.33)
                                split_category.label(text='Category:')
                                row_category = split_category.row(align=True)
                                row_category.prop(scene, 'materialcategories', text='')
                                row_category.operator('sx2.addmaterialcategory', text='', icon='ADD')
                                row_category.operator('sx2.delmaterialcategory', text='', icon='REMOVE')
                                for material in materials:
                                    name = material.name
                                    if material.category.replace(" ", "_").upper() == category:
                                        row_mat = box_fill.row(align=True)
                                        split_mat = row_mat.split(factor=0.33)
                                        split_mat.label(text=name)
                                        split2_mat = split_mat.split()
                                        row2_mat = split2_mat.row(align=True)
                                        for i in range(3):
                                            row2_mat.prop(material, 'color'+str(i), text='')
                                        if not scene.shift:
                                            mat_button = split2_mat.operator('sx2.applymaterial', text='Apply')
                                            mat_button.label = name
                                        else:
                                            mat_button = split2_mat.operator('sx2.delmaterial', text='Delete')
                                            mat_button.label = name

                if (scene.toolmode != 'PAL') and (scene.toolmode != 'MAT') and scene.expandfill:
                    split3_fill = box_fill.split(factor=0.3)
                    split3_fill.prop(scene, 'toolblend', text='')
                    split3_fill.prop(scene, 'toolopacity', slider=True)


            # Modifiers Module ----------------------------------------------
            if prefs.enable_modifiers:
                modifiers_header, modifiers_panel = layout.panel_prop(scene, 'expandcrease')
                modifiers_header_row = modifiers_header.row()
                modifiers_header_row.prop(scene, 'creasemode', expand=True)

                if modifiers_panel:
                    if scene.creasemode != 'SDS':
                        row_sets = modifiers_panel.row(align=True)
                        for i in range(4):
                            setbutton = row_sets.operator('sx2.setgroup', text=str(25*(i+1))+'%')
                            setbutton.setmode = scene.creasemode
                            setbutton.setvalue = 0.25 * (i+1)
                        col_sel = modifiers_panel.column(align=True)
                        col_sel.prop(scene, 'curvaturelimit', slider=True, text='Curvature Selector')
                        col_sel.prop(scene, 'curvaturetolerance', slider=True, text='Curvature Tolerance')
                        col_sets = modifiers_panel.column(align=True)
                        if (bpy.context.tool_settings.mesh_select_mode[0]):
                            clear_text = 'Clear Vertex Weights'
                        else:
                            clear_text = 'Clear Edge Weights'
                        setbutton = col_sets.operator('sx2.setgroup', text=clear_text)
                        setbutton.setmode = scene.creasemode
                        setbutton.setvalue = -1.0
                    elif scene.creasemode == 'SDS':
                        row_mod1 = modifiers_panel.row()
                        row_mod1.prop(
                            scene, 'expandmirror',
                            icon='DOWNARROW_HLT' if scene.expandmirror else 'RIGHTARROW',
                            icon_only=True, emboss=False)
                        row_mod1.label(text='Mirror Modifier Settings')
                        if scene.expandmirror:
                            row_mirror = modifiers_panel.row(align=True)
                            row_mirror.label(text='Axis:')
                            row_mirror.prop(sx2, 'xmirror', text='X', toggle=True)
                            row_mirror.prop(sx2, 'ymirror', text='Y', toggle=True)
                            row_mirror.prop(sx2, 'zmirror', text='Z', toggle=True)
                            row_mirrorobj = modifiers_panel.row()
                            row_mirrorobj.prop_search(sx2, 'mirrorobject', context.scene, 'objects')

                        row_mod2 = modifiers_panel.row()
                        row_mod2.prop(
                            scene, 'expandbevel',
                            icon='DOWNARROW_HLT' if scene.expandbevel else 'RIGHTARROW',
                            icon_only=True, emboss=False)
                        row_mod2.label(text='Bevel Modifier Settings')
                        if scene.expandbevel:
                            col2_sds = modifiers_panel.column(align=True)
                            col2_sds.prop(scene, 'autocrease', text='Auto Hard-Crease Bevels')
                            split_sds = col2_sds.split()
                            split_sds.label(text='Max Crease Mode:')
                            split_sds.prop(sx2, 'hardmode', text='')
                            split2_sds = col2_sds.split()
                            split2_sds.label(text='Bevel Type:')
                            split2_sds.prop(sx2, 'beveltype', text='')
                            col2_sds.prop(sx2, 'bevelsegments', text='Bevel Segments')
                            col2_sds.prop(sx2, 'bevelwidth', text='Bevel Width')

                        col3_sds = modifiers_panel.column(align=True)
                        col3_sds.prop(sx2, 'subdivisionlevel', text='Subdivision Level')
                        col3_sds.prop(sx2, 'smoothangle', text='Normal Smoothing Angle')
                        col3_sds.prop(sx2, 'weldthreshold', text='Weld Threshold')
                        if obj.sx2.subdivisionlevel > 0:
                            col3_sds.prop(sx2, 'decimation', text='Decimation Limit Angle')
                            if obj.sx2.decimation > 0.0:
                                row_tris = col3_sds.row()
                                row_tris.label(text='Selection Tris:')
                                row_tris.label(text=modifiers.calculate_triangles(objs))
                        col3_sds.separator()
                        col3_sds.prop(sx2, 'weightednormals', text='Weighted Normals', toggle=True)
                        col4_sds = modifiers_panel.column(align=True)
                        sxmodifiers = '\t'.join(obj.modifiers.keys())
                        if 'sx' in sxmodifiers:
                            if scene.shift:
                                col4_sds.operator('sx2.removemodifiers', text='Clear Modifiers')
                            else:
                                if obj.sx2.modifiervisibility:
                                    hide_text = 'Hide Modifiers'
                                else:
                                    hide_text = 'Show Modifiers'
                                col4_sds.operator('sx2.hidemodifiers', text=hide_text)
                        else:
                            if scene.expandmirror:
                                row_mirror.enabled = False
                            if scene.expandbevel:
                                col2_sds.enabled = False
                                split_sds.enabled = False
                            col3_sds.enabled = False
                            col4_sds.operator('sx2.addmodifiers', text='Add Modifiers')


            # Export Module -------------------------------------------------
            if prefs.enable_export:
                export_header, export_panel = layout.panel_prop(scene, "expandexport")
                export_header_row = export_header.row()
                export_header_row.prop(scene, 'exportmode', expand=True)

                if export_panel:
                    if scene.exportmode == 'MAGIC':
                        col_export = export_panel.column(align=True)
                        row_cat = col_export.row(align=True)
                        row_cat.label(text='Category:')
                        row_cat.prop(sx2, 'category', text='')

                        col_export.separator()

                        met_ovr_header, met_ovr_panel = col_export.panel('met_ovr_id', default_closed=True)
                        met_ovr_header.prop(sx2, 'metallicoverride', text='Override Palette Metallic', toggle=True)
                        if met_ovr_panel:
                            col_met_ovr = met_ovr_panel.column(align=True)
                            row_met_0 = col_met_ovr.row(align=True)
                            row_met_0.label(text='Palette Color 0 Metallic')
                            row_met_0.prop(sx2, 'metallic0', text='')
                            row_met_1 = col_met_ovr.row(align=True)
                            row_met_1.label(text='Palette Color 1 Metallic')
                            row_met_1.prop(sx2, 'metallic1', text='')
                            row_met_2 = col_met_ovr.row(align=True)
                            row_met_2.label(text='Palette Color 2 Metallic')
                            row_met_2.prop(sx2, 'metallic2', text='')
                            row_met_3 = col_met_ovr.row(align=True)
                            row_met_3.label(text='Palette Color 3 Metallic')
                            row_met_3.prop(sx2, 'metallic3', text='')
                            row_met_4 = col_met_ovr.row(align=True)
                            row_met_4.label(text='Palette Color 4 Metallic')
                            row_met_4.prop(sx2, 'metallic4', text='')
                            row_static_met = met_ovr_panel.row(align=True)
                            row_static_met.label(text='Static Layer Metallic')
                            row_static_met.prop(sx2, 'static_metallic', text='')
                            col_met_ovr.enabled = sx2.metallicoverride
                            met_ovr_panel.separator()

                        rgh_ovr_header, rgh_ovr_panel = col_export.panel('rgh_ovr_id', default_closed=True)
                        rgh_ovr_header.prop(sx2, 'roughnessoverride', text='Override Palette Roughness', toggle=True)
                        if rgh_ovr_panel:
                            col_rgh_ovr = rgh_ovr_panel.column(align=True)
                            row_rgh_0 = col_rgh_ovr.row(align=True)
                            row_rgh_0.label(text='Palette Color 0 Roughness')
                            row_rgh_0.prop(sx2, 'roughness0', text='')
                            row_rgh_1 = col_rgh_ovr.row(align=True)
                            row_rgh_1.label(text='Palette Color 1 Roughness')
                            row_rgh_1.prop(sx2, 'roughness1', text='')
                            row_rgh_2 = col_rgh_ovr.row(align=True)
                            row_rgh_2.label(text='Palette Color 2 Roughness')
                            row_rgh_2.prop(sx2, 'roughness2', text='')
                            row_rgh_3 = col_rgh_ovr.row(align=True)
                            row_rgh_3.label(text='Palette Color 3 Roughness')
                            row_rgh_3.prop(sx2, 'roughness3', text='')
                            row_rgh_4 = col_rgh_ovr.row(align=True)
                            row_rgh_4.label(text='Palette Color 4 Roughness')
                            row_rgh_4.prop(sx2, 'roughness4', text='')
                            row_static_rgh = rgh_ovr_panel.row(align=True)
                            row_static_rgh.label(text='Static Layer Roughness')
                            row_static_rgh.prop(sx2, 'static_roughness', text='')
                            col_rgh_ovr.enabled = sx2.roughnessoverride
                            rgh_ovr_panel.separator()

                        mat_ovr_header, mat_ovr_panel = col_export.panel('mat_ovr_id', default_closed=True)
                        mat_ovr_header.prop(sx2, 'materialoverride', text='Override Shading Properties', toggle=True)
                        if mat_ovr_panel:
                            col_mat_ovr = mat_ovr_panel.column(align=True)
                            row_occlusion = col_mat_ovr.row(align=True)
                            row_occlusion.label(text='Occlusion Opacity:')
                            row_occlusion.prop(sx2, 'mat_occlusion', text='')
                            row_overlay = col_mat_ovr.row(align=True)
                            row_overlay.label(text='Overlay Opacity:')
                            row_overlay.prop(sx2, 'mat_overlay', text='')
                            row_specular = col_mat_ovr.row(align=True)
                            row_specular.label(text='Specular:')
                            row_specular.prop(sx2, 'mat_specular', text='')
                            row_anisotropic = col_mat_ovr.row(align=True)
                            row_anisotropic.label(text='Anisotropic:')
                            row_anisotropic.prop(sx2, 'mat_anisotropic', text='')
                            row_clearcoat = col_mat_ovr.row(align=True)
                            row_clearcoat.label(text='Clearcoat:')
                            row_clearcoat.prop(sx2, 'mat_clearcoat', text='')
                            col_mat_ovr.enabled = sx2.materialoverride
                            mat_ovr_panel.separator()
                            mat_ovr_panel.prop(sx2, 'transmissionoverride', text='Preserve Transmission / SSS', toggle=True)

                        col_export.separator()
                        row_static = col_export.row(align=True)
                        row_static.label(text='Vertex Colors:')
                        row_static.prop(sx2, 'staticvertexcolors', text='')

                        row_quality = col_export.row(align=True)
                        row_quality.label(text='Bake Detail:')
                        row_quality.prop(scene, 'exportquality', text='')

                        row_pivot = col_export.row(align=True)
                        row_pivot.label(text='Auto-Pivot:')
                        row_pivot.prop(sx2, 'pivotmode', text='')

                        col_export.separator()

                        export_mesh_header, export_mesh_panel = col_export.panel('export_mesh_id', default_closed=True)
                        export_mesh_header.label(text='Export Mesh Generation')
                        if export_mesh_panel:
                            col_export_mesh = export_mesh_panel.column(align=True)

                            row_separate = col_export_mesh.row(align=True)
                            row_separate.label(text='Smart Separate on Export:')
                            row_separate.prop(sx2, 'smartseparate', text='', toggle=False)

                            row_lod = col_export_mesh.row(align=True)
                            row_lod.label(text='Generate LOD Meshes on Export:')
                            row_lod.prop(sx2, 'generatelodmeshes', text='', toggle=False)
                            if obj.sx2.generatelodmeshes:
                                col_lods = col_export_mesh.column(align=True)
                                col_lods.prop(sx2, 'lod1_bevels', text='LOD1 Bevels')
                                col_lods.prop(sx2, 'lod1_subdiv', text='LOD1 Subdiv')
                                col_lods.prop(sx2, 'lod2_bevels', text='LOD2 Bevels')
                                col_lods.prop(sx2, 'lod2_subdiv', text='LOD2 Subdiv')

                            row_emission = col_export_mesh.row(align=True)
                            row_emission.label(text='Generate Emission Meshes on Export:')
                            row_emission.prop(sx2, 'generateemissionmeshes', text='', toggle=False)

                            row_hulls = col_export_mesh.row(align=True)
                            row_hulls.label(text='Generate Convex Hulls on Export:')
                            row_hulls.prop(sx2, 'generatehulls', text='', toggle=False)
                            col_hulls = col_export_mesh.column(align=True)
                            if sx2.generatehulls:
                                box_cids = col_hulls.box()
                                box_cids.prop(sx2, 'use_cids', text='Use Collider IDs')
                            if sx2.use_cids and sx2.generatehulls:
                                box_cids.prop(sx2, 'separate_cids', text='Separate CID parts')
                                box_cids.prop(sx2, 'mergefragments', text='Merge CID fragments')
                                box_cids.prop(sx2, 'preserveborders', text='Preserve CID borders')
                            if sx2.generatehulls:
                                col_hulls.prop(sx2, 'hulltrimax', text='Hull Triangle Limit')
                                col_hulls.prop(sx2, 'hulltrifactor', text='Hull Low Bound Factor')
                                col_hulls.prop(sx2, 'collideroffsetfactor', text='Convex Hull Shrink Distance', slider=True)
                            col_hulls.enabled = sx2.generatehulls

                            if hasattr(bpy.types, bpy.ops.object.vhacd.idname()):
                                col_export_mesh.separator()
                                col_export_mesh.prop(scene, 'exportcolliders', text='Generate Mesh Colliders (V-HACD)')
                                if scene.exportcolliders:
                                    box_colliders = col_export_mesh.box()
                                    col_collideroffset = box_colliders.column(align=True)
                                    col_collideroffset.prop(sx2, 'collideroffset', text='Shrink Collision Mesh')
                                    col_collideroffset.prop(sx2, 'collideroffsetfactor', text='Shrink Factor', slider=True)
                                    row_colliders = box_colliders.row()
                                    row_colliders.prop(
                                        scene, 'expandcolliders',
                                        icon='DOWNARROW_HLT' if scene.expandcolliders else 'RIGHTARROW',
                                        icon_only=True, emboss=False)
                                    row_colliders.label(text='V-HACD Settings')
                                    if scene.expandcolliders:
                                        col_colliders = box_colliders.column(align=True)
                                        col_colliders.prop(scene, 'removedoubles', text='Remove Doubles')
                                        col_colliders.prop(scene, 'sourcesubdivision', text='Source Subdivision')
                                        col_colliders.prop(scene, 'voxelresolution', text='Voxel Resolution', slider=True)
                                        col_colliders.prop(scene, 'maxconcavity', text='Maximum Concavity', slider=True)
                                        col_colliders.prop(scene, 'hulldownsampling', text='Convex Hull Downsampling', slider=True)
                                        col_colliders.prop(scene, 'planedownsampling', text='Plane Downsampling', slider=True)
                                        col_colliders.prop(scene, 'collalpha', text='Symmetry Clipping Bias', slider=True)
                                        col_colliders.prop(scene, 'collbeta', text='Axis Clipping Bias', slider=True)
                                        col_colliders.prop(scene, 'maxhulls', text='Max Hulls', slider=True)
                                        col_colliders.prop(scene, 'maxvertsperhull', text='Max Vertices Per Convex Hull', slider=True)
                                        col_colliders.prop(scene, 'minvolumeperhull', text='Min Volume Per Convex Hull', slider=True)

                        col_export.separator()
                        remove_generated = False
                        if ('ExportObjects' in bpy.data.collections.keys()) and (len(bpy.data.collections['ExportObjects'].objects) > 0):
                            remove_generated = True
                        if ('SXColliders' in bpy.data.collections.keys()) and (len(bpy.data.collections['SXColliders'].objects) > 0):
                            remove_generated = True
                        if remove_generated:
                            col_export.operator('sx2.removeexports', text='Remove Generated Objects')
                        col_export.separator()
                        if scene.shift:
                            col_export.operator('sx2.revertobjects', text='Revert to Control Cages')
                        else:
                            col_export.operator('sx2.macro', text='Magic Button')

                    elif scene.exportmode == 'UTILS':
                        col_utils = export_panel.column(align=False)
                        if scene.shift:
                            group_text = 'Group Under World Origin'
                            pivot_text = 'Set Pivots to Bbox Center'
                        elif scene.ctrl:
                            group_text = 'Group Selected Objects'
                            pivot_text = 'Set Pivots to Bbox Base'
                        elif scene.alt:
                            group_text = 'Group Selected Objects'
                            pivot_text = 'Set Pivots to Origin'
                        else:
                            group_text = 'Group Selected Objects'
                            pivot_text = 'Set Pivots to Center of Mass'
                        col_utils.operator('sx2.setpivots', text=pivot_text)
                        col_utils.operator('sx2.groupobjects', text=group_text)
                        col_utils.operator('sx2.zeroverts', text='Snap Vertices to Mirror Axis')
                        row_debug = export_panel.row()
                        row_debug.prop(
                            scene, 'expanddebug',
                            icon='DOWNARROW_HLT' if scene.expanddebug else 'RIGHTARROW',
                            icon_only=True, emboss=False)
                        row_debug.label(text='Debug Tools')
                        if scene.expanddebug:
                            col_debug = export_panel.column(align=True)
                            col_debug.operator('sx2.sxtosx2')
                            col_debug.operator('sx2.smart_separate', text='Debug: Smart Separate')
                            col_debug.operator('sx2.generatehulls', text='Debug: Generate Convex Hulls')
                            col_debug.operator('sx2.generateemissionmeshes', text='Debug: Generate Emission Meshes')
                            col_debug.operator('sx2.create_sxcollection', text='Debug: Update SXCollection')
                            col_debug.operator('sx2.applymodifiers', text='Debug: Apply Modifiers')
                            col_debug.operator('sx2.generatemasks', text='Debug: Generate Masks')
                            col_debug.operator('sx2.createuv0', text='Debug: Create Reserved UV Sets')
                            col_debug.operator('sx2.bytestofloats', text='Debug: Convert to Float Colors')
                            col_debug.operator('sx2.generatelods', text='Debug: Create LOD Meshes')
                            col_debug.operator('sx2.resetscene', text='Debug: Reset scene (warning!)')
                            col_debug.operator('sx2.testbutton', text='Debug: Test button')

                    elif scene.exportmode == 'EXPORT':
                        row_exporttype = export_panel.row(align=True)
                        row_exporttype.prop(scene, 'exporttype', expand=True)
                        if len(prefs.cataloguepath) > 0:
                            row_batchexport = export_panel.row()
                            row_batchexport.prop(
                                scene, 'expandbatchexport',
                                icon='DOWNARROW_HLT' if scene.expandbatchexport else 'RIGHTARROW',
                                icon_only=True, emboss=False)
                            row_batchexport.label(text='Batch Export Settings')
                            if scene.expandbatchexport:
                                col_batchexport = export_panel.column(align=True)
                                col_cataloguebuttons = export_panel.column(align=True)
                                groups = utils.find_groups()
                                if not groups:
                                    col_batchexport.label(text='No Groups Selected')
                                else:
                                    col_batchexport.label(text='Groups to Batch Process:')
                                    for group in groups:
                                        row_group = col_batchexport.row(align=True)
                                        row_group.prop(group.sx2, 'exportready', text='')
                                        row_group.label(text='  ' + group.name)

                                col_cataloguebuttons.operator('sx2.catalogue_add', text='Add to Catalogue', icon='ADD')
                                col_cataloguebuttons.operator('sx2.catalogue_remove', text='Remove from Catalogue', icon='REMOVE')
                                col_cataloguebuttons.enabled = any(group.sx2.exportready for group in groups)

                        col2_export = export_panel.column(align=True)
                        col2_export.label(text='Export Folder:')
                        col2_export.prop(scene, 'exportfolder', text='')

                        split_export = export_panel.split(factor=0.1)
                        split_export.operator('sx2.checklist', text='', icon='INFO')

                        if scene.shift:
                            exp_text = 'Export All'
                        else:
                            exp_text = 'Export Selected'
                        if scene.exporttype == 'VERTEX':
                            split_export.operator('sx2.exportfiles', text=exp_text)
                        else:
                            split_export.operator('sx2.exportatlases', text=exp_text)

                        if (mode == 'EDIT') or (len(scene.exportfolder) == 0):
                            split_export.enabled = False

        elif (len(context.view_layer.objects.selected) > 0) and (context.view_layer.objects.selected[0].type == 'EMPTY') and (context.view_layer.objects.selected[0].sx2.exportready):
            col = layout.column()
            col.label(text='Export Folder:')
            col.prop(context.scene.sx2, 'exportfolder', text='')

            split = col.split(factor=0.1)
            split.operator('sx2.checklist', text='', icon='INFO')
            split.operator('sx2.exportgroups', text='Export Selected')

        else:
            col = layout.column()
            col.label(text='Select a mesh to continue')


class SXTOOLS2_UL_layerlist(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index, flt_flag):
        scene = context.scene.sx2
        objs = mesh_selection_validator(self, context)
        hide_icon = {False: 'HIDE_ON', True: 'HIDE_OFF'}
        lock_icon = {False: 'UNLOCKED', True: 'LOCKED'}
        paletted_icon = {False: 'RESTRICT_COLOR_OFF', True: 'RESTRICT_COLOR_ON'}

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row_item = layout.row(align=True)

            # Layer list element 1: Visibility toggle
            if (objs[0].sx2.shadingmode == 'FULL') or (objs[0].sx2.shadingmode == 'XRAY'):
                row_item.prop(item, 'visibility', text='', icon=hide_icon[item.visibility])
            else:
                row_item.label(icon=hide_icon[(utils.find_layer_index_by_name(objs[0], item.name) == objs[0].sx2.selectedlayer)])

            # Layer list element 2: Lock transparency
            if (scene.toolmode == 'PAL') or (scene.toolmode == 'MAT'):
                row_item.label(text='', icon='LOCKED')
            else:
                row_item.prop(item, 'locked', text='', icon=lock_icon[item.locked])

            if ((item.layer_type == 'COLOR') or (item.layer_type == 'EMI') or (item.layer_type == 'SSS')) and (scene.toolmode == 'PAL'):
                row_split = row_item.split(factor=0.6)
                row_split.label(text='   ' + item.name + '   ')

                row_palette = row_split.row(align=True)
                row_palette.prop(item, 'paletted', text='', icon=paletted_icon[item.paletted])
                if (utils.find_layer_index_by_name(objs[0], item.name) == objs[0].sx2.selectedlayer):
                    row_palette.operator('sx2.prev_palette', text='', icon='TRIA_LEFT')
                    row_palette.prop(scene, 'newpalette'+str(item.palette_index), text='')  # text=str(item.palette_index))
                    row_palette.operator('sx2.next_palette', text='', icon='TRIA_RIGHT')
                    # row_item.prop(item, 'palette_index', text='')
                else:
                    row_palette.label(text='', icon='DOT')
                    row_palette.prop(scene, 'newpalette'+str(item.palette_index), text='')  # text=str(item.palette_index))
                    row_palette.label(text='', icon='DOT')
                    # row_item.prop(item, 'palette_index', text='')

            else:
                row_item.label(text='   ' + item.name + '   ')
                if (item.variants):
                    if (utils.find_layer_index_by_name(objs[0], item.name) == objs[0].sx2.selectedlayer):
                        row_item.operator('sx2.prev_variant', text='', icon='TRIA_LEFT')
                        row_item.operator('sx2.next_variant', text='', icon='TRIA_RIGHT')
                        row_item.operator('sx2.add_variant', text='', icon='ADD')
                        row_item.operator('sx2.del_variant', text='', icon='REMOVE')
                    else:
                        row_item.label(text='', icon='DOT')
                    # row_item.prop(item, 'variant_index', text='')

        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text='', icon=hide_icon[item.visibility])


    # Called once to draw filtering/reordering options.
    def draw_filter(self, context, layout):
        pass


    def filter_items(self, context, data, propname):
        objs = mesh_selection_validator(self, context)
        if objs:
            flt_neworder = [layer.index for layer in objs[0].sx2layers]
            flt_flags = [self.bitflag_filter_item] * len(objs[0].sx2layers)

            return flt_flags, flt_neworder


class SXTOOLS2_preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    libraryfolder: bpy.props.StringProperty(
        name='Library Folder',
        description='Folder containing SX Tools 2 data files\n(materials.json, palettes.json, gradients.json)',
        default='',
        maxlen=1024,
        subtype='DIR_PATH',
        update=load_libraries)

    enable_modifiers: bpy.props.BoolProperty(
        name='Modifiers',
        description='Enable the custom workflow modifier module',
        default=False)

    enable_export: bpy.props.BoolProperty(
        name='Export',
        description='Enable export module',
        default=False)

    exportformat: bpy.props.EnumProperty(
        name='Export Format',
        description='Select the export file format',
        items=[
            ('FBX', 'FBX', ''),
            ('GLTF', 'glTF', '')],
        default='FBX')

    exportspace: bpy.props.EnumProperty(
        name='Color Space for Exports',
        description='Color space for exported vertex colors',
        items=[
            ('SRGB', 'sRGB', ''),
            ('LIN', 'Linear', '')],
        default='LIN')

    layerdatatype: bpy.props.EnumProperty(
        name='Layer Color Data Type',
        description='Select byte or float color',
        items=[
            ('BYTE', 'Byte Color', ''),
            ('FLOAT', 'Float Color', '')],
        default='FLOAT')

    exportroughness: bpy.props.EnumProperty(
        name='Roughness Mode',
        description='Game engines may expect roughness or smoothness',
        items=[
            ('ROUGH', 'Roughness', ''),
            ('SMOOTH', 'Smoothness', '')],
        default='SMOOTH')

    removelods: bpy.props.BoolProperty(
        name='Remove Generated Meshes After Export',
        description='Remove generated meshes from the scene after exporting',
        default=True)

    lodoffset: bpy.props.FloatProperty(
        name='LOD Mesh Preview Z-Offset',
        min=0.0,
        max=10.0,
        default=1.0)

    flipsmartx: bpy.props.BoolProperty(
        name='Flip Smart X',
        description='Reverse smart naming on X-axis',
        default=False)

    flipsmarty: bpy.props.BoolProperty(
        name='Flip Smart Y',
        description='Reverse smart naming on Y-axis',
        default=False)

    cataloguepath: bpy.props.StringProperty(
        name='Catalogue File',
        description='Catalogue file for batch exporting',
        default='',
        maxlen=1024,
        subtype='FILE_PATH')

    absolutepaths: bpy.props.BoolProperty(
        name='Use Absolute Paths',
        description='Absolute file paths increase reliability of export functions',
        default=True)


    def draw(self, context):
        layout = self.layout
        layout_split_modules_1 = layout.split()
        layout_split_modules_1.label(text='Enable SX Tools Modules:')
        layout_split_modules_2 = layout_split_modules_1.split()
        layout_split_modules_2.prop(self, 'enable_modifiers', toggle=True)
        layout_split_modules_2.prop(self, 'enable_export', toggle=True)
        layout_split_datatype = layout.split()
        layout_split_datatype.label(text='Layer Color Data Type:')
        layout_split_datatype.prop(self, 'layerdatatype', text='')
        layout_split_colorspace = layout.split()
        layout_split_colorspace.label(text='Export Color Space:')
        layout_split_colorspace.prop(self, 'exportspace', text='')
        layout_split_format = layout.split()
        layout_split_format.label(text='Export File Format:')
        layout_split_format.prop(self, 'exportformat', text='')
        layout_split_roughness = layout.split()
        layout_split_roughness.label(text='Export Roughness as:')
        layout_split_roughness.prop(self, 'exportroughness', text='')
        layout_split5 = layout.split()
        layout_split5.label(text='LOD Mesh Preview Z-Offset')
        layout_split5.prop(self, 'lodoffset', text='')
        layout_split6 = layout.split()
        layout_split6.label(text='Clear Generated Meshes After Export:')
        layout_split6.prop(self, 'removelods', text='')
        layout_split7 = layout.split()
        layout_split7.label(text='Reverse Smart Mirror Naming:')
        layout_split8 = layout_split7.split()
        layout_split8.prop(self, 'flipsmartx', text='X-Axis')
        layout_split8.prop(self, 'flipsmarty', text='Y-Axis')
        layout_split9 = layout.split()
        layout_split9.label(text='Use Absolute Paths:')
        layout_split9.prop(self, 'absolutepaths', text='')
        layout_split10 = layout.split()
        layout_split10.label(text='Library Folder:')
        layout_split10.prop(self, 'libraryfolder', text='')
        layout_split11 = layout.split()
        layout_split11.label(text='Catalogue File (Optional):')
        layout_split11.prop(self, 'cataloguepath', text='')

        bpy.context.preferences.filepaths.use_relative_paths = not self.absolutepaths


class SXTOOLS2_MT_piemenu(bpy.types.Menu):
    bl_idname = 'SXTOOLS2_MT_piemenu'
    bl_label = 'SX Tools 2'


    def draw(self, context):
        objs = mesh_selection_validator(self, context)
        if objs:
            obj = objs[0]

            layout = self.layout
            sx2 = obj.sx2
            scene = context.scene.sx2
            layer = obj.sx2layers[obj.sx2.selectedlayer]

            pie = layout.menu_pie()
            fill_row = pie.row()
            fill_row.prop(scene, 'fillcolor', text='')
            fill_row.operator('sx2.applytool', text='Apply Current Fill Tool')

            # grd_col = pie.column()
            # grd_col.prop(scene, 'rampmode', text='')
            # grd_col.operator('sxtools.applyramp', text='Apply Gradient')

            layer_col = pie.column()
            layer_col.prop(layer, 'blend_mode', text='')
            layer_col.prop(layer, 'opacity', slider=True, text='Layer Opacity')

            pie.prop(sx2, 'shadingmode', text='')

            pie.operator('sx2.copylayer', text='Copy Selection')
            pie.operator('sx2.pastelayer', text='Paste Selection')
            pie.operator('sx2.clear', text='Clear Layer')

            pie.operator('sx2.selmask', text='Select Mask')


# ------------------------------------------------------------------------
#   Operators
# ------------------------------------------------------------------------

# The selectionmonitor tracks shading mode and selection changes,
# triggering a refresh of palette swatches and other UI elements
class SXTOOLS2_OT_selectionmonitor(bpy.types.Operator):
    bl_idname = 'sx2.selectionmonitor'
    bl_label = 'Selection Monitor'
    bl_description = 'Refreshes the UI on selection change'


    @classmethod
    def poll(cls, context):
        if not bpy.app.background:
            if (context.area is not None) and (context.area.type == 'VIEW_3D'):
                return context.active_object is not None
            else:
                return False
        else:
            return context.active_object is not None


    def modal(self, context, event):
        if event.type == 'TIMER_REPORT':
            return {'PASS_THROUGH'}

        if not context.area:
            print('Selection Monitor: Context Lost')
            sxglobals.selection_modal_status = False
            return {'CANCELLED'}

        if (len(sxglobals.master_palette_list) == 0) or (len(sxglobals.material_list) == 0) or (len(sxglobals.ramp_dict) == 0) or (len(sxglobals.category_dict) == 0):
            sxglobals.libraries_status = False
            load_libraries(self, context)
            return {'PASS_THROUGH'}

        objs = mesh_selection_validator(self, context)
        if (len(objs) == 0) and (context.active_object is not None) and (context.object.mode == 'EDIT'):
            objs = [obj for obj in context.objects_in_mode if obj.type == 'MESH']
            if objs:
                for obj in objs:
                    obj.select_set(True)
                context.view_layer.objects.active = objs[0]

        if objs:
            mode = objs[0].mode
            if mode != sxglobals.prev_mode:
                # print('selectionmonitor: mode change')
                sxglobals.prev_mode = mode
                sxglobals.mode = mode
                refresh_swatches(self, context)
                return {'PASS_THROUGH'}

            if (objs[0].mode == 'EDIT'):
                selection = []
                for obj in objs:
                    if obj.mode == 'EDIT':
                        obj.update_from_editmode()
                        mesh = obj.data
                        obj_selection = [None] * len(mesh.vertices)
                        mesh.vertices.foreach_get('select', obj_selection)
                        selection.extend(obj_selection)
                # print('selectionmonitor: componentselection ', selection)

                if selection != sxglobals.prev_component_selection:
                    # print('selectionmonitor: component selection changed')
                    sxglobals.prev_component_selection = selection
                    refresh_swatches(self, context)
                    return {'PASS_THROUGH'}
                else:
                    return {'PASS_THROUGH'}
            else:
                selection = context.view_layer.objects.selected.keys()
                # print('selectionmonitor: selection ', selection)

                if selection != sxglobals.prev_selection:
                    # print('selectionmonitor: object selection changed')
                    sxglobals.prev_selection = selection[:]
                    if (selection is not None) and (len(selection) > 0) and (len(context.view_layer.objects[selection[0]].sx2layers) > 0) and (layer_validator(self, context)):
                        # match selected layer on all selected objects
                        # property update propagates the selection
                        if (objs[0].sx2.selectedlayer < len(objs[0].sx2layers)):
                            objs[0].sx2.selectedlayer = objs[0].sx2.selectedlayer
                        else:
                            objs[0].sx2.selectedlayer = 0
                    return {'PASS_THROUGH'}
                else:
                    return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        # bpy.app.timers.register(lambda: 0.01 if 'PASS_THROUGH' in self.modal(context, event) else None)
        sxglobals.prev_selection = context.view_layer.objects.selected.keys()[:]
        context.window_manager.modal_handler_add(self)
        print('SX Tools: Starting selection monitor')
        return {'RUNNING_MODAL'}


class SXTOOLS2_OT_keymonitor(bpy.types.Operator):
    bl_idname = 'sx2.keymonitor'
    bl_label = 'Key Monitor'

    redraw: bpy.props.BoolProperty(default=False)
    prevShift: bpy.props.BoolProperty(default=False)
    prevAlt: bpy.props.BoolProperty(default=False)
    prevCtrl: bpy.props.BoolProperty(default=False)


    def modal(self, context, event):
        if not context.area:
            print('Key Monitor: Context Lost')
            sxglobals.key_modal_status = False
            return {'CANCELLED'}

        scene = context.scene.sx2

        if (self.prevShift != event.shift):
            self.prevShift = event.shift
            scene.shift = event.shift
            self.redraw = True

        if (self.prevAlt != event.alt):
            self.prevAlt = event.alt
            scene.alt = event.alt
            self.redraw = True

        if (self.prevCtrl != event.ctrl):
            self.prevCtrl = event.ctrl
            scene.ctrl = event.ctrl
            self.redraw = True

        if self.redraw:
            context.area.tag_redraw()
            self.redraw = False
            return {'PASS_THROUGH'}
        else:
            return {'PASS_THROUGH'}

        return {'RUNNING_MODAL'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        print('SX Tools: Starting key monitor')
        return {'RUNNING_MODAL'}


class SXTOOLS2_OT_loadlibraries(bpy.types.Operator):
    bl_idname = 'sx2.loadlibraries'
    bl_label = 'Load Libraries'
    bl_description = 'Loads SX Tools libraries of\npalettes, materials, gradients and categories'


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        load_libraries(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_layerinfo(bpy.types.Operator):
    bl_idname = 'sx2.layerinfo'
    bl_label = 'Special Layer Info'
    bl_descriptions = 'Information about special layers'


    def invoke(self, context, event):
        message_box('SPECIAL LAYERS:\n' +
                    'Gradient1, Gradient2\n' +
                    'Grayscale layers that receive color from selected palette indices\n\n' +
                    'Occlusion, Metallic, Roughness, Transmission, Emission\n' +
                    'Grayscale layers, filled with the luminance of the selected color.\n\n' +
                    'Collision IDs\n' +
                    'An RGB layer for marking convex regions of the mesh for collider generation')
        return {'FINISHED'}


class SXTOOLS2_OT_add_ramp(bpy.types.Operator):
    bl_idname = 'sx2.addramp'
    bl_label = 'Add Ramp Preset'
    bl_description = 'Add ramp preset to Gradient Library'

    rampName: bpy.props.StringProperty(name='Ramp Name')


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'rampName', text='')


    def execute(self, context):
        files.save_ramp(self.rampName)
        return {'FINISHED'}


class SXTOOLS2_OT_del_ramp(bpy.types.Operator):
    bl_idname = 'sx2.delramp'
    bl_label = 'Remove Ramp Preset'
    bl_description = 'Delete ramp preset from Gradient Library'


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        rampEnum = context.scene.sx2.ramplist[:]
        rampName = sxglobals.preset_lookup[context.scene.sx2.ramplist]
        del sxglobals.ramp_dict[rampName]
        del sxglobals.preset_lookup[rampEnum]
        files.save_file('gradients')
        return {'FINISHED'}


class SXTOOLS2_OT_addpalettecategory(bpy.types.Operator):
    bl_idname = 'sx2.addpalettecategory'
    bl_label = 'Add Palette Category'
    bl_description = 'Adds a palette category to the Palette Library'

    paletteCategoryName: bpy.props.StringProperty(name='Category Name')


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'paletteCategoryName', text='')


    def execute(self, context):
        found = False
        for i, category_dict in enumerate(sxglobals.master_palette_list):
            for category in category_dict:
                if category == self.paletteCategoryName.replace(" ", ""):
                    message_box('Palette Category Exists!')
                    found = True
                    break

        if not found:
            categoryName = self.paletteCategoryName.replace(" ", "")
            categoryEnum = categoryName.upper()
            category_dict = {categoryName: {}}

            sxglobals.master_palette_list.append(category_dict)
            files.save_file('palettes')
            files.load_file('palettes')
            context.scene.sx2.palettecategories = categoryEnum
        return {'FINISHED'}


class SXTOOLS2_OT_delpalettecategory(bpy.types.Operator):
    bl_idname = 'sx2.delpalettecategory'
    bl_label = 'Remove Palette Category'
    bl_description = 'Removes a palette category from the Palette Library'


    def invoke(self, context, event):
        categoryEnum = context.scene.sx2.palettecategories[:]
        categoryName = sxglobals.preset_lookup[context.scene.sx2.palettecategories]

        for i, category_dict in enumerate(sxglobals.master_palette_list):
            for category in category_dict:
                if category == categoryName:
                    sxglobals.master_palette_list.remove(category_dict)
                    del sxglobals.preset_lookup[categoryEnum]
                    break

        files.save_file('palettes')
        files.load_file('palettes')
        return {'FINISHED'}


class SXTOOLS2_OT_addmaterialcategory(bpy.types.Operator):
    bl_idname = 'sx2.addmaterialcategory'
    bl_label = 'Add Material Category'
    bl_description = 'Adds a material category to the Material Library'

    materialCategoryName: bpy.props.StringProperty(name='Category Name')


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'materialCategoryName', text='')


    def execute(self, context):
        found = False
        for i, category_dict in enumerate(sxglobals.material_list):
            for category in category_dict:
                if category == self.materialCategoryName.replace(" ", ""):
                    message_box('Palette Category Exists!')
                    found = True
                    break

        if not found:
            categoryName = self.materialCategoryName.replace(" ", "")
            categoryEnum = categoryName.upper()
            category_dict = {categoryName: {}}

            sxglobals.material_list.append(category_dict)
            files.save_file('materials')
            files.load_file('materials')
            context.scene.sx2.materialcategories = categoryEnum
        return {'FINISHED'}


class SXTOOLS2_OT_delmaterialcategory(bpy.types.Operator):
    bl_idname = 'sx2.delmaterialcategory'
    bl_label = 'Remove Material Category'
    bl_description = 'Removes a material category from the Material Library'


    def invoke(self, context, event):
        categoryEnum = context.scene.sx2.materialcategories[:]
        categoryName = sxglobals.preset_lookup[context.scene.sx2.materialcategories]

        for i, category_dict in enumerate(sxglobals.material_list):
            for category in category_dict:
                if category == categoryName:
                    sxglobals.material_list.remove(category_dict)
                    del sxglobals.preset_lookup[categoryEnum]
                    break

        files.save_file('materials')
        files.load_file('materials')
        return {'FINISHED'}


class SXTOOLS2_OT_layers_to_palette(bpy.types.Operator):
    bl_idname = 'sx2.layerstopalette'
    bl_label = 'Layer Colors to Palette'
    bl_description = 'Picks current layer colors to palette swatches'


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            layers.color_layers_to_swatches(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_addpalette(bpy.types.Operator):
    bl_idname = 'sx2.addpalette'
    bl_label = 'Add Palette Preset'
    bl_description = 'Add palette preset to Palette Library'

    label: bpy.props.StringProperty(name='Palette Name')


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'label', text='')


    def execute(self, context):
        scene = context.scene.sx2
        paletteName = self.label.replace(" ", "")
        if paletteName in context.scene.sx2palettes:
            message_box('Palette Name Already in Use!')
        else:
            for category_dict in sxglobals.master_palette_list:
                for i, category in enumerate(category_dict.keys()):
                    if category.casefold() == context.scene.sx2.palettecategories.casefold():
                        category_dict[category][paletteName] = [
                            [
                                scene.newpalette0[0],
                                scene.newpalette0[1],
                                scene.newpalette0[2]
                            ],
                            [
                                scene.newpalette1[0],
                                scene.newpalette1[1],
                                scene.newpalette1[2]
                            ],
                            [
                                scene.newpalette2[0],
                                scene.newpalette2[1],
                                scene.newpalette2[2]
                            ],
                            [
                                scene.newpalette3[0],
                                scene.newpalette3[1],
                                scene.newpalette3[2]
                            ],
                            [
                                scene.newpalette4[0],
                                scene.newpalette4[1],
                                scene.newpalette4[2]
                            ]]
                    break

        files.save_file('palettes')
        files.load_file('palettes')
        return {'FINISHED'}


class SXTOOLS2_OT_delpalette(bpy.types.Operator):
    bl_idname = 'sx2.delpalette'
    bl_label = 'Remove Palette Preset'
    bl_description = 'Delete palette preset from Palette Library'

    label: bpy.props.StringProperty(name='Palette Name')


    def invoke(self, context, event):
        paletteName = self.label.replace(" ", "")
        category = context.scene.sx2palettes[paletteName].category

        for i, category_dict in enumerate(sxglobals.master_palette_list):
            if category in category_dict:
                del category_dict[category][paletteName]

        files.save_file('palettes')
        files.load_file('palettes')
        return {'FINISHED'}


class SXTOOLS2_OT_addmaterial(bpy.types.Operator):
    bl_idname = 'sx2.addmaterial'
    bl_label = 'Add Material Preset'
    bl_description = 'Add palette preset to Material Library'

    materialName: bpy.props.StringProperty(name='Material Name')


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'materialName', text='')


    def execute(self, context):
        scene = context.scene.sx2
        materialName = self.materialName.replace(" ", "")
        if materialName in context.scene.sx2materials.keys():
            message_box('Material Name Already in Use!')
        else:
            for category_dict in sxglobals.material_list:
                for category in category_dict.keys():
                    if category.casefold() == context.scene.sx2.materialcategories.casefold():
                        category_dict[category][materialName] = [
                            [
                                scene.newmaterial0[0],
                                scene.newmaterial0[1],
                                scene.newmaterial0[2]
                            ],
                            [
                                scene.newmaterial1[0],
                                scene.newmaterial1[1],
                                scene.newmaterial1[2]
                            ],
                            [
                                scene.newmaterial2[0],
                                scene.newmaterial2[1],
                                scene.newmaterial2[2]
                            ]]
                    break

            files.save_file('materials')
            files.load_file('materials')
        return {'FINISHED'}


class SXTOOLS2_OT_delmaterial(bpy.types.Operator):
    bl_idname = 'sx2.delmaterial'
    bl_label = 'Remove Material Preset'
    bl_description = 'Delete material preset from Material Library'

    label: bpy.props.StringProperty(name='Material Name')


    def invoke(self, context, event):
        materialName = self.label.replace(" ", "")
        category = context.scene.sx2materials[materialName].category

        for i, category_dict in enumerate(sxglobals.material_list):
            if category in category_dict:
                del category_dict[category][materialName]

        files.save_file('materials')
        files.load_file('materials')
        return {'FINISHED'}


class SXTOOLS2_OT_applytool(bpy.types.Operator):
    bl_idname = 'sx2.applytool'
    bl_label = 'Apply Tool'
    bl_description = 'Applies the selected mode fill\nto the selected components or objects'
    bl_options = {'UNDO'}


    @classmethod
    def description(cls, context, properties):
        mode = context.scene.sx2.toolmode

        if mode == 'COL':
            modeString = 'Color tool:\nApplies the selected color to an RGBA layer\nor the selected value to a material layer\nor direct alpha values in Alpha shading mode'
        elif mode == 'GRD':
            modeString = 'Gradient tool:\nApplies the selected gradient'
        elif mode == 'PAL':
            modeString = 'Palette tool:\nPreviews and applies palette colors\nto layers with paletting enabled\n\nThe previewed palette can be blended with the layer colors when applied\nNOTE: Palette tool retains existing alpha values of all layers'
        elif mode == 'MAT':
            modeString = 'Material tool:\nApplies albedo, metallic, and roughness\nfrom the selected material\nto the selected layer and material channels\nNOTE: Never overwrites alpha! Always respects the existing color layer mask'
        elif mode == 'NSE':
            modeString = 'Noise tool:\nApplies simple vertex color or value noise'
        elif mode == 'CRV':
            modeString = 'Curvature tool:\nApplies the curvature of each vertex\nas a grayscale value,\nwith convex areas being light tones\nand concave areas as dark'
        elif mode == 'OCC':
            modeString = 'Ambient Occlusion tool:\nApplies grayscale occlusion values to vertices\nThe optional ground plane is spawned\n0.5 units below the bounding box of the object'
        elif mode == 'THK':
            modeString = 'Mesh Thickness tool:\nApplies grayscale values\naccording to the thickness of the object\nNOTE: Thinner areas are brighter!'
        elif mode == 'DIR':
            modeString = 'Directional tool:\nBrightens the object like a directional light\nwith optional spread for softer effect'
        elif mode == 'LUM':
            modeString = 'Luminance Remap tool:\nMaps any existing grayscale values on the mesh\nto a selected gradient'
        elif mode == 'BLR':
            modeString = 'Blur tool:\nSmoothens color transitions\nby blending vertex colors with their neighbors\nNOTE: Adjust fill tool opacity for greater control!'
        else:
            modeString = 'Applies the selected tool\nto the selected objects or components'

        return modeString


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            if not objs[0].sx2layers:
                layers.add_layer(objs)
                setup.update_sx2material(context)

            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            if objs[0].sx2.shadingmode != 'CHANNEL':
                fillcolor = convert.srgb_to_linear(context.scene.sx2.fillcolor)
            else:
                fillcolor = context.scene.sx2.fillcolor

            if objs[0].mode == 'EDIT':
                context.scene.sx2.rampalpha = True

            if (objs[0].sx2.shadingmode == 'CHANNEL'):
                apply_channel = objs[0].sx2.debugmode
            else:
                apply_channel = None

            if context.scene.sx2.toolmode == 'COL':
                tools.apply_tool(objs, layer, color=fillcolor, channel=apply_channel)
                tools.update_recent_colors(fillcolor)
            else:
                tools.apply_tool(objs, layer, channel=apply_channel)

            refresh_swatches(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_add_layer(bpy.types.Operator):
    bl_idname = 'sx2.add_layer'
    bl_label = 'Add Layer'
    bl_description = 'Add a new layer'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and (len(mesh_objs[0].data.color_attributes) - len([attribute for attribute in mesh_objs[0].data.color_attributes if '_var' in attribute.name])) < 14:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        utils.mode_manager(objs, set_mode=True, mode_id='add_layer')
        # for obj in objs:
        #     if not obj.sx2layers:
        #         export.create_reserved_uvsets([obj, ])
        layers.add_layer(objs)
        setup.update_sx2material(context)
        utils.mode_manager(objs, set_mode=False, mode_id='add_layer')
        refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_del_layer(bpy.types.Operator):
    bl_idname = 'sx2.del_layer'
    bl_label = 'Remove Layer'
    bl_description = 'Remove layer'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='del_layer')
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layers.del_layer(objs, layer.name)
            setup.update_sx2material(context)
            utils.mode_manager(objs, set_mode=False, mode_id='del_layer')
            if objs[0].sx2layers:
                refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_layer_up(bpy.types.Operator):
    bl_idname = 'sx2.layer_up'
    bl_label = 'Layer Up'
    bl_description = 'Move layer up'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if (layer.index < (len(sxglobals.layer_stack_dict.get(mesh_objs[0].name, [])) - 1)):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            idx = objs[0].sx2.selectedlayer
            updated = False
            for obj in objs:
                if obj.sx2layers:
                    utils.mode_manager(objs, set_mode=True, mode_id='layer_up')
                    new_index = obj.sx2layers[idx].index + 1 if obj.sx2layers[idx].index + 1 < len(obj.sx2layers) else obj.sx2layers[idx].index
                    obj.sx2layers[idx].index = utils.insert_layer_at_index(obj, obj.sx2layers[idx], new_index)
                    updated = True
                    utils.mode_manager(objs, set_mode=False, mode_id='layer_up')

            if updated:
                setup.update_sx2material(context)

        return {'FINISHED'}


class SXTOOLS2_OT_layer_down(bpy.types.Operator):
    bl_idname = 'sx2.layer_down'
    bl_label = 'Layer Down'
    bl_description = 'Move layer down'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if (layer.index > 0):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            idx = objs[0].sx2.selectedlayer
            updated = False
            for obj in objs:
                if obj.sx2layers:
                    utils.mode_manager(objs, set_mode=True, mode_id='layer_down')
                    new_index = obj.sx2layers[idx].index - 1 if obj.sx2layers[idx].index - 1 >= 0 else 0
                    obj.sx2layers[idx].index = utils.insert_layer_at_index(obj, obj.sx2layers[idx], new_index)
                    updated = True
                    utils.mode_manager(objs, set_mode=False, mode_id='layer_down')

            if updated:
                setup.update_sx2material(context)

        return {'FINISHED'}


class SXTOOLS2_OT_layer_props(bpy.types.Operator):
    bl_idname = 'sx2.layer_props'
    bl_label = 'Layer Properties'
    bl_description = 'Edit layer properties'
    bl_options = {'UNDO'}

    layer_name: bpy.props.StringProperty(
        name='Ramp Name')

    layer_type: bpy.props.EnumProperty(
        name='Layer Type',
        items=[
            ('COLOR', 'Color', ''),
            ('OCC', 'Occlusion', ''),
            ('MET', 'Metallic', ''),
            ('RGH', 'Roughness', ''),
            ('TRN', 'Transmission', ''),
            ('SSS', 'Subsurface', ''),
            ('EMI', 'Emission', ''),
            ('CMP', 'Composite', ''),
            ('CID', 'Collider IDs', '')],
        default='COLOR')

    layer_variants: bpy.props.BoolProperty(
        name='Layer Variants',
        default=False)


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if (len(objs) > 0) and (len(objs[0].sx2layers) > 0):
            idx = objs[0].sx2.selectedlayer
            self.layer_name = objs[0].sx2layers[idx].name
            self.layer_type = objs[0].sx2layers[idx].layer_type
            self.layer_variants = objs[0].sx2layers[idx].variants

            return context.window_manager.invoke_props_dialog(self)
        else:
            return {'FINISHED'}


    def draw(self, context):
        objs = mesh_selection_validator(self, context)
        idx = objs[0].sx2.selectedlayer

        alpha_mats = ['OCC', 'MET', 'RGH', 'TRN']
        layer = objs[0].sx2layers[idx]

        layout = self.layout
        row_name = layout.row()
        row_name.prop(self, 'layer_name', text='Layer Name')
        row_type = layout.row()
        row_type.prop(self, 'layer_type', text='Layer Type')
        if self.layer_type != 'COLOR':
            row_name.enabled = False
        row_variants = layout.row(align=True)
        row_variants.label(text='Layer Variants:')
        row_variants.prop(self, 'layer_variants', text='')
        row_variants.enabled = layer.layer_type not in alpha_mats

        return None


    def execute(self, context):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        data_types = {'FLOAT': 'FLOAT_COLOR', 'BYTE': 'BYTE_COLOR'}
        alpha_mats = ['OCC', 'MET', 'RGH', 'TRN']
        objs = mesh_selection_validator(self, context)

        if (objs[0].sx2layers[objs[0].sx2.selectedlayer].variants != self.layer_variants):
            objs[0].sx2layers[objs[0].sx2.selectedlayer].variants = self.layer_variants
            return {'FINISHED'}

        for obj in objs:
            layer = obj.sx2layers[obj.sx2.selectedlayer]

            for sx2layer in obj.sx2layers:
                if (self.layer_type == sx2layer.layer_type) and (self.layer_type != 'COLOR') and (layer.variants == self.layer_variants):
                    message_box('Layer already exists!')
                    return {'FINISHED'}

            # max layer count needs to account for layers not used in compositing
            cmp_exists = False
            cid_exists = False
            for sx2layer in obj.sx2layers:
                if sx2layer.layer_type == 'CMP':
                    cmp_exists = True
                if sx2layer.layer_type == 'CID':
                    cid_exists = True

            var_layers = len([attribute for attribute in obj.data.color_attributes if '_var' in attribute.name])

            layermax = 14 + cmp_exists + cid_exists + var_layers

            if (self.layer_type != 'CMP') and (self.layer_type != 'CID') and (len(obj.data.color_attributes) == layermax) and (layer.layer_type in alpha_mats) and (self.layer_type not in alpha_mats):
                message_box('Layer stack at max, delete a color layer and try again.')
                return {'FINISHED'}

            if obj.sx2layers:
                if self.layer_type == 'OCC':
                    name = 'Occlusion'
                elif self.layer_type == 'MET':
                    name = 'Metallic'
                elif self.layer_type == 'RGH':
                    name = 'Roughness'
                elif self.layer_type == 'TRN':
                    name = 'Transmission'
                elif self.layer_type == 'SSS':
                    name = 'Subsurface'
                elif self.layer_type == 'EMI':
                    name = 'Emission'
                elif self.layer_type == 'CMP':
                    name = 'Composite'
                elif self.layer_type == 'CID':
                    name = 'Collider IDs'
                else:
                    name = self.layer_name

                for sx2layer in obj.sx2layers:
                    if (name == sx2layer.name) and (layer.variants == self.layer_variants):
                        message_box(f'Layer name {name} already in use!')
                        return {'FINISHED'}

                if layer.layer_type == 'COLOR':
                    layer.name = name
                else:
                    layer.name = 'Layer ' + str(obj.sx2.layercount)

                # juggle layer colors in case of layer type change
                source_colors = layers.get_layer(obj, layer)
                source_type = layer.layer_type[:]
                empty_check = generate.color_list(obj, layer.default_color, masklayer=layer)
                if (source_type in alpha_mats) and (self.layer_type not in alpha_mats):
                    layers.clear_layers([obj, ], layer)

                layer.layer_type = self.layer_type
                if layer.layer_type in ['OCC', 'MET', 'RGH', 'TRN', 'CMP', 'CID']:
                    layer.paletted = False
                else:
                    layer.paletted = True
                layer.default_color = sxglobals.default_colors[self.layer_type]

                # Create a color attribute for alpha materials if not in data
                if ('Alpha Materials' not in obj.data.color_attributes.keys()) and (layer.layer_type in alpha_mats):
                    obj.data.color_attributes.new(name='Alpha Materials', type=data_types[prefs.layerdatatype], domain='CORNER')
                if (layer.layer_type in alpha_mats) and (layer.color_attribute != 'Alpha Materials'):
                    obj.data.attributes.remove(obj.data.attributes[layer.color_attribute])
                    layer.color_attribute = 'Alpha Materials'

                # Fill layer with default color if no pre-existing data
                if empty_check is None:
                    colors = generate.color_list(obj, layer.default_color)
                    layers.set_layer(obj, colors, layer)
                else:
                    # copy luminance to correct alpha channel
                    if layer.layer_type in alpha_mats:
                        layers.set_layer(obj, source_colors, layer)
                    elif (source_type in alpha_mats) and (layer.layer_type not in alpha_mats):
                        obj.data.color_attributes.new(name=layer.name, type=data_types[prefs.layerdatatype], domain='CORNER')
                        layer.color_attribute = layer.name
                        layers.set_layer(obj, source_colors, layer)

                utils.sort_stack_indices(obj)

        setup.update_sx2material(context)

        return {'FINISHED'}


class SXTOOLS2_OT_add_variant(bpy.types.Operator):
    bl_idname = 'sx2.add_variant'
    bl_label = 'Add Variant Layer'
    bl_description = 'Add a new variant layer'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        utils.mode_manager(objs, set_mode=True, mode_id='add_layer')
        layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
        layers.add_variant_layer(objs[0], layer)
        setup.update_sx2material(context)
        utils.mode_manager(objs, set_mode=False, mode_id='add_layer')
        refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_del_variant(bpy.types.Operator):
    bl_idname = 'sx2.del_variant'
    bl_label = 'Remove Variant Layer'
    bl_description = 'Remove a variant layer'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']
        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[objs[0].sx2.selectedlayer]
            if (layer.variant_count > 0):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='del_layer')
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            index = objs[0].sx2layers[objs[0].sx2.selectedlayer].variant_index
            layers.del_variant_layer(objs[0], layer, index)
            setup.update_sx2material(context)
            utils.mode_manager(objs, set_mode=False, mode_id='del_layer')
            refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_prev_variant(bpy.types.Operator):
    bl_idname = 'sx2.prev_variant'
    bl_label = 'Previous Variant Layer'
    bl_description = 'Select previous variant layer'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']
        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[objs[0].sx2.selectedlayer]
            if (layer.variant_count > 0) and (layer.variant_index > 0):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='prev_variant')
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layer.variant_index -= 1
            layers.set_variant_layer(layer, layer.variant_index)
            setup.update_sx2material(context)
            utils.mode_manager(objs, set_mode=False, mode_id='prev_variant')
            refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_next_variant(bpy.types.Operator):
    bl_idname = 'sx2.next_variant'
    bl_label = 'Next Variant Layer'
    bl_description = 'Select next variant layer'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']
        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[objs[0].sx2.selectedlayer]
            if (layer.variant_count > 0) and (layer.variant_index < layer.variant_count):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='next_variant')
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layer.variant_index += 1
            layers.set_variant_layer(layer, layer.variant_index)
            setup.update_sx2material(context)
            utils.mode_manager(objs, set_mode=False, mode_id='next_variant')
            refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_prev_palette(bpy.types.Operator):
    bl_idname = 'sx2.prev_palette'
    bl_label = 'Previous Palette Index'
    bl_description = 'Select previous palette index'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']
        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[objs[0].sx2.selectedlayer]
            if (layer.palette_index > 0) and (layer.palette_index <= 4):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='prev_variant')
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layer.palette_index -= 1
            # setup.update_sx2material(context)
            utils.mode_manager(objs, set_mode=False, mode_id='prev_variant')
            # refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_next_palette(bpy.types.Operator):
    bl_idname = 'sx2.next_palette'
    bl_label = 'Next Palette Index'
    bl_description = 'Select next palette index'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']
        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[objs[0].sx2.selectedlayer]
            if (layer.palette_index >= 0) and (layer.variant_index < 4):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='next_palette')
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layer.palette_index += 1
            # setup.update_sx2material(context)
            utils.mode_manager(objs, set_mode=False, mode_id='next_palette')
            # refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_mergeup(bpy.types.Operator):
    bl_idname = 'sx2.mergeup'
    bl_label = 'Merge Up'
    bl_description = 'Merges the selected color layer with the one above'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if layer.layer_type == 'CMP':
                enabled = False
            elif (layer.layer_type == 'COLOR') and (layer.index < (len(sxglobals.layer_stack_dict.get(mesh_objs[0].name, [])) - 1)):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            for obj in objs:
                utils.sort_stack_indices(obj)
            baseLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            topLayer = utils.find_layer_by_stack_index(objs[0], baseLayer.index + 1)
            layers.merge_layers(objs, topLayer, baseLayer, topLayer)
            layers.del_layer(objs, baseLayer.name)

            if len(objs[0].sx2layers) > 1:
                layer = utils.find_layer_by_stack_index(objs[0], objs[0].sx2layers[objs[0].sx2.selectedlayer].index + 1)
                idx = utils.find_layer_index_by_name(objs[0], layer.name)
            else:
                idx = 0
            objs[0].sx2.selectedlayer = idx

            setup.update_sx2material(context)
            refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_mergedown(bpy.types.Operator):
    bl_idname = 'sx2.mergedown'
    bl_label = 'Merge Down'
    bl_description = 'Merges the selected color layer with the one below'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if layer.layer_type == 'CMP':
                enabled = False
            elif layer.index > 0:
                nextLayer = utils.find_layer_by_stack_index(mesh_objs[0], layer.index - 1)
                if (nextLayer.layer_type == 'COLOR') and (layer.layer_type == 'COLOR'):
                    enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            for obj in objs:
                utils.sort_stack_indices(obj)
            topLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            baseLayer = utils.find_layer_by_stack_index(objs[0], topLayer.index - 1)
            layers.merge_layers(objs, topLayer, baseLayer, baseLayer)
            layers.del_layer(objs, topLayer.name)
            setup.update_sx2material(context)
            refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_copylayer(bpy.types.Operator):
    bl_idname = 'sx2.copylayer'
    bl_label = 'Copy'
    bl_description = 'Copy layer or selection'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        utils.mode_manager(objs, set_mode=True, mode_id='copy_layer')
        if objs:
            for obj in objs:
                colors = layers.get_layer(obj, obj.sx2layers[objs[0].sx2.selectedlayer])
                sxglobals.copy_buffer[obj.name] = generate.mask_list(obj, colors)
        utils.mode_manager(objs, set_mode=False, mode_id='copy_layer')
        return {'FINISHED'}

 
class SXTOOLS2_OT_pastelayer(bpy.types.Operator):
    bl_idname = 'sx2.pastelayer'
    bl_label = 'Paste'
    bl_description = 'Ctrl-click to paste alpha only'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            for obj in objs:
                target_layer = obj.sx2layers[objs[0].sx2.selectedlayer]

                if event.shift:
                    mode = 'mask'
                elif event.ctrl:
                    mode = 'lumtomask'
                else:
                    mode = False

                if obj.name not in sxglobals.copy_buffer:
                    message_box('Nothing to paste to ' + obj.name + '!')
                else:
                    layers.paste_layer([obj, ], target_layer, mode)
                    refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_clearlayers(bpy.types.Operator):
    bl_idname = 'sx2.clear'
    bl_label = 'Clear Layer'
    bl_description = 'Shift-click to clear all layers\non selected objects or components\nCtrl-click to flatten all color layers'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='clearlayers')
            if event.ctrl:
                sxglobals.magic_in_progress = True
                sxglobals.refresh_in_progress = True
                for obj in objs:
                    layers.add_layer([obj, ], name='Flattened', layer_type='COLOR')
                    comp_layers = utils.find_color_layers(obj)
                    if comp_layers:
                        layers.blend_layers([obj, ], comp_layers, comp_layers[0], obj.sx2layers['Flattened'])
                    else:
                        message_box('No layers to flatten')

                for obj in objs:
                    comp_layers = utils.find_color_layers(obj)
                    layer_attributes = [(layer.color_attribute, layer.name) for layer in comp_layers]
                    for color_attribute, layer_name in layer_attributes:
                        idx = utils.find_layer_index_by_name(obj, layer_name)
                        obj.sx2layers.remove(idx)
                    for color_attribute, layer_name in layer_attributes:
                        obj.data.attributes.remove(obj.data.attributes[color_attribute])

                    utils.sort_stack_indices(obj)

                sxglobals.magic_in_progress = False
                sxglobals.refresh_in_progress = False

                objs[0].sx2.selectedlayer = utils.find_layer_index_by_name(objs[0], 'Flattened')
                setup.update_sx2material(context)
            else:
                if event.shift:
                    layer = None
                else:
                    layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

                layers.clear_layers(objs, layer)

            utils.mode_manager(objs, set_mode=False, mode_id='clearlayers')
            refresh_swatches(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_selmask(bpy.types.Operator):
    bl_idname = 'sx2.selmask'
    bl_label = 'Select Layer Mask'
    bl_description = 'Click to select components with alpha\nShift-click to invert selection\nCtrl-click to select visible faces with Fill Color'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            context.view_layer.objects.active = objs[0]
            inverse = event.shift

            if event.ctrl:
                color = convert.srgb_to_linear(context.scene.sx2.fillcolor)
                tools.select_color_mask(objs, color, inverse)
            else:
                layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
                tools.select_mask(objs, layer, inverse)

        return {'FINISHED'}


class SXTOOLS2_OT_previewpalette(bpy.types.Operator):
    bl_idname = 'sx2.previewpalette'
    bl_label = 'Preview Palette'
    bl_description = 'Previews the selected palette on the selected objects'
    bl_options = {'UNDO'}

    label: bpy.props.StringProperty()


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            palette = self.label
            tools.preview_palette(objs, palette)

        return {'FINISHED'}


class SXTOOLS2_OT_applypalette(bpy.types.Operator):
    bl_idname = 'sx2.applypalette'
    bl_label = 'Apply Palette'
    bl_description = 'Applies the selected palette to selected objects'
    bl_options = {'UNDO'}

    label: bpy.props.StringProperty()


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            palette = self.label
            tools.apply_palette(objs, palette)
            for obj in objs:
                obj.sx2.palettedshading = 0.0

            if not bpy.app.background:
                refresh_swatches(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_applymaterial(bpy.types.Operator):
    bl_idname = 'sx2.applymaterial'
    bl_label = 'Apply PBR Material'
    bl_description = 'Applies the selected material to selected objects\nAlbedo color to selected layer\nMetallic and Roughness to the respective channels\n\nNOTE: Never overwrites alpha!\nAlways respects the existing color layer mask'
    bl_options = {'UNDO'}

    label: bpy.props.StringProperty()


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            material = self.label
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            tools.apply_material(objs, layer, material)
        return {'FINISHED'}


class SXTOOLS2_OT_selectup(bpy.types.Operator):
    bl_idname = 'sx2.selectup'
    bl_label = 'Select Layer Up'
    bl_description = 'Selects the layer above'
    bl_options = {'UNDO'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            for obj in objs:
                utils.sort_stack_indices(obj)

            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layer_above = utils.find_layer_by_stack_index(objs[0], layer.index + 1)
            if layer_above is not None:
                objs[0].sx2.selectedlayer = utils.find_layer_index_by_name(objs[0], layer_above.name)
        return {'FINISHED'}


class SXTOOLS2_OT_selectdown(bpy.types.Operator):
    bl_idname = 'sx2.selectdown'
    bl_label = 'Select Layer Down'
    bl_description = 'Selects the layer below'
    bl_options = {'UNDO'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            for obj in objs:
                utils.sort_stack_indices(obj)

            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layer_below = utils.find_layer_by_stack_index(objs[0], layer.index - 1)
            if layer_below is not None:
                objs[0].sx2.selectedlayer = utils.find_layer_index_by_name(objs[0], layer_below.name)
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Modifier Operators
# ------------------------------------------------------------------------
class SXTOOLS2_OT_addmodifiers(bpy.types.Operator):
    bl_idname = 'sx2.addmodifiers'
    bl_label = 'Add Modifiers'
    bl_description = 'Adds Subdivision, Edge Split and Weighted Normals modifiers\nto selected objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            modifiers.add_modifiers(objs)

            if objs[0].mode == 'OBJECT':
                bpy.ops.object.shade_smooth()

        return {'FINISHED'}


class SXTOOLS2_OT_hidemodifiers(bpy.types.Operator):
    bl_idname = 'sx2.hidemodifiers'
    bl_label = 'Hide Modifiers'
    bl_description = 'Hide and show modifiers on selected objects\nShift-click to remove modifiers'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        objs[0].sx2.modifiervisibility = not objs[0].sx2.modifiervisibility

        return {'FINISHED'}


class SXTOOLS2_OT_applymodifiers(bpy.types.Operator):
    bl_idname = 'sx2.applymodifiers'
    bl_label = 'Apply Modifiers'
    bl_description = 'Applies modifiers to the selected objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            modifiers.apply_modifiers(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_removemodifiers(bpy.types.Operator):
    bl_idname = 'sx2.removemodifiers'
    bl_label = 'Remove Modifiers'
    bl_description = 'Remove SX Tools modifiers from selected objects\nand clear their settings'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            modifiers.remove_modifiers(objs, reset=True)

            if objs[0].mode == 'OBJECT':
                bpy.ops.object.shade_flat()
        return {'FINISHED'}


class SXTOOLS2_OT_setgroup(bpy.types.Operator):
    bl_idname = 'sx2.setgroup'
    bl_label = 'Set Value'
    bl_description = 'Click to apply modifier weight to selected edges\nShift-click to add edges to selection\nAlt-click to clear selection and select verts or edges'
    bl_options = {'UNDO'}

    setmode: bpy.props.StringProperty()
    setvalue: bpy.props.FloatProperty()

    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        scene = context.scene.sx2
        if objs:
            setmode = self.setmode
            setvalue = self.setvalue
            if event.shift:
                modifiers.select_set(objs, setvalue, setmode)
            elif event.alt:
                modifiers.select_set(objs, setvalue, setmode, True)
            else:
                modifiers.assign_set(objs, setvalue, setmode)
                if (scene.autocrease) and (setmode == 'BEV') and (setvalue != -1.0):
                    mode = bpy.context.tool_settings.mesh_select_mode[:]
                    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
                    modifiers.assign_set(objs, 1.0, 'CRS')
                    bpy.context.tool_settings.mesh_select_mode = mode
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Export Operators
# ------------------------------------------------------------------------
class SXTOOLS2_OT_exportfiles(bpy.types.Operator):
    bl_idname = 'sx2.exportfiles'
    bl_label = 'Export Selected'
    bl_description = 'Saves FBX files of multi-part objects\nAll EMPTY-type groups at root based on selection are exported\nShift-click to export all SX Tools objects in the scene'


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        prefs = context.preferences.addons['sxtools2'].preferences
        selected = None

        # Batch export only groups that are marked ready
        if bpy.app.background:
            selected = export.create_sxcollection()
            groups = utils.find_groups(selected, exportready=True)
            selected = []
            for group in groups:
                selected += utils.find_children(group, recursive=True, child_type='MESH')
        # Export all, make sure objects are in groups
        elif (event is not None) and (event.shift):
            selected = export.create_sxcollection()
            for obj in selected:
                if obj.parent is None:
                    obj.hide_viewport = False
                    export.group_objects([obj, ])
            groups = utils.find_groups(selected)
        # Export selected
        else:
            selected = mesh_selection_validator(self, context)
            for obj in selected:
                if obj.parent is None:
                    obj.hide_viewport = False
                    if obj.type == 'MESH':
                        export.group_objects([obj, ])
            groups = utils.find_groups(selected)

        if groups:
            if context.scene.sx2.exportquality == 'LO':
                emission_selected = [obj for obj in selected if obj.sx2.generateemissionmeshes]
                emission_objs = export.generate_emission_meshes(emission_selected)
                export.smart_separate(selected + emission_objs)
                export.create_sxcollection()

                # for group in groups:
                #     hull_objs = utils.find_children(group, recursive=True, child_type='MESH')
                #     export.generate_hulls(hull_objs, group)

            files.export_files(groups)

            if (prefs.removelods) and (not bpy.app.background):
                export.remove_exports()
                setup.update_sx2material(context)
        return {'FINISHED'}


class SXTOOLS2_OT_exportgroups(bpy.types.Operator):
    bl_idname = 'sx2.exportgroups'
    bl_label = 'Export Group'
    bl_description = 'Saves FBX file of selected exportready group'


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        prefs = context.preferences.addons['sxtools2'].preferences
        group = context.view_layer.objects.selected[0]
        meshes = utils.find_children(group, recursive=True, child_type='MESH')
        emission_meshes = [obj for obj in meshes if obj.sx2.generateemissionmeshes]
        added_meshes = export.generate_emission_meshes(emission_meshes)
        meshes += added_meshes
        export.smart_separate(meshes)
        files.export_files([group, ])
        if prefs.removelods:
            export.remove_exports()
            setup.update_sx2material(context)
        return {'FINISHED'}


class SXTOOLS2_OT_setpivots(bpy.types.Operator):
    bl_idname = 'sx2.setpivots'
    bl_label = 'Set Pivots'
    bl_description = 'Set pivot to center of mass on selected objects\nShift-click to set pivot to center of bounding box\nCtrl-click to set pivot to base of bounding box\nAlt-click to set pivot to origin'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        if event.shift:
            pivotmode = 2
        elif event.ctrl:
            pivotmode = 3
        elif event.alt:
            pivotmode = 4
        else:
            pivotmode = 1

        objs = mesh_selection_validator(self, context)
        if objs:
            export.set_pivots(objs, pivotmode, force=True)

        return {'FINISHED'}


class SXTOOLS2_OT_removeexports(bpy.types.Operator):
    bl_idname = 'sx2.removeexports'
    bl_label = 'Remove LODs, collision meshes, and separated parts'
    bl_description = 'Deletes generated export meshes'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        export.remove_exports()
        setup.update_sx2material(context)
        return {'FINISHED'}


class SXTOOLS2_OT_createuv0(bpy.types.Operator):
    bl_idname = 'sx2.createuv0'
    bl_label = 'Create Reserved UV Sets'
    bl_description = 'Checks if any engine-reserved UV Set is missing\nand adds it at the top of the UV sets'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            export.create_reserved_uvsets(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_groupobjects(bpy.types.Operator):
    bl_idname = 'sx2.groupobjects'
    bl_label = 'Group Objects'
    bl_description = 'Groups objects under an empty.\nDefault pivot placed at the bottom center\nShift-click to place pivot in world origin'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        origin = event.shift
        if objs:
            export.group_objects(objs, origin)
        return {'FINISHED'}


class SXTOOLS2_OT_generatelods(bpy.types.Operator):
    bl_idname = 'sx2.generatelods'
    bl_label = 'Generate LOD Meshes'
    bl_description = 'Creates LOD meshes\nfrom selected objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        org_objs = []

        if objs:
            org_objs = [obj for obj in objs if '_LOD' not in obj.name]

        if org_objs:
            for obj in org_objs:
                if obj.parent is None:
                    obj.hide_viewport = False
                    if obj.type == 'MESH':
                        export.group_objects([obj, ])
            export.generate_lods(org_objs)
        return {'FINISHED'}


class SXTOOLS2_OT_revertobjects(bpy.types.Operator):
    bl_idname = 'sx2.revertobjects'
    bl_label = 'Revert to Control Cages'
    bl_description = 'Removes modifiers and clears\nlayers generated by processing'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            utils.mode_manager(objs, set_mode=True, mode_id='revertobjects')
            export.revert_objects(objs)
            objs[0].sx2.shadingmode = 'FULL'
            refresh_swatches(self, context)

            utils.mode_manager(objs, set_mode=False, mode_id='revertobjects')
        return {'FINISHED'}


class SXTOOLS2_OT_zeroverts(bpy.types.Operator):
    bl_idname = 'sx2.zeroverts'
    bl_label = 'Set Vertices to Zero'
    bl_description = 'Sets the mirror axis position of\nselected vertices to zero'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            export.zero_verts(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_bytestofloats(bpy.types.Operator):
    bl_idname = 'sx2.bytestofloats'
    bl_label = 'Color Attributes to Float Type'
    bl_description = 'Sets color attributes to Float if they have legacy Byte Color data'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            export.bytes_to_floats(objs)

        return {'FINISHED'}


class SXTOOLS2_OT_macro(bpy.types.Operator):
    bl_idname = 'sx2.macro'
    bl_label = 'Process Exports'
    bl_description = 'Calculates material channels\naccording to category.\nApplies modifiers in High Detail mode'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = [obj for obj in objs if obj.type == 'MESH']

        if mesh_objs and mesh_objs[0].sx2layers:
            enabled = True

        return enabled


    @classmethod
    def description(cls, context, properties):
        objs = mesh_selection_validator(cls, context)
        category = objs[0].sx2.category
        mode = context.scene.sx2.exportquality

        if mode == 'HI':
            modeString = 'High Detail export:\nModifiers are applied before baking\n\n'
        else:
            modeString = 'Low Detail export:\nBaking is performed on control cage (base mesh)\n\n'

        if category == 'DEFAULT':
            modeString += 'Default batch process:\nApplies palette roughness overrides'
        elif category == 'STATIC':
            modeString += 'Static batch process:\n1) Overlay bake\n2) Occlusion bake\nNOTE: Overwrites existing overlay and occlusion'
        elif category == 'PALETTED':
            modeString += 'Paletted batch process:\n1) Occlusion bake\n2) Custom roughness & metallic bake\n3) Custom overlay bake\n4) Emissive faces are smooth and non-occluded\nNOTE: Overwrites existing overlay, occlusion, transmission, metallic and roughness'
        elif category == 'VEHICLES':
            modeString += 'Vehicle batch process:\n1) Custom occlusion bake\n2) Custom roughness & metallic bake\n3) Custom overlay bake\n4) Emissive faces are smooth and non-occluded\nNOTE: Overwrites existing overlay, occlusion, transmission, metallic and roughness'
        elif category == 'BUILDINGS':
            modeString += 'Buildings batch process:\n1) Overlay bake\n2) Occlusion bake\nNOTE: Overwrites existing overlay and occlusion'
        elif category == 'TREES':
            modeString += 'Trees batch process:\n1) Overlay bake\n2) Occlusion bake\nNOTE: Overwrites existing overlay and occlusion'
        elif category == 'PARTICLES':
            modeString += 'Particle batch process:\n1) Bake static vtx colors\n2) Strip all UV channels and additional vertex color sets'
        else:
            modeString += 'Batch process:\nCalculates material channels according to category'
        return modeString


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        scene = context.scene.sx2
        objs = mesh_selection_validator(self, context)
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        if bpy.app.background:
            filtered_objs = []
            groups = utils.find_groups(objs, exportready=True)
            if groups:
                for group in groups:
                    filtered_objs += utils.find_children(group, recursive=True, child_type='MESH')

            bpy.ops.object.select_all(action='DESELECT')
            for obj in filtered_objs:
                obj.select_set(True)

            objs = filtered_objs

        if objs:
            if bpy.app.background:
                setup.create_sxtoolmaterial()
                sxglobals.export_in_progress = True

            bpy.context.view_layer.objects.active = objs[0]
            check = export.validate_objects(objs)
            if check:
                sxglobals.magic_in_progress = True
                magic.process_objects(objs)

                if sxglobals.export_in_progress:
                    sxglobals.refresh_in_progress = True

                    source_objs = [obj for obj in objs if obj.sx2.generatehulls]

                    groups = utils.find_groups(source_objs)
                    processed_objs = []
                    if groups:
                        for group in groups:
                            children = utils.find_children(group, recursive=True, child_type='MESH')
                            hull_objs = [child for child in children if child in source_objs]
                            processed_objs += hull_objs
                            export.generate_hulls(hull_objs, group)
                        
                        groupless = [obj for obj in source_objs if obj not in processed_objs]
                        if groupless:
                            export.generate_hulls(groupless)
                    else:
                        export.generate_hulls(source_objs)

                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in objs:
                        obj.select_set(True)
                    sxglobals.refresh_in_progress = False

                if hasattr(bpy.types, bpy.ops.object.vhacd.idname()) and scene.exportcolliders:
                    export.generate_mesh_colliders(objs)

                sxglobals.magic_in_progress = False

                if not bpy.app.background:
                    objs[0].sx2.shadingmode = 'FULL'
                    refresh_swatches(self, context)

                bpy.ops.object.shade_smooth()

        return {'FINISHED'}


class SXTOOLS2_OT_checklist(bpy.types.Operator):
    bl_idname = 'sx2.checklist'
    bl_label = 'Export Checklist'
    bl_description = 'Steps to check before exporting'


    def invoke(self, context, event):
        message_box('1) Choose category\n2) Paint color layers\n2) Crease edges\n3) Set object pivots\n4) Objects must be in groups\n5) Set subdivision\n6) Set base roughness and overlay strength\n7) Press Magic Button\n8) Set folder, Export Selected')
        return {'FINISHED'}


class SXTOOLS2_OT_catalogue_add(bpy.types.Operator):
    bl_idname = 'sx2.catalogue_add'
    bl_label = 'Add File to Asset Catalogue'
    bl_description = 'Add current file to the Asset Catalogue for batch exporting'
    bl_options = {'UNDO'}

    assetTags: bpy.props.StringProperty(name='Tags')

    def load_asset_data(self, catalogue_path):
        if len(catalogue_path) > 0:
            try:
                with open(catalogue_path, 'r') as input:
                    temp_dict = {}
                    temp_dict = json.load(input)
                    input.close()
                return True, temp_dict
            except ValueError:
                message_box('Invalid Asset Catalogue file.', 'SX Tools Error', 'ERROR')
                return False, None
            except IOError:
                message_box('Asset Catalogue file not found!', 'SX Tools Error', 'ERROR')
                return False, None
        else:
            message_box('Invalid catalogue path', 'SX Tools Error', 'ERROR')
            return False, None


    def save_asset_data(self, catalogue_path, data_dict):
        if len(catalogue_path) > 0:
            with open(catalogue_path, 'w') as output:
                json.dump(data_dict, output, indent=4)
                output.close()
            message_box(catalogue_path + ' saved')
        else:
            message_box('Invalid catalogue path', 'SX Tools Error', 'ERROR')


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'assetTags')
        col.label(text='Use only spaces between multiple tags')


    def execute(self, context):
        catalogue_dict = {}
        asset_dict = {}
        prefs = context.preferences.addons['sxtools2'].preferences
        objs = mesh_selection_validator(self, context)
        result, catalogue_dict = self.load_asset_data(prefs.cataloguepath)
        if not result:
            return {'FINISHED'}

        for obj in objs:
            obj['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))

        asset_category = objs[0].sx2.category.lower()
        asset_tags = self.assetTags.split(' ')
        groups = utils.find_groups(exportready=True)
        cost = modifiers.calculate_triangles(objs)

        revision = 1
        for obj in bpy.data.objects:
            if (len(obj.sx2.keys()) > 0):
                if ('revision' not in obj.keys()):
                    obj['revision'] = 1
                elif obj['revision'] > revision:
                    revision = obj['revision']

        file_path = bpy.data.filepath

        # Check if the open scene has been saved to a file
        if len(file_path) == 0:
            message_box('Current file not saved!', 'SX Tools Error', 'ERROR')
            return {'FINISHED'}

        asset_path = os.path.split(prefs.cataloguepath)[0]
        prefix = os.path.commonpath([asset_path, file_path])
        file_rel_path = os.path.relpath(file_path, asset_path)

        # Check if file is located under the specified folder
        if not os.path.samefile(asset_path, prefix):
            message_box('File not located under asset folders!', 'SX Tools Error', 'ERROR')
            return {'FINISHED'}

        # Check if the Catalogue already contains the asset category
        if asset_category not in catalogue_dict:
            catalogue_dict[asset_category] = {}

        # Save entry with a platform-independent path separator
        asset_dict['tags'] = asset_tags
        asset_dict['objects'] = [group.name for group in groups]
        asset_dict['cost'] = cost
        asset_dict['revision'] = revision

        catalogue_dict[asset_category][file_rel_path.replace(os.path.sep, '//')] = asset_dict
        self.save_asset_data(prefs.cataloguepath, catalogue_dict)
        return {'FINISHED'}


class SXTOOLS2_OT_catalogue_remove(bpy.types.Operator):
    bl_idname = 'sx2.catalogue_remove'
    bl_label = 'Remove File from Asset Catalogue'
    bl_description = 'Remove current file from the Asset Catalogue'
    bl_options = {'UNDO'}


    def load_asset_data(self, catalogue_path):
        if len(catalogue_path) > 0:
            try:
                with open(catalogue_path, 'r') as input:
                    temp_dict = {}
                    temp_dict = json.load(input)
                    input.close()
                return True, temp_dict
            except ValueError:
                message_box('Invalid Asset Catalogue file.', 'SX Tools Error', 'ERROR')
                return False, None
            except IOError:
                message_box('Asset Catalogue file not found!', 'SX Tools Error', 'ERROR')
                return False, None
        else:
            message_box('Invalid catalogue path', 'SX Tools Error', 'ERROR')
            return False, None


    def save_asset_data(self, catalogue_path, data_dict):
        if len(catalogue_path) > 0:
            with open(catalogue_path, 'w') as output:
                json.dump(data_dict, output, indent=4)
                output.close()
            message_box(catalogue_path + ' saved')
        else:
            message_box('Invalid catalogue path', 'SX Tools Error', 'ERROR')


    def invoke(self, context, event):
        prefs = context.preferences.addons['sxtools2'].preferences
        result, asset_dict = self.load_asset_data(prefs.cataloguepath)
        if not result:
            return {'FINISHED'}

        file_path = bpy.data.filepath
        if len(file_path) == 0:
            message_box('Current file not saved!', 'SX Tools Error', 'ERROR')
            return {'FINISHED'}

        asset_path = os.path.split(prefs.cataloguepath)[0]
        paths = [asset_path, file_path]
        prefix = os.path.commonpath(paths)

        file_rel_path = os.path.relpath(file_path, asset_path)

        if not os.path.samefile(asset_path, prefix):
            message_box('File not located under asset folders!', 'SX Tools Error', 'ERROR')
            return {'FINISHED'}

        for asset_category in asset_dict:
            asset_dict[asset_category].pop(file_rel_path.replace(os.path.sep, '//'), None)

        self.save_asset_data(prefs.cataloguepath, asset_dict)
        return {'FINISHED'}


class SXTOOLS2_OT_create_sxcollection(bpy.types.Operator):
    bl_idname = 'sx2.create_sxcollection'
    bl_label = 'Update SXCollection'
    bl_description = 'Links all SXTools meshes into their own collection'
    bl_options = {'UNDO'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        export.create_sxcollection()
        return {'FINISHED'}


class SXTOOLS2_OT_smart_separate(bpy.types.Operator):
    bl_idname = 'sx2.smart_separate'
    bl_label = 'Smart Separate'
    bl_description = 'Separates and logically renames mirrored mesh parts'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        sep_objs = [obj for obj in objs if obj.sx2.smartseparate]

        if sep_objs:
            for obj in sep_objs:
                if obj.parent is None:
                    obj.hide_viewport = False
                    if obj.type == 'MESH':
                        export.group_objects([obj, ])
            export.smart_separate(sep_objs)
        else:
            message_box('No objects selected with Smart Separate enabled!')

        return {'FINISHED'}


class SXTOOLS2_OT_generate_hulls(bpy.types.Operator):
    bl_idname = 'sx2.generatehulls'
    bl_label = 'Generate Hulls'
    bl_description = 'Creates convex hull colliders'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        source_objs = [obj for obj in objs if obj.sx2.generatehulls]

        if source_objs:
            if not sxglobals.refresh_in_progress:
                sxglobals.refresh_in_progress = True
                sxglobals.magic_in_progress = True

            groups = utils.find_groups(source_objs)
            processed_objs = []
            if groups:
                for group in groups:
                    children = utils.find_children(group, recursive=True, child_type='MESH')
                    hull_objs = [child for child in children if child in source_objs]
                    processed_objs += hull_objs
                    export.generate_hulls(hull_objs, group)
                
                groupless = [obj for obj in source_objs if obj not in processed_objs]
                if groupless:
                    export.generate_hulls(groupless)
            else:
                export.generate_hulls(source_objs)

            sxglobals.refresh_in_progress = False
            sxglobals.magic_in_progress = False
        else:
            message_box('No objects selected with Generate Convex Hulls enabled!')

        return {'FINISHED'}


class SXTOOLS2_OT_generate_emission_meshes(bpy.types.Operator):
    bl_idname = 'sx2.generateemissionmeshes'
    bl_label = 'Generate Emission Meshes'
    bl_description = 'Creates separate objects from faces with emission'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        emissive_objs = [obj for obj in objs if obj.sx2.generateemissionmeshes]
        if emissive_objs:
            export.generate_emission_meshes(emissive_objs)
        else:
            message_box('No objects with emissive mesh generation selected')

        return {'FINISHED'}


class SXTOOLS2_OT_generatemasks(bpy.types.Operator):
    bl_idname = 'sx2.generatemasks'
    bl_label = 'Create Palette Masks'
    bl_description = 'Bakes masks of color layers into a UV channel\nfor exporting to game engine\nif dynamic palettes are used'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            export.generate_palette_masks(objs)
            export.generate_uv_channels(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_resetscene(bpy.types.Operator):
    bl_idname = 'sx2.resetscene'
    bl_label = 'Reset Scene'
    bl_description = 'Clears all SX Tools data from objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        setup.reset_scene()
        return {'FINISHED'}


class SXTOOLS2_OT_sxtosx2(bpy.types.Operator):
    bl_idname = 'sx2.sxtosx2'
    bl_label = 'Convert SX PBR to SX2'
    bl_description = 'Converts SXTools objects to SXTools2 data format'
    bl_options = {'UNDO'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        test_objs = mesh_selection_validator(self, context)
        objs = [obj for obj in test_objs if 'sxtools' in obj]

        if objs:
            if not sxglobals.refresh_in_progress:
                sxglobals.refresh_in_progress = True
                sxglobals.magic_in_progress = True
                if not sxglobals.libraries_status:
                    load_libraries(self, context)

                utils.mode_manager(objs, set_mode=True, mode_id='sxtosx2')
                categories = list(sxglobals.category_dict.keys())

                for obj in objs:
                    context.view_layer.objects.active = obj
                    # grab modifier values
                    obj.sx2.smoothangle = obj['sxtools'].get('smoothangle', 180)
                    obj.sx2.subdivisionlevel = obj['sxtools'].get('subdivisionlevel', 0)
                    mirror_obj_value = obj['sxtools'].get('mirrorobject', None)
                    if type(mirror_obj_value) is None or type(mirror_obj_value) is bpy.types.Object:
                        obj.sx2.mirrorobject = mirror_obj_value
                    elif hasattr(mirror_obj_value, 'to_dict'):
                        obj.sx2.mirrorobject = None
                    obj.sx2.decimation = obj['sxtools'].get('decimation', 0)
                    obj.sx2.weldthreshold = obj['sxtools'].get('weldthreshold', 0)
                    obj.sx2.smartseparate = obj['sxtools'].get('smartseparate', False)
                    pivot_dict = {0: 'OFF', 1: 'MASS', 2: 'BBOX', 3: 'ROOT', 4: 'ORG', 5: 'PAR', 6: 'CID'}
                    obj.sx2.pivotmode = pivot_dict[obj['sxtools'].get('pivotmode', 0)]
                    obj.sx2.generatelodmeshes = obj['sxtools'].get('lodmeshes', False)
                    obj.sx2.xmirror = obj['sxtools'].get('xmirror', False)
                    obj.sx2.ymirror = obj['sxtools'].get('ymirror', False)
                    obj.sx2.zmirror = obj['sxtools'].get('zmirror', False)
                    obj.sx2.bevelwidth = obj['sxtools'].get('bevelwidth', 0.05)
                    obj.sx2.bevelsegments = obj['sxtools'].get('bevelsegments', 2)
                    bevel_dict = {0: 'OFFSET', 1: 'WIDTH', 2: 'DEPTH', 3: 'PERCENT', 4: 'ABSOLUTE'}
                    obj.sx2.beveltype = bevel_dict[obj['sxtools'].get('beveltype', 1)]
                    hardmode_dict = {0: 'SHARP', 1: 'SMOOTH'}
                    obj.sx2.hardmode = hardmode_dict[obj['sxtools'].get('hardmode', 0)]
                    obj.sx2.staticvertexcolors = str(obj['sxtools'].get('staticvertexcolors', 1))
                    if obj.parent is not None:
                        obj.parent.sx2.exportready = obj.parent['sxtools'].get('exportready', False)

                    count = len(obj.data.color_attributes)
                    for i in range(count):
                        # omit legacy composite layer
                        if i > 0:
                            layers.add_layer([obj, ], name='Layer '+str(i), layer_type='COLOR', color_attribute='VertexColor'+str(i), clear=False)

                    layers.add_layer([obj, ], name='Gradient1', layer_type='COLOR')
                    layers.add_layer([obj, ], name='Gradient2', layer_type='COLOR')
                    layers.add_layer([obj, ], name='Overlay', layer_type='COLOR')

                    layers.add_layer([obj, ], name='Occlusion', layer_type='OCC')
                    layers.add_layer([obj, ], name='Metallic', layer_type='MET')
                    layers.add_layer([obj, ], name='Roughness', layer_type='RGH')
                    layers.add_layer([obj, ], name='Transmission', layer_type='TRN')
                    layers.add_layer([obj, ], name='Emission', layer_type='EMI')

                    for uv in obj.data.uv_layers:
                        if uv.name == 'UVSet0':
                            pass
                        elif uv.name == 'UVSet1':
                            # grab occlusion from V
                            values = layers.get_uvs(obj, uv.name, 'V')
                            colors = convert.values_to_colors(values)
                            layers.set_layer(obj, colors, obj.sx2layers['Occlusion'])
                        elif uv.name == 'UVSet2':
                            # grab transmission, emission
                            values = layers.get_uvs(obj, uv.name, 'U')
                            colors = convert.values_to_colors(values)
                            layers.set_layer(obj, colors, obj.sx2layers['Transmission'])
                            values = layers.get_uvs(obj, uv.name, 'V')
                            colors = convert.values_to_colors(values, with_mask=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Emission'])
                        elif uv.name == 'UVSet3':
                            # grab metallic, smoothness
                            values = layers.get_uvs(obj, uv.name, 'U')
                            colors = convert.values_to_colors(values)
                            layers.set_layer(obj, colors, obj.sx2layers['Metallic'])
                            values = layers.get_uvs(obj, uv.name, 'V')
                            colors = convert.values_to_colors(values, invert=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Roughness'])
                        elif uv.name == 'UVSet4':
                            # grab gradient 1, 2
                            values = layers.get_uvs(obj, uv.name, 'U')
                            colors = convert.values_to_colors(values, as_alpha=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Gradient1'])
                            values = layers.get_uvs(obj, uv.name, 'V')
                            colors = convert.values_to_colors(values, as_alpha=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Gradient2'])
                        elif uv.name == 'UVSet5':
                            # grab and store overlay RG
                            pass
                        elif uv.name == 'UVSet6':
                            # grab overlay BA
                            pass

                    sx_category = categories[obj['sxtools'].get('category', 0) + 1].upper()
                    if sx_category == 'DEFAULT':
                        sx_category = 'STATIC'
                    elif sx_category == 'TRANSPARENT':
                        sx_category = 'STATIC'
                    obj.sx2.category = sx_category
                    category_data = sxglobals.category_dict[sxglobals.preset_lookup[obj.sx2.category]]

                    for i in range(len(category_data['layers'])):
                        layer = utils.find_layer_by_stack_index(obj, i)
                        if layer is None:
                            layer_dict = layers.add_layer([obj, ], name=category_data["layers"][i][0], layer_type=category_data["layers"][i][1])
                            layer = layer_dict[obj]
                        else:
                            layer.name = category_data["layers"][i][0]
                            layer.layer_type = category_data["layers"][i][1]
                        
                        layer.blend_mode = category_data["layers"][i][2]
                        layer.opacity = category_data["layers"][i][3]
                        if category_data["layers"][i][4] == "None":
                            layer.paletted = False
                        else:
                            layer.paletted = True
                            layer.palette_index = category_data["layers"][i][4]

                    obj.sx2.staticvertexcolors = str(category_data['staticvertexcolors'])
                    obj.sx2.metallicoverride = bool(category_data['metallic_override'])
                    obj.sx2.metallic0 = category_data['palette_metallic'][0]
                    obj.sx2.metallic1 = category_data['palette_metallic'][1]
                    obj.sx2.metallic2 = category_data['palette_metallic'][2]
                    obj.sx2.metallic3 = category_data['palette_metallic'][3]
                    obj.sx2.metallic4 = category_data['palette_metallic'][4]
                    obj.sx2.roughnessoverride = bool(category_data['roughness_override'])
                    obj.sx2.roughness0 = category_data['palette_roughness'][0]
                    obj.sx2.roughness1 = category_data['palette_roughness'][1]
                    obj.sx2.roughness2 = category_data['palette_roughness'][2]
                    obj.sx2.roughness3 = category_data['palette_roughness'][3]
                    obj.sx2.roughness4 = category_data['palette_roughness'][4]
                    obj.sx2.selectedlayer = 1

                    utils.sort_stack_indices(obj)
                    print(f'SX Tools: Converted {obj.name}, assigned to {sx_category}')

                modifiers.remove_modifiers(objs)
                if ('sx_tiler' in bpy.data.node_groups):
                    bpy.data.node_groups.remove(bpy.data.node_groups['sx_tiler'], do_unlink=True)
                modifiers.add_modifiers(objs)

                sxglobals.refresh_in_progress = False
                utils.mode_manager(objs, set_mode=False, mode_id='sxtosx2')

                # Delete redundant object properties and the legacy material
                for obj in objs:
                    to_delete = []
                    for key in obj.keys():
                        if '_visibility' in key:
                            to_delete.append(key)
                        if key == 'sxtools':
                            to_delete.append(key)
                        if key == 'sxlayers':
                            to_delete.append(key)
                    for key in to_delete:
                        del obj[key]

                if 'SXMaterial' in bpy.data.materials:
                    bpy.data.materials.remove(bpy.data.materials['SXMaterial'])

                sxglobals.magic_in_progress = False
                setup.update_sx2material(context)
                refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_exportatlases(bpy.types.Operator):
    bl_idname = 'sx2.exportatlases'
    bl_label = 'Export FBX files with atlases'
    bl_description = 'Export FBX files with palette atlas texture files'
    bl_options = {'UNDO'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if objs:
            material_sources = ['Occlusion', 'Metallic', 'Roughness', 'Transmission', 'Subsurface', 'Emission']
            export.composite_color_layers(objs)
            raw_palette = utils.find_colors_by_frequency(objs, 'Composite')
            palette = []
            for color in raw_palette:
                if len(palette) == 0:
                    palette.append(color)
                else:
                    match = False
                    for filtered_color in palette:
                        if utils.color_compare(color, filtered_color, 0.01):
                            match = True
                    if not match:
                        palette.append(color)

            grid = 1 if len(palette) == 0 else 2**math.ceil(math.log2(math.sqrt(len(palette))))

            # assign UV coords per palette color
            color_uv_coords = {}
            offset = 1.0/grid * 0.5
            j = -1
            for i, color in enumerate(palette):
                a = i % grid
                if a == 0:
                    j += 1
                u = a*offset*2 + offset
                v = j*offset*2 + offset
                color_uv_coords[color] = [u, v]

            # make sure reserved UV sets exist
            export.create_reserved_uvsets(objs)

            # match vertex color with palette color, assign UV accordingly
            face_colors = True
            for obj in objs:
                colors = layers.get_layer(obj, obj.sx2layers['Composite'], as_tuple=True)
                uvs = layers.get_uvs(obj, 'UVSet0')

                for i, color in enumerate(colors):
                    for key in color_uv_coords.keys():
                        if utils.color_compare(color, key, 0.01):
                            uvs[i*2:i*2+2] = color_uv_coords[key]

                # second loop, snap all loop verts into the color of the first to reduce artifacting
                for poly in obj.data.polygons:
                    loop_uv = None
                    for i, loop_idx in enumerate(poly.loop_indices):
                        if i == 0:
                            loop_uv = obj.data.uv_layers['UVSet0'].data[loop_idx].uv[:]
                        else:
                            if obj.data.uv_layers['UVSet0'].data[loop_idx].uv != loop_uv:
                                obj.data.uv_layers['UVSet0'].data[loop_idx].uv = loop_uv
                                face_colors = False

                layers.set_uvs(obj, 'UVSet0', uvs)

            # build material channel atlases by finding the most frequent channel value per palette entry
            palette_dict = {'Composite': palette}

            for source in material_sources:
                if source in objs[0].sx2layers.keys():
                    palette_dict[source] = [None] * len(palette)

            for i, color in enumerate(palette):
                # color = convert.linear_to_srgb(color)
                for source in material_sources:
                    if source in objs[0].sx2layers.keys():
                        typical_color = utils.find_colors_by_frequency(objs, source, numcolors=1, maskcolor=color)[0]
                        palette_dict[source][i] = typical_color

            files.export_palette_atlases(palette_dict)
            files.export_files_simple(objs)
            refresh_swatches(self, context)
            if not face_colors:
                message_box('Objects have non-uniform face colors.\nPlease assign colors per-face only,\nor switch to vertex color exporting.')

        return {'FINISHED'}


class SXTOOLS2_OT_testbutton(bpy.types.Operator):
    bl_idname = 'sx2.testbutton'
    bl_label = 'Test Button'
    bl_description = 'Test button for features in development'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        return {'FINISHED'}



# ------------------------------------------------------------------------
#    Registration and initialization
# ------------------------------------------------------------------------
sxglobals = SXTOOLS2_sxglobals()
utils = SXTOOLS2_utils()
convert = SXTOOLS2_convert()
generate = SXTOOLS2_generate()
layers = SXTOOLS2_layers()
tools = SXTOOLS2_tools()
setup = SXTOOLS2_setup()
files = SXTOOLS2_files()
modifiers = SXTOOLS2_modifiers()
export = SXTOOLS2_export()
magic = SXTOOLS2_magic()

core_classes = (
    SXTOOLS2_objectprops,
    SXTOOLS2_sceneprops,
    SXTOOLS2_layerprops,
    SXTOOLS2_preferences,
    SXTOOLS2_masterpalette,
    SXTOOLS2_material,
    SXTOOLS2_rampcolor,
    SXTOOLS2_PT_panel,
    SXTOOLS2_UL_layerlist,
    SXTOOLS2_MT_piemenu,
    SXTOOLS2_OT_selectionmonitor,
    SXTOOLS2_OT_keymonitor,
    SXTOOLS2_OT_layerinfo,
    SXTOOLS2_OT_loadlibraries,
    SXTOOLS2_OT_applytool,
    SXTOOLS2_OT_add_layer,
    SXTOOLS2_OT_del_layer,
    SXTOOLS2_OT_layer_up,
    SXTOOLS2_OT_layer_down,
    SXTOOLS2_OT_layer_props,
    SXTOOLS2_OT_add_variant,
    SXTOOLS2_OT_del_variant,
    SXTOOLS2_OT_prev_variant,
    SXTOOLS2_OT_next_variant,
    SXTOOLS2_OT_prev_palette,
    SXTOOLS2_OT_next_palette,
    SXTOOLS2_OT_mergeup,
    SXTOOLS2_OT_mergedown,
    SXTOOLS2_OT_copylayer,
    SXTOOLS2_OT_pastelayer,
    SXTOOLS2_OT_clearlayers,
    SXTOOLS2_OT_selmask,
    SXTOOLS2_OT_add_ramp,
    SXTOOLS2_OT_del_ramp,
    SXTOOLS2_OT_layers_to_palette,
    SXTOOLS2_OT_addpalette,
    SXTOOLS2_OT_delpalette,
    SXTOOLS2_OT_addpalettecategory,
    SXTOOLS2_OT_delpalettecategory,
    SXTOOLS2_OT_addmaterial,
    SXTOOLS2_OT_delmaterial,
    SXTOOLS2_OT_addmaterialcategory,
    SXTOOLS2_OT_delmaterialcategory,
    SXTOOLS2_OT_selectup,
    SXTOOLS2_OT_selectdown)

modifier_classes = (
    SXTOOLS2_OT_setgroup,
    SXTOOLS2_OT_previewpalette,
    SXTOOLS2_OT_applypalette,
    SXTOOLS2_OT_applymaterial,
    SXTOOLS2_OT_addmodifiers,
    SXTOOLS2_OT_hidemodifiers,
    SXTOOLS2_OT_applymodifiers,
    SXTOOLS2_OT_removemodifiers)

export_classes = (
    SXTOOLS2_OT_exportfiles,
    SXTOOLS2_OT_exportgroups,
    SXTOOLS2_OT_setpivots,
    SXTOOLS2_OT_removeexports,
    SXTOOLS2_OT_createuv0,
    SXTOOLS2_OT_groupobjects,
    SXTOOLS2_OT_generatelods,
    SXTOOLS2_OT_revertobjects,
    SXTOOLS2_OT_zeroverts,
    SXTOOLS2_OT_bytestofloats,
    SXTOOLS2_OT_macro,
    SXTOOLS2_OT_checklist,
    SXTOOLS2_OT_catalogue_add,
    SXTOOLS2_OT_catalogue_remove,
    SXTOOLS2_OT_create_sxcollection,
    SXTOOLS2_OT_smart_separate,
    SXTOOLS2_OT_generate_hulls,
    SXTOOLS2_OT_generate_emission_meshes,
    SXTOOLS2_OT_generatemasks,
    SXTOOLS2_OT_resetscene,
    SXTOOLS2_OT_testbutton,
    SXTOOLS2_OT_sxtosx2,
    SXTOOLS2_OT_exportatlases)

addon_keymaps = []


def register():
    from bpy.utils import register_class
    for cls in core_classes:
        register_class(cls)
    for cls in modifier_classes:
        register_class(cls)
    for cls in export_classes:
        register_class(cls)

    bpy.types.Object.sx2 = bpy.props.PointerProperty(type=SXTOOLS2_objectprops)
    bpy.types.Object.sx2layers = bpy.props.CollectionProperty(type=SXTOOLS2_layerprops)
    bpy.types.Scene.sx2 = bpy.props.PointerProperty(type=SXTOOLS2_sceneprops)
    bpy.types.Scene.sx2palettes = bpy.props.CollectionProperty(type=SXTOOLS2_masterpalette)
    bpy.types.Scene.sx2materials = bpy.props.CollectionProperty(type=SXTOOLS2_material)

    bpy.app.handlers.load_post.append(load_post_handler)
    bpy.app.handlers.save_pre.append(save_pre_handler)
    bpy.app.handlers.save_post.append(save_post_handler)

    if not bpy.app.background:
        bpy.types.SpaceView3D.draw_handler_add(start_modal, (), 'WINDOW', 'POST_VIEW')

    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new('sx2.selectup', 'UP_ARROW', 'PRESS', ctrl=True, shift=True)
        addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('sx2.selectdown', 'DOWN_ARROW', 'PRESS', ctrl=True, shift=True)
        addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('wm.call_menu_pie', 'COMMA', 'PRESS', shift=True)
        kmi.properties.name = SXTOOLS2_MT_piemenu.bl_idname
        addon_keymaps.append((km, kmi))


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(export_classes):
        unregister_class(cls)
    for cls in reversed(modifier_classes):
        unregister_class(cls)
    for cls in reversed(core_classes):
        unregister_class(cls)

    del bpy.types.Object.sx2
    del bpy.types.Object.sx2layers
    del bpy.types.Scene.sx2
    del bpy.types.Scene.sx2palettes
    del bpy.types.Scene.sx2materials

    bpy.app.handlers.load_post.remove(load_post_handler)
    bpy.app.handlers.save_pre.remove(save_pre_handler)
    bpy.app.handlers.save_post.remove(save_post_handler)

    if not bpy.app.background:
        modal_list = [op.name for op in bpy.context.window.modal_operators]
        if ('SX2_OT_selectionmonitor' in modal_list) or ('SX2_OT_keymonitor' in modal_list):
            bpy.types.SpaceView3D.draw_handler_remove(start_modal, 'WINDOW')

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        for km, kmi in addon_keymaps:
            km.keymap_items.remove(kmi)
    addon_keymaps.clear()

if __name__ == '__main__':
    try:
        unregister()
    except:
        pass
    register()


# TODO:
# BUG: Grouping of objs with armatures
# FEAT: match existing layers when loading category
# FEAT: review non-metallic PBR material values
