bl_info = {
    'name': 'SX Tools 2',
    'author': 'Jani Kahrama / Secret Exit Ltd.',
    'version': (0, 0, 1),
    'blender': (3, 4, 0),
    'location': 'View3D',
    'description': 'Multi-layer vertex coloring tool',
    'doc_url': 'https://www.notion.so/SX-Tools-for-Blender-Documentation-9ad98e239f224624bf98246822a671a6',
    'tracker_url': 'https://github.com/FrandSX/sxtools-blender/issues',
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
import numpy as np
from bpy.app.handlers import persistent
from collections import Counter
from mathutils import Vector


# ------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------
class SXTOOLS2_sxglobals(object):
    def __init__(self):
        self.librariesLoaded = False
        self.refresh_in_progress = False
        self.magic_in_progress = False
        self.hsl_update = False
        self.mat_update = False
        self.curvature_update = False
        self.modal_status = False
        self.copy_buffer = {}
        self.prev_mode = 'OBJECT'
        self.mode = None
        self.modeID = None
        self.randomseed = 42

        self.prev_selection = []
        self.prev_component_selection = []
        self.ramp_dict = {}
        self.categoryDict = {}
        self.preset_lookup = {}
        self.paletteDict = {}
        self.masterPaletteArray = []
        self.materialArray = []

        self.default_colors = {
            'COLOR': (0.0, 0.0, 0.0, 0.0),
            'OCC': (1.0, 1.0, 1.0, 1.0),
            'MET': (0.0, 0.0, 0.0, 0.0),
            'RGH': (0.5, 0.5, 0.5, 1.0),
            'TRN': (0.0, 0.0, 0.0, 0.0),
            'SSS': (0.0, 0.0, 0.0, 0.0),
            'EMI': (0.0, 0.0, 0.0, 0.0),
            'CMP': (0.0, 0.0, 0.0, 0.0)
            }

        # {obj.name: {stack_layer_index: (layer.name, layer.color_attribute, layer.layer_type)}}
        self.layer_stack_dict = {}

        # Keywords used by Smart Separate. Avoid using these in regular object names
        self.keywords = ['_org', '_LOD0', '_top', '_bottom', '_front', '_rear', '_left', '_right']


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
        filePath = directory + mode + '.json'

        if len(directory) > 0:
            try:
                with open(filePath, 'r') as input:
                    temp_dict = {}
                    temp_dict = json.load(input)
                    if mode == 'palettes':
                        del sxglobals.masterPaletteArray[:]
                        bpy.context.scene.sx2palettes.clear()
                        sxglobals.masterPaletteArray = temp_dict['Palettes']
                    elif mode == 'materials':
                        del sxglobals.materialArray[:]
                        bpy.context.scene.sx2materials.clear()
                        sxglobals.materialArray = temp_dict['Materials']
                    elif mode == 'gradients':
                        sxglobals.ramp_dict.clear()
                        sxglobals.ramp_dict = temp_dict
                    elif mode == 'categories':
                        sxglobals.categoryDict.clear()
                        sxglobals.categoryDict = temp_dict

                    input.close()
                print(f'SX Tools: {mode} loaded from {filePath}')
            except ValueError:
                print(f'SX Tools Error: Invalid {mode} file.')
                prefs.libraryfolder = ''
                return False
            except IOError:
                print(f'SX Tools Error: {mode} file not found!')
                return False
        else:
            print(f'SX Tools: No {mode} file found')
            return False

        if mode == 'palettes':
            self.load_swatches(sxglobals.masterPaletteArray)
            return True
        elif mode == 'materials':
            self.load_swatches(sxglobals.materialArray)
            return True
        else:
            return True


    def save_file(self, mode):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        directory = prefs.libraryfolder
        filePath = directory + mode + '.json'
        # Palettes.json Materials.json

        if len(directory) > 0:
            with open(filePath, 'w') as output:
                if mode == 'palettes':
                    temp_dict = {}
                    temp_dict['Palettes'] = sxglobals.masterPaletteArray
                    json.dump(temp_dict, output, indent=4)
                elif mode == 'materials':
                    temp_dict = {}
                    temp_dict['Materials'] = sxglobals.materialArray
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


    def load_swatches(self, swatcharray):
        if swatcharray == sxglobals.materialArray:
            swatchcount = 3
            sxlist = bpy.context.scene.sx2materials
        elif swatcharray == sxglobals.masterPaletteArray:
            swatchcount = 5
            sxlist = bpy.context.scene.sx2palettes

        for categoryDict in swatcharray:
            for category in categoryDict:
                if len(categoryDict[category]) == 0:
                    item = sxlist.add()
                    item.name = 'Empty'
                    item.category = category
                    for i in range(swatchcount):
                        incolor = [0.0, 0.0, 0.0, 1.0]
                        setattr(item, 'color'+str(i), incolor[:])
                else:
                    for entry in categoryDict[category]:
                        item = sxlist.add()
                        item.name = entry
                        item.category = category
                        for i in range(swatchcount):
                            incolor = [0.0, 0.0, 0.0, 1.0]
                            incolor[0] = categoryDict[category][entry][i][0]
                            incolor[1] = categoryDict[category][entry][i][1]
                            incolor[2] = categoryDict[category][entry][i][2]
                            setattr(item, 'color'+str(i), convert.srgb_to_linear(incolor))


    def save_ramp(self, rampName):
        temp_dict = {}
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['ColorRamp'].color_ramp

        temp_dict['mode'] = ramp.color_mode
        temp_dict['interpolation'] = ramp.interpolation
        temp_dict['hue_interpolation'] = ramp.hue_interpolation

        tempColorArray = []
        for i, element in enumerate(ramp.elements):
            tempElement = [None, [None, None, None, None], None]
            tempElement[0] = element.position
            tempElement[1] = [element.color[0], element.color[1], element.color[2], element.color[3]]
            tempColorArray.append(tempElement)

        temp_dict['elements'] = tempColorArray
        sxglobals.ramp_dict[rampName] = temp_dict

        self.save_file('gradients')


    # In paletted export mode, gradients and overlays are
    # not composited as that will be
    # done by the shader on the game engine side
    def export_files(self, groups):
        scene = bpy.context.scene.sx2
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        export_dict = {'LIN': 'LINEAR', 'SRGB': 'SRGB'}
        empty = True

        if 'ExportObjects' not in bpy.data.collections:
            exportObjects = bpy.data.collections.new('ExportObjects')
        else:
            exportObjects = bpy.data.collections['ExportObjects']

        groupNames = []
        for group in groups:
            bpy.context.view_layer.objects.active = group
            bpy.ops.object.select_all(action='DESELECT')
            group.select_set(True)
            org_loc = group.location.copy()
            group.location = (0, 0, 0)

            selArray = utils.find_children(group, recursive=True)
            for sel in selArray:
                sel.select_set(True)
            group.select_set(False)

            # Check for mesh colliders
            collider_array = []
            if 'SXColliders' in bpy.data.collections:
                colliders = bpy.data.collections['SXColliders'].objects
                if group.name.endswith('_root'):
                    collider_id = group.name[:-5]
                else:
                    collider_id = group.name

                for collider in colliders:
                    if collider_id in collider.name:
                        collider_array.append(collider)
                        collider['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))

            # Only groups with meshes and armatures as children are exported
            objArray = []
            for sel in selArray:
                if sel.type == 'MESH':
                    objArray.append(sel)
                    sel['staticVertexColors'] = sel.sx2.staticvertexcolors
                    sel['sxToolsVersion'] = 'SX Tools 2 for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))
                    sel['colorSpace'] = export_dict[prefs.exportspace]

            if len(objArray) > 0:
                empty = False

                # Create palette masks
                layers.generate_palette_masks(objArray)

                for obj in objArray:
                    bpy.context.view_layer.objects.active = obj
                    compLayers = utils.find_comp_layers(obj, obj['staticVertexColors'])
                    layer0 = utils.find_color_layers(obj, 0)
                    layer1 = utils.find_color_layers(obj, 1)
                    layers.blend_layers([obj, ], compLayers, layer1, layer0, single_as_alpha=True)

                    obj.data.attributes.active_color = obj.data.attributes[layer0.vertexColorLayer]
                    bpy.ops.geometry.color_attribute_render_set(name=layer0.vertexColorLayer)

                    # bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                    if 'sxTiler' in obj.modifiers:
                        obj.modifiers.remove(obj.modifiers['sxTiler'])
                        obj.data.use_auto_smooth = True

                # bpy.context.view_layer.update()
                bpy.context.view_layer.objects.active = group

                category = objArray[0].sx2.category.lower()
                print(f'Determining path: {objArray[0].name} {category}')
                path = scene.exportfolder + category + os.path.sep
                pathlib.Path(path).mkdir(exist_ok=True)

                if len(collider_array) > 0:
                    for collider in collider_array:
                        collider.select_set(True)

                exportPath = path + group.name + '.' + 'fbx'
                export_settings = ['FBX_SCALE_UNITS', False, False, False, 'Z', '-Y', '-Y', '-X', export_dict[prefs.exportspace]]

                bpy.ops.export_scene.fbx(
                    filepath=exportPath,
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

                groupNames.append(group.name)
                group.location = org_loc

                modifiers.remove_modifiers(objArray)
                modifiers.add_modifiers(objArray)

                bpy.context.view_layer.objects.active = group

        if empty:
            message_box('No objects exported!')
        else:
            message_box('Exported:\n' + str('\n').join(groupNames))
            for group_name in groupNames:
                print(f'Completed: {group_name}')


    def export_palette_texture(self, colors):
        swatch_size = 16
        grid = 1 if len(colors) == 0 else 2**math.ceil(math.log2(math.sqrt(len(colors))))
        pixels = [0.0, 0.0, 0.0, 0.1] * grid * grid * swatch_size * swatch_size

        z = 0
        for y in range(grid):
            row = [0.0, 0.0, 0.0, 1.0] * grid * swatch_size
            row_length = 4 * grid * swatch_size
            for x in range(grid):
                color = colors[z] if z < len(colors) else (0.0, 0.0, 0.0, 1.0)
                color = convert.linear_to_srgb(color)
                z += 1
                for i in range(swatch_size):
                    row[x*swatch_size*4+i*4:x*swatch_size*4+i*4+4] = [color[0], color[1], color[2], color[3]]
            for j in range(swatch_size):
                pixels[y*swatch_size*row_length+j*row_length:y*swatch_size*row_length+(j+1)*row_length] = row

        image = bpy.data.images.new('palette_atlas', alpha=True, width=grid*swatch_size, height=grid*swatch_size)
        image.alpha_mode = 'STRAIGHT'
        image.pixels = pixels
        image.filepath_raw = '/tmp/palette_atlas.png'
        image.file_format = 'PNG'
        image.save()


# ------------------------------------------------------------------------
#    Useful Miscellaneous Functions
# ------------------------------------------------------------------------
class SXTOOLS2_utils(object):
    def __init__(self):
        return None


    def mode_manager(self, objs, set_mode=False, revert=False, mode_id=None):
        if set_mode:
            if sxglobals.modeID is None:
                sxglobals.modeID = mode_id
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

        elif revert:
            if sxglobals.modeID == mode_id:
                if bpy.context.view_layer.objects.active is None:
                    bpy.context.view_layer.objects.active = objs[0]
                bpy.ops.object.mode_set(mode=sxglobals.mode)
                sxglobals.modeID = None


    def find_layer_by_stack_index(self, obj, index):
        if len(obj.sx2layers) > 0:
            stack_dict = sxglobals.layer_stack_dict.get(obj.name, {})
            if len(stack_dict) == 0:
                layers = []
                for layer in obj.sx2layers:
                    layers.append((layer.index, layer.color_attribute))

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


    def find_colors_by_frequency(self, objs, layer, numcolors=None, masklayer=None, obj_sel_override=False):
        colorArray = []

        for obj in objs:
            if obj_sel_override:
                values = layers.get_layer(obj, layer, as_tuple=True)
            else:
                values = generate.mask_list(obj, layers.get_layer(obj, layer), masklayer=masklayer, as_tuple=True)

            if values is not None:
                colorArray.extend(values)

        # colors = list(filter(lambda a: a != (0.0, 0.0, 0.0, 0.0), colorArray))
        colors = list(filter(lambda a: a[3] != 0.0, colorArray))
        sortList = [color for color, count in Counter(colors).most_common(numcolors)]

        if numcolors is not None:
            while len(sortList) < numcolors:
                sortList.append([0.0, 0.0, 0.0, 1.0])

            sortColors = []
            for i in range(numcolors):
                sortColors.append(sortList[i])

            return sortColors
        else:
            return sortList


    def find_color_layers(self, obj, index=None):
        color_layers = []
        if len(obj.sx2layers) > 0:
            for layer in obj.sx2layers:
                if layer.layer_type == 'COLOR':
                    color_layers.append(layer)
            color_layers.sort(key=lambda x: x.index)
            return color_layers if index is None else color_layers[index]
        else:
            return None


    def find_root_pivot(self, objs):
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_object_bounding_box(objs)
        pivot = ((xmax + xmin)*0.5, (ymax + ymin)*0.5, zmin)

        return pivot


    # ARMATUREs check for grandparent
    def find_children(self, group, objs=None, recursive=False):
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
                if len(child_list) > 0:
                    results.extend(child_list)
                    child_recurse(child_list)

        results = []
        if objs is None:
            objs = bpy.data.objects.values()

        if recursive:
            child_list = [group, ]
            child_recurse(child_list)
            return results
        else:
            return get_children(group)


    # Finds groups to be exported,
    # only EMPTY objects with no parents
    # treat ARMATURE objects as a special case
    def find_groups(self, objs, all_groups=False):
        groups = []
        if all_groups:
            objs = bpy.context.view_layer.objects

        for obj in objs:
            if (obj.type == 'EMPTY') and (obj.parent is None):
                groups.append(obj)

            parent = obj.parent
            if (parent is not None) and (parent.type == 'EMPTY') and (parent.parent is None):
                groups.append(obj.parent)
            elif (parent is not None) and (parent.type == 'ARMATURE') and (parent.parent is not None) and (parent.parent.type == 'EMPTY'):
                groups.append(parent.parent)

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


    def get_object_bounding_box(self, objs, local=False):
        bbx_x = []
        bbx_y = []
        bbx_z = []
        for obj in objs:
            if local:
                corners = [Vector(corner) for corner in obj.bound_box]
            else:
                corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

            for corner in corners:
                bbx_x.append(corner[0])
                bbx_y.append(corner[1])
                bbx_z.append(corner[2])
        xmin, xmax = min(bbx_x), max(bbx_x)
        ymin, ymax = min(bbx_y), max(bbx_y)
        zmin, zmax = min(bbx_z), max(bbx_z)

        return xmin, xmax, ymin, ymax, zmin, zmax


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
        if len(stack_dict) > 0:
            sxglobals.layer_stack_dict[obj.name].clear()

        if len(obj.sx2layers) > 0:
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
        vec1 = Vector(color1)
        vec2 = Vector(color2)
        difference = vec1 - vec2

        return difference.length <= tolerance


    # if vertex pos x y z is at bbx limit, and mirror axis is set, round vertex position
    def round_tiling_verts(self, objs):
        for obj in objs:
            if obj.sx2.tiling:
                vert_dict = generate.vertex_data_dict(obj)
                xmin, xmax, ymin, ymax, zmin, zmax = self.get_object_bounding_box([obj, ], local=True)

                if len(vert_dict.keys()) > 0:
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
                    if len(task_list) == 0:
                        break

        bpy.context.view_layer.objects.active = active


    def __del__(self):
        print('SX Tools: Exiting utils')


# ------------------------------------------------------------------------
#    Color Conversions
# ------------------------------------------------------------------------
class SXTOOLS2_convert(object):
    def __init__(self):
        return None


    def color_to_luminance(self, in_rgba):
        lumR = 0.212655
        lumG = 0.715158
        lumB = 0.072187
        alpha = in_rgba[3]

        linLum = lumR * in_rgba[0] + lumG * in_rgba[1] + lumB * in_rgba[2]
        # luminance = convert.linear_to_srgb((linLum, linLum, linLum, alpha))[0]
        return linLum * alpha  # luminance * alpha


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
        R = in_rgba[0]
        G = in_rgba[1]
        B = in_rgba[2]
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


    def uvs_to_colors(self, obj, layer_name, channel, invert=False, as_alpha=False, with_mask=False):

        print('layer name:', layer_name, 'channel:', channel)
        uvs = layers.get_uvs(obj, layer_name, channel)
        if invert:
            for i, uv in enumerate(uvs):
                uvs[i] = 1.0 - uv
        count = len(uvs) 
        values = [None] * count * 4
        for i in range(count):
            if as_alpha:
                values[(0+i*4):(4+i*4)] = [1.0, 1.0, 1.0, uvs[i]]
            elif with_mask:
                alpha = 1.0 if uvs[i] > 0.0 else 0.0
                values[(0+i*4):(4+i*4)] = [uvs[i], uvs[i], uvs[i], alpha]
            else:
                values[(0+i*4):(4+i*4)] = self.luminance_to_color(uvs[i])
        return values


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


    def curvature_list(self, obj, masklayer=None, returndict=False):
        scene = bpy.context.scene.sx2
        normalize = scene.curvaturenormalize
        vert_curv_dict = {}
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.normal_update()

        for vert in bm.verts:
            numConnected = len(vert.link_edges)
            if numConnected > 0:
                edgeWeights = []
                angles = []
                for edge in vert.link_edges:
                    edgeWeights.append(edge.calc_length())
                    pos1 = vert.co
                    pos2 = edge.other_vert(vert).co
                    edgeVec = Vector((float(pos2[0] - pos1[0]), float(pos2[1] - pos1[1]), float(pos2[2] - pos1[2])))
                    angles.append(math.acos(vert.normal.normalized() @ edgeVec.normalized()))

                vtxCurvature = 0.0
                for i in range(numConnected):
                    curvature = angles[i] / math.pi - 0.5
                    vtxCurvature += curvature

                vtxCurvature = min(vtxCurvature / float(numConnected), 1.0)

                vert_curv_dict[vert.index] = vtxCurvature
            else:
                vert_curv_dict[vert.index] = 0.0

        bm.free()

        # Normalize convex and concave separately
        # to maximize artist ability to crease

        if normalize:
            minCurv = min(vert_curv_dict.values())
            maxCurv = max(vert_curv_dict.values())

            for vert, vtxCurvature in vert_curv_dict.items():
                if vtxCurvature < 0.0:
                    vert_curv_dict[vert] = (vtxCurvature / float(minCurv)) * -0.5 + 0.5
                elif vtxCurvature == 0.0:
                    vert_curv_dict[vert] = 0.5
                else:
                    vert_curv_dict[vert] = (vtxCurvature / float(maxCurv)) * 0.5 + 0.5
        else:
            for vert, vtxCurvature in vert_curv_dict.items():
                vert_curv_dict[vert] = (vtxCurvature + 0.5)

        if returndict:
            return vert_curv_dict

        else:
            # Clear the border edges if the object is tiling
            if obj.sx2.tiling:
                bpy.context.view_layer.objects.active = obj

                bpy.ops.object.mode_set(mode='EDIT', toggle=False)
                bpy.context.tool_settings.mesh_select_mode = (False, False, True)
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.region_to_loop()
                bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

                sel_verts = [None] * len(obj.data.vertices)
                obj.data.vertices.foreach_get('select', sel_verts)

                for i, sel in enumerate(sel_verts):
                    if sel:
                        vert_curv_dict[i] = 0.5

            vert_curv_list = self.vert_dict_to_loop_list(obj, vert_curv_dict, 1, 4)
            curv_list = self.mask_list(obj, vert_curv_list, masklayer)

            return curv_list


    def direction_list(self, obj, masklayer=None):
        scene = bpy.context.scene.sx2
        coneangle = scene.dirCone
        cone = coneangle*0.5
        random.seed(sxglobals.randomseed)

        if coneangle == 0:
            samples = 1
        else:
            samples = coneangle * 5

        vert_dir_dict = {}
        vert_dict = self.vertex_data_dict(obj, masklayer)

        if len(vert_dict.keys()) > 0:
            for vert_id in vert_dict:
                vert_dir_dict[vert_id] = 0.0

            for i in range(samples):
                inclination = math.radians(scene.dirInclination + random.uniform(-cone, cone) - 90.0)
                angle = math.radians(scene.dirAngle + random.uniform(-cone, cone) + 90)

                direction = Vector((math.sin(inclination) * math.cos(angle), math.sin(inclination) * math.sin(angle), math.cos(inclination)))

                for vert_id in vert_dict:
                    vertWorldNormal = Vector(vert_dict[vert_id][3])
                    vert_dir_dict[vert_id] += max(min(vertWorldNormal @ direction, 1.0), 0.0)

            values = np.array(self.vert_dict_to_loop_list(obj, vert_dir_dict, 1, 1))
            values *= 1.0/values.max()
            values = values.tolist()

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

        noise_dict = {}
        for vtx_id in vert_ids:
            noise_dict[vtx_id] = make_noise(amplitude, offset, mono)

        noise_list = self.vert_dict_to_loop_list(obj, noise_dict, 4, 4)
        return self.mask_list(obj, noise_list, masklayer)


    def ray_randomizer(self, count):
        hemiSphere = [None] * count
        random.seed(sxglobals.randomseed)

        for i in range(count):
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(u1)
            theta = 2*math.pi*u2

            x = r * math.cos(theta)
            y = r * math.sin(theta)

            hemiSphere[i] = (x, y, math.sqrt(max(0, 1 - u1)))

        return hemiSphere


    def ground_plane(self, size, pos):
        vertArray = []
        faceArray = []
        size *= 0.5

        vert = [(pos[0]-size, pos[1]-size, pos[2])]
        vertArray.extend(vert)
        vert = [(pos[0]+size, pos[1]-size, pos[2])]
        vertArray.extend(vert)
        vert = [(pos[0]-size, pos[1]+size, pos[2])]
        vertArray.extend(vert)
        vert = [(pos[0]+size, pos[1]+size, pos[2])]
        vertArray.extend(vert)

        face = [(0, 1, 3, 2)]
        faceArray.extend(face)

        mesh = bpy.data.meshes.new('groundPlane_mesh')
        groundPlane = bpy.data.objects.new('groundPlane', mesh)
        bpy.context.scene.collection.objects.link(groundPlane)

        mesh.from_pydata(vertArray, [], faceArray)
        mesh.update(calc_edges=True)

        # groundPlane.location = pos
        return groundPlane, mesh


    def thickness_list(self, obj, raycount, masklayer=None):

        def dist_hit(vert_id, loc, vertPos, dist_list):
            distanceVec = Vector((loc[0] - vertPos[0], loc[1] - vertPos[1], loc[2] - vertPos[2]))
            dist_list.append(distanceVec.length)

        def thick_hit(vert_id, loc, vertPos, dist_list):
            vert_occ_dict[vert_id] += contribution

        def ray_caster(obj, raycount, vert_dict, hitfunction, raydistance=1.70141e+38):
            hemiSphere = self.ray_randomizer(raycount)

            for vert_id in vert_dict:
                vert_occ_dict[vert_id] = 0.0
                vertLoc = Vector(vert_dict[vert_id][0])
                vertNormal = Vector(vert_dict[vert_id][1])
                bias = 0.001

                # Invert normal to cast inside object
                invNormal = tuple([-1*x for x in vertNormal])

                # Raycast for bias
                hit, loc, normal, index = obj.ray_cast(vertLoc, invNormal, distance=raydistance)
                if hit and (normal.dot(invNormal) < 0):
                    hit_dist = Vector((loc[0] - vertLoc[0], loc[1] - vertLoc[1], loc[2] - vertLoc[2])).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                biasVec = tuple([bias*x for x in invNormal])
                rotQuat = forward.rotation_difference(invNormal)

                # offset ray origin with normal bias
                vertPos = (vertLoc[0] + biasVec[0], vertLoc[1] + biasVec[1], vertLoc[2] + biasVec[2])

                for sample in hemiSphere:
                    sample = Vector(sample)
                    sample.rotate(rotQuat)

                    hit, loc, normal, index = obj.ray_cast(vertPos, sample, distance=raydistance)

                    if hit:
                        hitfunction(vert_id, loc, vertPos, dist_list)

        contribution = 1.0/float(raycount)
        forward = Vector((0.0, 0.0, 1.0))

        dist_list = []
        vert_occ_dict = {}
        vert_dict = self.vertex_data_dict(obj, masklayer)

        if len(vert_dict.keys()) > 0:
            for modifier in obj.modifiers:
                if modifier.type == 'SUBSURF':
                    modifier.show_viewport = False

            # First pass to analyze ray hit distances,
            # then set max ray distance to half of median distance
            ray_caster(obj, 20, vert_dict, dist_hit)
            distance = statistics.median(dist_list) * 0.5

            # Second pass for final results
            ray_caster(obj, raycount, vert_dict, thick_hit, raydistance=distance)

            for modifier in obj.modifiers:
                if modifier.type == 'SUBSURF':
                    modifier.show_viewport = True  # obj.sx2.modifiervisibility

            vert_occ_list = generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)
            return self.mask_list(obj, vert_occ_list, masklayer)
        else:
            return None


    def occlusion_list(self, obj, raycount=500, blend=0.5, dist=10.0, groundplane=False, masklayer=None):
        scene = bpy.context.scene
        contribution = 1.0/float(raycount)
        hemiSphere = self.ray_randomizer(raycount)
        mix = max(min(blend, 1.0), 0.0)
        forward = Vector((0.0, 0.0, 1.0))

        if obj.sx2.tiling:
            blend = 0.0
            groundplane = False
            obj.modifiers['sxTiler'].show_viewport = False
            bpy.context.view_layer.update()
            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ], local=True)
            dist = 2.0 * min(xmax-xmin, ymax-ymin, zmax-zmin)
            obj.modifiers['sxTiler'].show_viewport = True

        edg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(edg)

        vert_occ_dict = {}
        vert_dict = self.vertex_data_dict(obj, masklayer)

        if len(vert_dict.keys()) > 0:

            if groundplane:
                pivot = utils.find_root_pivot([obj, ])
                pivot = (pivot[0], pivot[1], -0.5)  # pivot[2] - 0.5)
                ground, groundmesh = self.ground_plane(20, pivot)

            for vert_id in vert_dict:
                bias = 0.001
                occValue = 1.0
                scnOccValue = 1.0
                vertLoc = Vector(vert_dict[vert_id][0])
                vertNormal = Vector(vert_dict[vert_id][1])
                vertWorldLoc = Vector(vert_dict[vert_id][2])
                vertWorldNormal = Vector(vert_dict[vert_id][3])

                # use modified tile-border normals to reduce seam artifacts
                # if vertex pos x y z is at bbx limit, and mirror axis is set, modify respective normal vector component to zero
                if obj.sx2.tiling:
                    mod_normal = [0.0, 0.0, 0.0]
                    match = False

                    for i, coord in enumerate(vertLoc):
                        if i == 0:
                            if obj.sx2.tile_neg_x and (round(coord, 2) == round(xmin, 2)):
                                match = True
                                mod_normal[i] = 0.0
                            elif obj.sx2.tile_pos_x and (round(coord, 2) == round(xmax, 2)):
                                match = True
                                mod_normal[i] = 0.0
                            else:
                                mod_normal[i] = vertNormal[i]
                        elif i == 1:
                            if obj.sx2.tile_neg_y and (round(coord, 2) == round(ymin, 2)):
                                match = True
                                mod_normal[i] = 0.0
                            elif obj.sx2.tile_pos_y and (round(coord, 2) == round(ymax, 2)):
                                match = True
                                mod_normal[i] = 0.0
                            else:
                                mod_normal[i] = vertNormal[i]
                        else:
                            if obj.sx2.tile_neg_z and (round(coord, 2) == round(zmin, 2)):
                                match = True
                                mod_normal[i] = 0.0
                            elif obj.sx2.tile_pos_z and (round(coord, 2) == round(zmax, 2)):
                                match = True
                                mod_normal[i] = 0.0
                            else:
                                mod_normal[i] = vertNormal[i]

                    if match:
                        vertNormal = Vector(mod_normal[:]).normalized()

                # Pass 0: Raycast for bias
                hit, loc, normal, index = obj.ray_cast(vertLoc, vertNormal, distance=dist)
                if hit and (normal.dot(vertNormal) > 0):
                    hit_dist = Vector((loc[0] - vertLoc[0], loc[1] - vertLoc[1], loc[2] - vertLoc[2])).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                # Pass 1: Local space occlusion for individual object
                if 0.0 <= mix < 1.0:
                    biasVec = tuple([bias*x for x in vertNormal])
                    rotQuat = forward.rotation_difference(vertNormal)

                    # offset ray origin with normal bias
                    vertPos = (vertLoc[0] + biasVec[0], vertLoc[1] + biasVec[1], vertLoc[2] + biasVec[2])

                    for sample in hemiSphere:
                        sample = Vector(sample)
                        sample.rotate(rotQuat)

                        hit, loc, normal, index = obj_eval.ray_cast(vertPos, sample, distance=dist)

                        if hit:
                            occValue -= contribution

                # Pass 2: Worldspace occlusion for scene
                if 0.0 < mix <= 1.0:
                    biasVec = tuple([bias*x for x in vertWorldNormal])
                    rotQuat = forward.rotation_difference(vertWorldNormal)

                    # offset ray origin with normal bias
                    scnVertPos = (vertWorldLoc[0] + biasVec[0], vertWorldLoc[1] + biasVec[1], vertWorldLoc[2] + biasVec[2])

                    for sample in hemiSphere:
                        sample = Vector(sample)
                        sample.rotate(rotQuat)

                        scnHit, scnLoc, scnNormal, scnIndex, scnObj, ma = scene.ray_cast(edg, scnVertPos, sample, distance=dist)
                        # scene.ray_cast(scene.view_layers[0].depsgraph, scnVertPos, sample, distance=dist)

                        if scnHit:
                            scnOccValue -= contribution

                vert_occ_dict[vert_id] = float((occValue * (1.0 - mix)) + (scnOccValue * mix))

            if groundplane:
                bpy.data.objects.remove(ground, do_unlink=True)
                bpy.data.meshes.remove(groundmesh, do_unlink=True)

            if obj.sx2.tiling:
                obj.modifiers['sxTiler'].show_viewport = False
                obj.data.use_auto_smooth = True

            vert_occ_list = generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)
            return self.mask_list(obj, vert_occ_list, masklayer)

        else:
            return None


    def mask_list(self, obj, colors, masklayer=None, as_tuple=False, override_mask=False):
        count = len(colors)//4

        if (masklayer is None) and (sxglobals.mode == 'OBJECT'):
            if as_tuple:
                rgba = [None] * count
                for i in range(count):
                    rgba[i] = tuple(colors[(0+i*4):(4+i*4)])
                return rgba
            else:
                return colors
        else:
            if masklayer is None:
                mask, empty = self.selection_mask(obj)
                if empty:
                    return None
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
        count = len(obj.data.color_attributes[0].data)
        colors = [color[0], color[1], color[2], color[3]] * count
        return self.mask_list(obj, colors, masklayer, as_tuple)


    def ramp_list(self, obj, objs, rampmode, masklayer=None, mergebbx=True):
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['ColorRamp']

        # For OBJECT mode selections
        if sxglobals.mode == 'OBJECT':
            if mergebbx:
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box(objs)
            else:
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ])

        # For EDIT mode multi-obj component selection
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_selection_bounding_box(objs)

        vertPosDict = self.vertex_data_dict(obj, masklayer)
        ramp_dict = {}

        for vert_id in vertPosDict:
            ratioRaw = None
            ratio = None
            fvPos = vertPosDict[vert_id][2]

            if rampmode == 'X':
                xdiv = float(xmax - xmin)
                if xdiv == 0.0:
                    xdiv = 1.0
                ratioRaw = ((fvPos[0] - xmin) / xdiv)
            elif rampmode == 'Y':
                ydiv = float(ymax - ymin)
                if ydiv == 0.0:
                    ydiv = 1.0
                ratioRaw = ((fvPos[1] - ymin) / ydiv)
            elif rampmode == 'Z':
                zdiv = float(zmax - zmin)
                if zdiv == 0.0:
                    zdiv = 1.0
                ratioRaw = ((fvPos[2] - zmin) / zdiv)

            ratio = max(min(ratioRaw, 1.0), 0.0)
            ramp_dict[vert_id] = ramp.color_ramp.evaluate(ratio)

        ramp_list = self.vert_dict_to_loop_list(obj, ramp_dict, 4, 4)

        return self.mask_list(obj, ramp_list, masklayer)


    def luminance_remap_list(self, obj, layer=None, masklayer=None, values=None):
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['ColorRamp']
        if values is None:
            values = layers.get_luminances(obj, layer, as_rgba=False)
        colors = generate.empty_list(obj, 4)
        count = len(values)

        for i in range(count):
            ratio = max(min(values[i], 1.0), 0.0)
            colors[(0+i*4):(4+i*4)] = ramp.color_ramp.evaluate(ratio)

        return self.mask_list(obj, colors, masklayer)


    def vertex_id_list(self, obj):
        mesh = obj.data
        count = len(mesh.vertices)
        ids = [None] * count
        mesh.vertices.foreach_get('index', ids)
        return ids


    def empty_list(self, obj, channelcount):
        count = len(obj.data.uv_layers[0].data)
        looplist = [0.0] * count * channelcount
        return looplist


    def vert_dict_to_loop_list(self, obj, vert_dict, dictchannelcount, listchannelcount):
        mesh = obj.data
        loop_list = self.empty_list(obj, listchannelcount)

        if dictchannelcount < listchannelcount:
            if (dictchannelcount == 1) and (listchannelcount == 2):
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = [value, value]
                        i += 1
            elif (dictchannelcount == 1) and (listchannelcount == 4):
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = [value, value, value, 1.0]
                        i += 1
            elif (dictchannelcount == 3) and (listchannelcount == 4):
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, [0.0, 0.0, 0.0])
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = [value[0], value[1], value[2], 1.0]
                        i += 1
        else:
            if listchannelcount == 1:
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[i] = vert_dict.get(vert_idx, 0.0)
                        i += 1
            else:
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = vert_dict.get(vert_idx, [0.0] * listchannelcount)
                        i += 1

        return loop_list


    def vertex_data_dict(self, obj, masklayer=None):
        mesh = obj.data
        mat = obj.matrix_world
        ids = self.vertex_id_list(obj)

        vertex_dict = {}
        if masklayer is not None:
            mask, empty = layers.get_layer_mask(obj, masklayer)
            if not empty:
                i = 0
                for poly in mesh.polygons:
                    for vert_id, loop_idx in zip(poly.vertices, poly.loop_indices):
                        if mask[i] > 0.0:
                            vertex_dict[vert_id] = (mesh.vertices[vert_id].co, mesh.vertices[vert_id].normal, mat @ mesh.vertices[vert_id].co, (mat @ mesh.vertices[vert_id].normal - mat @ Vector()).normalized())
                        i += 1
        elif sxglobals.mode == 'EDIT':
            vert_sel = [None] * len(mesh.vertices)
            mesh.vertices.foreach_get('select', vert_sel)
            if True in vert_sel:
                for vert_id, sel in enumerate(vert_sel):
                    if sel:
                        vertex_dict[vert_id] = (mesh.vertices[vert_id].co, mesh.vertices[vert_id].normal, mat @ mesh.vertices[vert_id].co, (mat @ mesh.vertices[vert_id].normal - mat @ Vector()).normalized())
        else:
            for vert_id in ids:
                vertex_dict[vert_id] = (mesh.vertices[vert_id].co, mesh.vertices[vert_id].normal, mat @ mesh.vertices[vert_id].co, (mat @ mesh.vertices[vert_id].normal - mat @ Vector()).normalized())

        return vertex_dict


    def selection_mask(self, obj):
        mesh = obj.data
        mask = self.empty_list(obj, 1)

        pre_check = [None] * len(mesh.vertices)
        mesh.vertices.foreach_get('select', pre_check)
        if True not in pre_check:
            return mask, True
        else:
            i = 0
            if bpy.context.tool_settings.mesh_select_mode[2]:
                for poly in mesh.polygons:
                    sel = float(poly.select)
                    for loop_idx in poly.loop_indices:
                        mask[i] = sel
                        i += 1
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        mask[i] = float(mesh.vertices[vert_idx].select)
                        i += 1

            return mask, False


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
        alpha_mats = {'OCC': 'Occlusion', 'MET': 'Metallic', 'RGH': 'Roughness', 'TRN': 'Transmission'}
        layer_dict = {}
        if len(objs) > 0:
            for obj in objs:
                item = obj.sx2layers.add()
                item.name = 'Layer ' + str(obj.sx2.layercount) if name is None else name
                if layer_type not in alpha_mats:
                    item.color_attribute = color_attribute if color_attribute is not None else item.name
                else:
                    item.color_attribute = 'Alpha Materials'
                item.layer_type = 'COLOR' if layer_type is None else layer_type
                item.default_color = sxglobals.default_colors[item.layer_type]
                if layer_type in alpha_mats:
                    item.paletted = False

                if layer_type not in alpha_mats:
                    if color_attribute not in obj.data.color_attributes.keys():
                        obj.data.color_attributes.new(name=item.name, type='FLOAT_COLOR', domain='CORNER')
                elif ('Alpha Materials' not in obj.data.color_attributes.keys()) and (layer_type in alpha_mats):
                    obj.data.color_attributes.new(name='Alpha Materials', type='FLOAT_COLOR', domain='CORNER')
                    
                item.index = utils.insert_layer_at_index(obj, item, obj.sx2layers[obj.sx2.selectedlayer].index + 1)

                if clear:
                    colors = generate.color_list(obj, item.default_color)
                    layers.set_layer(obj, colors, obj.sx2layers[len(obj.sx2layers) - 1])
                layer_dict[obj] = item

            # Second loop needed for proper operation in multi-object cases
            # selectedlayer needs to point to an existing layer on all objs
            for obj in objs:
                obj.sx2.selectedlayer = len(obj.sx2layers) - 1
                obj.sx2.layercount += 1

        return layer_dict


    def del_layer(self, objs, layer_to_delete):
        for obj in objs:
            bottom_layer_index = obj.sx2layers[layer_to_delete.name].index - 1 if obj.sx2layers[layer_to_delete.name].index - 1 > 0 else 0
            if layer_to_delete.color_attribute == 'Alpha Materials':
                alpha_mat_count = 0
                for layer in obj.sx2layers:
                    if layer.color_attribute == 'Alpha Materials':
                        alpha_mat_count += 1
                if alpha_mat_count == 1:
                    obj.data.attributes.remove(obj.data.attributes[layer_to_delete.color_attribute])
            else:
                if layer_to_delete.color_attribute in obj.data.color_attributes.keys():
                    obj.data.color_attributes.remove(obj.data.color_attributes[layer_to_delete.color_attribute])
            idx = utils.find_layer_index_by_name(obj, layer_to_delete.name)
            obj.sx2layers.remove(idx)

        for obj in objs:
            utils.sort_stack_indices(obj)
            for i, layer in enumerate(obj.sx2layers):
                if layer.index == bottom_layer_index:
                    obj.sx2.selectedlayer = i


    # wrapper for low-level functions, always returns layerdata in RGBA
    def get_layer(self, obj, sourcelayer, as_tuple=False, single_as_alpha=False, apply_layer_alpha=False, gradient_with_palette=False):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
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

        elif sourcelayer.layer_type == 'UV':
            uvs = self.get_uvs(obj, sourcelayer.uvLayer0, channel=sourcelayer.uvChannel0)
            count = len(uvs)  # len(obj.data.uv_layers[0].data)
            values = [None] * count * 4

            # if gradient_with_palette:
            #     if sourcelayer.name == 'Gradient 1':
            #         dv = sxmaterial.nodes['PaletteColor3'].outputs[0].default_value
            #     elif sourcelayer.name == 'Gradient 2':
            #         dv = sxmaterial.nodes['PaletteColor4'].outputs[0].default_value

            if single_as_alpha:
                for i in range(count):
                    if uvs[i] > 0.0:
                        values[(0+i*4):(4+i*4)] = [dv[0], dv[1], dv[2], uvs[i]]
                    else:
                        values[(0+i*4):(4+i*4)] = [0.0, 0.0, 0.0, uvs[i]]
            else:
                for i in range(count):
                    values[(0+i*4):(4+i*4)] = [uvs[i], uvs[i], uvs[i], 1.0]

        elif sourcelayer.layer_type == 'UV4':
            values = layers.get_uv4(obj, sourcelayer)

        if apply_layer_alpha and sourcelayer.opacity != 1.0:
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


    # takes RGBA buffers, converts and writes to appropriate uv and vertex sets
    def set_layer(self, obj, colors, targetlayer):
        def constant_alpha_test(values):
            for value in values:
                if value != 1.0:
                    return False
            return True

        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
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

        elif target_type == 'UV':
            if (targetlayer.name == 'Gradient 1') or (targetlayer.name == 'Gradient 2'):
                target_uvs = colors[3::4]
                if constant_alpha_test(target_uvs):
                    target_uvs = layers.get_luminances(obj, sourcelayer=None, colors=colors, as_rgba=False)
            else:
                target_uvs = layers.get_luminances(obj, sourcelayer=None, colors=colors, as_rgba=False)

            layers.set_uvs(obj, targetlayer.uvLayer0, target_uvs, targetlayer.uvChannel0)

        elif target_type == 'UV4':
            layers.set_uv4(obj, targetlayer, colors)


    def get_layer_mask(self, obj, sourcelayer):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        layer_type = sourcelayer.layer_type

        if layer_type in rgba_targets:
            colors = self.get_colors(obj, sourcelayer.color_attribute)
            values = colors[3::4]
        elif layer_type in alpha_targets:
            colors = self.get_colors(obj, sourcelayer.color_attribute)
            values = colors[alpha_targets[layer_type]::4]
        elif layer_type == 'UV':
            values = self.get_uvs(obj, sourcelayer.uvLayer0, sourcelayer.uvChannel0)
        elif layer_type == 'UV4':
            values = self.get_uvs(obj, sourcelayer.uvLayer3, sourcelayer.uvChannel3)

        if any(v != 0.0 for v in values):
            return values, False
        else:
            return values, True


    def get_colors(self, obj, source):
        source_colors = obj.data.color_attributes[source].data
        colors = [None] * len(source_colors) * 4
        source_colors.foreach_get('color', colors)
        return colors


    def set_colors(self, obj, target, colors):
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()


    def get_luminances(self, obj, sourcelayer=None, colors=None, as_rgba=False, as_alpha=False):
        if colors is None:
            colors = self.get_layer(obj, sourcelayer)

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
        print('source:', source)
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


    def get_uv4(self, obj, sourcelayer):
        channels = {'U': 0, 'V': 1}
        sourceUVs0 = obj.data.uv_layers[sourcelayer.uvLayer0].data
        sourceUVs1 = obj.data.uv_layers[sourcelayer.uvLayer2].data
        count = len(sourceUVs0)
        source_uvs0 = [None] * count * 2
        source_uvs1 = [None] * count * 2
        sourceUVs0.foreach_get('uv', source_uvs0)
        sourceUVs1.foreach_get('uv', source_uvs1)

        uv0 = channels[sourcelayer.uvChannel0]
        uv1 = channels[sourcelayer.uvChannel1]
        uv2 = channels[sourcelayer.uvChannel2]
        uv3 = channels[sourcelayer.uvChannel3]

        colors = [None] * count * 4
        for i in range(count):
            colors[(0+i*4):(4+i*4)] = [source_uvs0[uv0+i*2], source_uvs0[uv1+i*2], source_uvs1[uv2+i*2], source_uvs1[uv3+i*2]]

        return colors


    def set_uv4(self, obj, targetlayer, colors):
        channels = {'U': 0, 'V': 1}

        uvs0 = generate.empty_list(obj, 2)
        uvs1 = generate.empty_list(obj, 2)

        uv0 = channels[targetlayer.uvChannel0]
        uv1 = channels[targetlayer.uvChannel1]
        uv2 = channels[targetlayer.uvChannel2]
        uv3 = channels[targetlayer.uvChannel3]

        target1 = targetlayer.uvLayer0
        target2 = targetlayer.uvLayer2

        count = len(uvs0)//2
        for i in range(count):
            [uvs0[(uv0+i*2)], uvs0[(uv1+i*2)], uvs1[(uv2+i*2)], uvs1[(uv3+i*2)]] = colors[(0+i*4):(4+i*4)]

        self.set_uvs(obj, target1, uvs0, None)
        self.set_uvs(obj, target2, uvs1, None)


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
                mask, empty = generate.selection_mask(obj)
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


    def blend_layers(self, objs, topLayerArray, baseLayer, resultLayer, single_as_alpha=False):
        active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = objs[0]

        for obj in objs:
            basecolors = self.get_layer(obj, baseLayer, single_as_alpha=single_as_alpha)

            for layer in topLayerArray:
                if getattr(obj.sx2layers[layer.name], 'visibility'):
                    blendmode = getattr(obj.sx2layers[layer.name], 'blend_mode')
                    layeralpha = getattr(obj.sx2layers[layer.name], 'opacity')
                    topcolors = self.get_layer(obj, obj.sx2layers[layer.name], single_as_alpha=single_as_alpha, gradient_with_palette=True)
                    basecolors = tools.blend_values(topcolors, basecolors, blendmode, layeralpha)

            self.set_layer(obj, basecolors, resultLayer)
        bpy.context.view_layer.objects.active = active


    def linear_alphas(self, objs):
        def value_to_linear(value):
            value = value[0]
            if value < 0.0:
                output = 0.0
            elif 0.0 <= value <= 0.0404482362771082:
                output = float(value) / 12.92
            elif 0.0404482362771082 < value <= 1.0:
                output = ((value + 0.055) / 1.055) ** 2.4
            else:
                output = 1.0
            return [output]

        for obj in objs:
            for layer in obj.sx2layers:
                values = self.get_colors(obj, layer.color_attribute)
                for i in range(len(values)//4):
                    values[(3+i*4):(4+i*4)] = value_to_linear(values[(3+i*4):(4+i*4)])
                self.set_colors(obj, layer.color_attribute, values)
            # obj.data.update()


    def merge_layers(self, objs, toplayer, baselayer, targetlayer):
        for obj in objs:
            basecolors = self.get_layer(obj, baselayer, apply_layer_alpha=True, single_as_alpha=True)
            topcolors = self.get_layer(obj, toplayer, apply_layer_alpha=True, single_as_alpha=True)

            blendmode = toplayer.blend_mode
            colors = tools.combine_layers(topcolors, basecolors, blendmode)
            self.set_layer(obj, colors, targetlayer)

        for obj in objs:
            setattr(obj.sx2layers[targetlayer.name], 'visibility', True)
            setattr(obj.sx2layers[targetlayer.name], 'blend_mode', 'ALPHA')
            setattr(obj.sx2layers[targetlayer.name], 'opacity', 1.0)
            obj.sx2.selectedlayer = utils.find_layer_index_by_name(obj, targetlayer.name)

        if toplayer.name == targetlayer.name:
            self.clear_layers(objs, baselayer)
        else:
            self.clear_layers(objs, toplayer)


    def paste_layer(self, objs, targetlayer, fillmode):
        utils.mode_manager(objs, set_mode=True, mode_id='paste_layer')

        if fillmode == 'mask':
            for obj in objs:
                colors = layers.get_layer(obj, targetlayer)
                alphas = sxglobals.copy_buffer[obj.name]
                count = len(colors)//4
                for i in range(count):
                    colors[(3+i*4):(4+i*4)] = alphas[(3+i*4):(4+i*4)]
                layers.set_layer(obj, colors, targetlayer)
        else:
            for obj in objs:
                colors = sxglobals.copy_buffer[obj.name]
                targetvalues = self.get_layer(obj, targetlayer)
                if sxglobals.mode == 'EDIT':
                    colors = generate.mask_list(obj, colors)
                colors = tools.blend_values(colors, targetvalues, 'ALPHA', 1.0)
                layers.set_layer(obj, colors, targetlayer)

        utils.mode_manager(objs, revert=True, mode_id='paste_layer')


    def color_layers_to_values(self, objs):
        scene = bpy.context.scene.sx2

        for obj in objs:
            for i in range(5):
                for layer in obj.sx2layers:
                    if layer.paletted and (layer.palette_index == i):
                        palettecolor = utils.find_colors_by_frequency(objs, layer, 1, obj_sel_override=True)[0]
                        tabcolor = getattr(scene, 'newpalette' + str(i))

                        if not utils.color_compare(palettecolor, tabcolor) and not utils.color_compare(palettecolor, (0.0, 0.0, 0.0, 1.0)):
                            setattr(scene, 'newpalette' + str(i), palettecolor)


    def material_layers_to_values(self, objs):
        scene = bpy.context.scene.sx2
        layers = [objs[0].sx2.selectedlayer, 12, 13]

        for i, idx in enumerate(layers):
            layer = objs[0].sx2layers[idx]
            if sxglobals.mode == 'EDIT':
                masklayer = None
            else:
                masklayer = utils.find_layer_by_stack_index(objs[0], layers[0])
            layercolors = utils.find_colors_by_frequency(objs, layer, 8, masklayer=masklayer)
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


    def blend_values(self, topcolors, basecolors, blendmode, blendvalue):
        if topcolors is None:
            return basecolors
        else:
            count = len(basecolors)//4
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

                elif blendmode == 'MUL':
                    for j in range(3):
                        base[j] *= top[j] * a + (1 - a)

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

                if base[3] == 0.0:
                    base = [0.0, 0.0, 0.0, 0.0]

                colors[(0+i*4):(4+i*4)] = base
            return colors


    def combine_layers(self, topcolors, basecolors, blendmode):
        count = len(basecolors)//4
        colors = [None] * count * 4
        midpoint = 0.5  # convert.srgb_to_linear([0.5, 0.5, 0.5, 1.0])[0]

        for i in range(count):
            top_rgb = Vector(topcolors[(0+i*4):(3+i*4)])
            top_alpha = topcolors[3+i*4]
            base_rgb = Vector(basecolors[(0+i*4):(3+i*4)])
            base_alpha = basecolors[3+i*4]
            result_rgb = [0.0, 0.0, 0.0]
            result_alpha = min((base_alpha + top_alpha), 1.0)

            if result_alpha > 0.0:
                if blendmode == 'ALPHA':
                    result_rgb = (top_rgb * top_alpha + base_rgb * (1.0 - top_alpha))

                elif blendmode == 'ADD':
                    result_rgb = base_rgb + top_rgb

                elif blendmode == 'MUL':
                    for j in range(3):
                        result_rgb[j] = base_rgb[j] * (top_rgb[j] * top_alpha + (1.0 - top_alpha))

                elif blendmode == 'OVR':
                    over = Vector([0.0, 0.0, 0.0])
                    over_alpha = top_alpha
                    for j in range(3):
                        if base_rgb[j] < midpoint:
                            over[j] = 2.0 * base_rgb[j] * top_rgb[j]
                        else:
                            over[j] = 1.0 - 2.0 * (1.0 - base_rgb[j]) * (1.0 - top_rgb[j])
                    result_rgb = over * over_alpha + base_rgb * (1.0 - over_alpha)

            # print('blendmode: ', blendmode)
            # print('top:       ', top_rgb[0], top_rgb[1], top_rgb[2], top_alpha)
            # print('base:      ', base_rgb[0], base_rgb[1], base_rgb[2], base_alpha)
            # print('result:    ', result_rgb[0], result_rgb[1], result_rgb[2], result_alpha, '\n')

            colors[(0+i*4):(4+i*4)] = [result_rgb[0], result_rgb[1], result_rgb[2], result_alpha]
        return colors


    # NOTE: Make sure color parameter is always provided in linear color space!
    def apply_tool(self, objs, targetlayer, masklayer=None, invertmask=False, color=None):
        # then = time.perf_counter()
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
            if (masklayer is None) and (targetlayer.locked) and (sxglobals.mode == 'OBJECT'):
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
                colors = generate.curvature_list(obj, masklayer)
            elif scene.toolmode == 'OCC':
                colors = generate.occlusion_list(obj, scene.occlusionrays, scene.occlusionblend, scene.occlusiondistance, scene.occlusiongroundplane, masklayer)
            elif scene.toolmode == 'THK':
                colors = generate.thickness_list(obj, scene.occlusionrays, masklayer)
            elif scene.toolmode == 'DIR':
                colors = generate.direction_list(obj, masklayer)
            elif scene.toolmode == 'LUM':
                colors = generate.luminance_remap_list(obj, targetlayer, masklayer)

            if colors is not None:
                # Write to target
                target_colors = layers.get_layer(obj, targetlayer)
                colors = self.blend_values(colors, target_colors, blendmode, blendvalue)
                layers.set_layer(obj, colors, targetlayer)

            # obj.data.update()
        utils.mode_manager(objs, revert=True, mode_id='apply_tool')
        # now = time.perf_counter()
        # print('Apply tool ', scene.toolmode, ' duration: ', now-then, ' seconds')


    # mode 0: hue, mode 1: saturation, mode 2: lightness
    def apply_hsl(self, objs, layer, hslmode, newValue):
        utils.mode_manager(objs, set_mode=True, mode_id='apply_hsl')

        colors = utils.find_colors_by_frequency(objs, layer)
        if len(colors) > 0:
            valueArray = []
            for color in colors:
                hsl = convert.rgb_to_hsl(color)
                valueArray.append(hsl[hslmode])
            offset = newValue - max(valueArray)
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

        utils.mode_manager(objs, revert=True, mode_id='apply_hsl')


    def preview_palette(self, objs, palette):
        utils.mode_manager(objs, set_mode=True, mode_id='apply_palette')

        scene = bpy.context.scene
        palette = [
            scene.sx2palettes[palette].color0,
            scene.sx2palettes[palette].color1,
            scene.sx2palettes[palette].color2,
            scene.sx2palettes[palette].color3,
            scene.sx2palettes[palette].color4]

        for i in range(5):
            setattr(scene.sx2, 'newpalette'+str(i), palette[i])

        for obj in objs:
            obj.sx2.palettedshading = 1.0

        utils.mode_manager(objs, revert=True, mode_id='apply_palette')


    def apply_palette(self, objs, palette):
        utils.mode_manager(objs, set_mode=True, mode_id='apply_palette')
        if not sxglobals.refresh_in_progress:
            sxglobals.refresh_in_progress = True

        scene = bpy.context.scene.sx2
        palette = [
            scene.newpalette0,
            scene.newpalette1,
            scene.newpalette2,
            scene.newpalette3,
            scene.newpalette4]

        for obj in objs:
            if len(obj.sx2layers) > 0:
                for layer in obj.sx2layers:
                    if layer.paletted:
                        source_colors = layers.get_layer(obj, layer)
                        colors = generate.color_list(obj, palette[layer.palette_index], masklayer=layer)
                        if colors is not None:
                            values = tools.blend_values(colors, source_colors, 'ALPHA', obj.sx2.palettedshading)
                            layers.set_layer(obj, values, layer)

        sxglobals.refresh_in_progress = False
        utils.mode_manager(objs, revert=True, mode_id='apply_palette')


    # NOTE: Material library file contains all entries in srgb color change by default
    def apply_material(self, objs, targetlayer, material):
        utils.mode_manager(objs, set_mode=True, mode_id='apply_material')
        if not sxglobals.refresh_in_progress:
            sxglobals.refresh_in_progress = True

        material = bpy.context.scene.sx2materials[material]
        scene = bpy.context.scene.sx2

        # material library contains smoothness values, Blender uses roughness
        invert_color = [None, None, None, 1.0]
        for i in range(3):
            invert_color[i] = 1 - material.color2[i]

        # for obj in objs:
        #     scene.toolopacity = 1.0
        #     scene.toolblend = 'ALPHA'
        #     self.apply_tool([obj, ], targetlayer, color=material.color0)

        #     if sxglobals.mode == 'EDIT':
        #         self.apply_tool([obj, ], obj.sx2layers['Metallic'], color=material.color1)
        #         self.apply_tool([obj, ], obj.sx2layers['Roughness'], color=material.color2)
        #     else:
        #         self.apply_tool([obj, ], obj.sx2layers['Metallic'], masklayer=targetlayer, color=material.color1)
        #         self.apply_tool([obj, ], obj.sx2layers['Roughness'], masklayer=targetlayer, color=material.color2)

        setattr(scene, 'newmaterial0', material.color0)
        setattr(scene, 'newmaterial1', material.color1)
        setattr(scene, 'newmaterial2', invert_color)

        sxglobals.refresh_in_progress = False
        utils.mode_manager(objs, revert=True, mode_id='apply_material')


    def select_color_mask(self, objs, color, invertmask=False):
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        for obj in objs:
            colors = layers.get_layer(obj, obj.sx2layers['composite'])
            mesh = obj.data
            i = 0
            if bpy.context.tool_settings.mesh_select_mode[2]:
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        if invertmask:
                            if not utils.color_compare(colors[(0+i*4):(4+i*4)], color):
                                poly.select = True
                        else:
                            if utils.color_compare(colors[(0+i*4):(4+i*4)], color):
                                poly.select = True
                        i += 1
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        if invertmask:
                            if not utils.color_compare(colors[(0+i*4):(4+i*4)], color):
                                mesh.vertices[vert_idx].select = True
                        else:
                            if utils.color_compare(colors[(0+i*4):(4+i*4)], color):
                                mesh.vertices[vert_idx].select = True
                        i += 1

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)


    def select_mask(self, objs, layer, invertmask=False):
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        for obj in objs:
            mesh = obj.data
            mask = (layers.get_layer_mask(obj, layer))[0]

            i = 0
            if bpy.context.tool_settings.mesh_select_mode[2]:
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        if invertmask:
                            if mask[i] == 0.0:
                                poly.select = True
                        else:
                            if mask[i] > 0.0:
                                poly.select = True
                        i += 1
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        if invertmask:
                            if mask[i] == 0.0:
                                mesh.vertices[vert_idx].select = True
                        else:
                            if mask[i] > 0.0:
                                mesh.vertices[vert_idx].select = True
                        i += 1

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)


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
        colorArray = [
            color, palCols[0], palCols[1], palCols[2],
            palCols[3], palCols[4], palCols[5], palCols[6]]

        fillColor = color[:]
        if (fillColor not in palCols) and (fillColor[3] > 0.0):
            for i in range(8):
                setattr(scene, 'fillpalette' + str(i + 1), colorArray[i])


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
        if (setmode == 'CRS') and (weight == -1.0):
            weight = 0.0 

        utils.mode_manager(objs, set_mode=True, mode_id='assign_set')
        active = bpy.context.view_layer.objects.active

        # Create necessary data layers
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            mesh = obj.data
            if not mesh.has_crease_vertex:
                bpy.ops.mesh.customdata_crease_vertex_add()
            if not mesh.has_crease_edge:
                bpy.ops.mesh.customdata_crease_edge_add()
            if not mesh.has_bevel_weight_edge:
                bpy.ops.mesh.customdata_bevel_weight_edge_add()
            if not mesh.has_bevel_weight_vertex:
                bpy.ops.mesh.customdata_bevel_weight_vertex_add()

            if (mode == 'EDIT'):
                # EDIT mode vertex creasing
                # vertex beveling not supported
                if (bpy.context.tool_settings.mesh_select_mode[0]):
                    weight_values = [None] * len(mesh.vertices)
                    select_values = [None] * len(mesh.vertices)
                    mesh.vertices.foreach_get('select', select_values)
                    if setmode == 'CRS':
                        mesh.vertex_creases[0].data.foreach_get('value', weight_values)
                        for i, sel in enumerate(select_values):
                            if sel:
                                weight_values[i] = weight

                    mesh.vertex_creases[0].data.foreach_set('value', weight_values)

                # EDIT mode edge creasing and beveling
                elif (bpy.context.tool_settings.mesh_select_mode[1] or bpy.context.tool_settings.mesh_select_mode[2]):
                    weight_values = [None] * len(mesh.edges)
                    sharp_values = [None] * len(mesh.edges)
                    select_values = [None] * len(mesh.edges)
                    mesh.edges.foreach_get(mode_dict[setmode], weight_values)
                    mesh.edges.foreach_get('use_edge_sharp', sharp_values)
                    mesh.edges.foreach_get('select', select_values)

                    for i, sel in enumerate(select_values):
                        if sel:
                            weight_values[i] = weight
                            if setmode == 'CRS':
                                if weight == 1.0:
                                    sharp_values[i] = True
                                else:
                                    sharp_values[i] = False

                    mesh.edges.foreach_set(mode_dict[setmode], weight_values)
                    mesh.edges.foreach_set('use_edge_sharp', sharp_values)

            # OBJECT mode
            else:
                # vertex creasing
                if (bpy.context.tool_settings.mesh_select_mode[0]):
                    if setmode == 'CRS':
                        weight_values = [weight] * len(mesh.vertices)
                        mesh.vertex_creases[0].data.foreach_set('value', weight_values)

                # edge creasing and beveling
                else:
                    if setmode == 'CRS':
                        if weight == 1.0:
                            sharp_values = [True] * len(mesh.edges)
                        else:
                            sharp_values = [False] * len(mesh.edges)
                        mesh.edges.foreach_set('use_edge_sharp', sharp_values)

                    weight_values = [weight] * len(mesh.edges)
                    mesh.edges.foreach_set(mode_dict[setmode], weight_values)

            mesh.update()
        utils.mode_manager(objs, revert=True, mode_id='assign_set')
        bpy.context.view_layer.objects.active = active


    def select_set(self, objs, setvalue, setmode, clearsel=False):
        modeDict = {
            'CRS': 'SubSurfCrease',
            'BEV': 'BevelWeight'}
        weight = setvalue
        modename = modeDict[setmode]

        bpy.context.view_layer.objects.active = objs[0]

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        if bpy.context.tool_settings.mesh_select_mode[0]:
            bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
        else:
            bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
        if clearsel:
            bpy.ops.mesh.select_all(action='DESELECT')

        for obj in objs:
            mesh = obj.data

            if bpy.context.tool_settings.mesh_select_mode[0]:
                utils.mode_manager(objs, set_mode=True, mode_id='select_set')
                for vert in mesh.vertices:
                    creaseweight = mesh.vertex_creases[0].data[vert.index].value
                    if math.isclose(creaseweight, weight, abs_tol=0.1):
                        vert.select = True
                utils.mode_manager(objs, revert=True, mode_id='select_set')
            else:
                bm = bmesh.from_edit_mesh(mesh)

                if setmode == 'CRS':
                    if modename in bm.edges.layers.crease:
                        bmlayer = bm.edges.layers.crease[modename]
                    else:
                        bmlayer = bm.edges.layers.crease.new(modename)
                else:
                    if modename in bm.edges.layers.bevel_weight:
                        bmlayer = bm.edges.layers.bevel_weight[modename]
                    else:
                        bmlayer = bm.edges.layers.bevel_weight.new(modename)

                for edge in bm.edges:
                    if math.isclose(edge[bmlayer], weight, abs_tol=0.1):
                        edge.select = True

                bmesh.update_edit_mesh(mesh)
                bm.free()


    def add_modifiers(self, objs):
        for obj in objs:
            obj.data.use_auto_smooth = True
            if 'sxMirror' not in obj.modifiers:
                obj.modifiers.new(type='MIRROR', name='sxMirror')
                if obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror:
                    obj.modifiers['sxMirror'].show_viewport = True
                else:
                    obj.modifiers['sxMirror'].show_viewport = False
                obj.modifiers['sxMirror'].show_expanded = False
                obj.modifiers['sxMirror'].use_axis[0] = obj.sx2.xmirror
                obj.modifiers['sxMirror'].use_axis[1] = obj.sx2.ymirror
                obj.modifiers['sxMirror'].use_axis[2] = obj.sx2.zmirror
                if obj.sx2.mirrorobject is not None:
                    obj.modifiers['sxMirror'].mirror_object = obj.sx2.mirrorobject
                else:
                    obj.modifiers['sxMirror'].mirror_object = None
                obj.modifiers['sxMirror'].use_clip = True
                obj.modifiers['sxMirror'].use_mirror_merge = True
            if 'sxTiler' not in obj.modifiers:
                if 'sx_tiler' not in bpy.data.node_groups:
                    setup.create_tiler()
                tiler = obj.modifiers.new(type='NODES', name='sxTiler')
                tiler.node_group = bpy.data.node_groups['sx_tiler']
                tiler['Input_1'] = obj.sx2.tile_offset
                tiler['Input_2'] = obj.sx2.tile_neg_x
                tiler['Input_3'] = obj.sx2.tile_pos_x
                tiler['Input_4'] = obj.sx2.tile_neg_y
                tiler['Input_5'] = obj.sx2.tile_pos_y
                tiler['Input_6'] = obj.sx2.tile_neg_z
                tiler['Input_7'] = obj.sx2.tile_pos_z
                tiler.show_viewport = False
                tiler.show_expanded = False
                obj.data.use_auto_smooth = True
            if 'sxSubdivision' not in obj.modifiers:
                obj.modifiers.new(type='SUBSURF', name='sxSubdivision')
                obj.modifiers['sxSubdivision'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxSubdivision'].show_expanded = False
                obj.modifiers['sxSubdivision'].quality = 6
                obj.modifiers['sxSubdivision'].levels = obj.sx2.subdivisionlevel
                obj.modifiers['sxSubdivision'].uv_smooth = 'NONE'
                obj.modifiers['sxSubdivision'].show_only_control_edges = True
                obj.modifiers['sxSubdivision'].show_on_cage = True
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
            if 'sxWeld' not in obj.modifiers:
                obj.modifiers.new(type='WELD', name='sxWeld')
                if obj.sx2.weldthreshold == 0:
                    obj.modifiers['sxWeld'].show_viewport = False
                else:
                    obj.modifiers['sxWeld'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxWeld'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxWeld'].show_expanded = False
                obj.modifiers['sxWeld'].merge_threshold = obj.sx2.weldthreshold
            if 'sxDecimate' not in obj.modifiers:
                obj.modifiers.new(type='DECIMATE', name='sxDecimate')
                if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                    obj.modifiers['sxDecimate'].show_viewport = False
                else:
                    obj.modifiers['sxDecimate'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxDecimate'].show_expanded = False
                obj.modifiers['sxDecimate'].decimate_type = 'DISSOLVE'
                obj.modifiers['sxDecimate'].angle_limit = math.radians(obj.sx2.decimation)
                obj.modifiers['sxDecimate'].use_dissolve_boundaries = True
                obj.modifiers['sxDecimate'].delimit = {'SHARP', 'UV'}
            if 'sxDecimate2' not in obj.modifiers:
                obj.modifiers.new(type='DECIMATE', name='sxDecimate2')
                if (obj.sx2.subdivisionlevel == 0) or (obj.sx2.decimation == 0.0):
                    obj.modifiers['sxDecimate2'].show_viewport = False
                else:
                    obj.modifiers['sxDecimate2'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxDecimate2'].show_expanded = False
                obj.modifiers['sxDecimate2'].decimate_type = 'COLLAPSE'
                obj.modifiers['sxDecimate2'].ratio = 0.99
                obj.modifiers['sxDecimate2'].use_collapse_triangulate = True
            if 'sxWeightedNormal' not in obj.modifiers:
                obj.modifiers.new(type='WEIGHTED_NORMAL', name='sxWeightedNormal')
                if not obj.sx2.weightednormals:
                    obj.modifiers['sxWeightedNormal'].show_viewport = False
                else:
                    obj.modifiers['sxWeightedNormal'].show_viewport = obj.sx2.modifiervisibility
                obj.modifiers['sxWeightedNormal'].show_expanded = False
                obj.modifiers['sxWeightedNormal'].mode = 'FACE_AREA_WITH_ANGLE'
                obj.modifiers['sxWeightedNormal'].weight = 50
                if obj.sx2.hardmode == 'SMOOTH':
                    obj.modifiers['sxWeightedNormal'].keep_sharp = False
                else:
                    obj.modifiers['sxWeightedNormal'].keep_sharp = True

            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(obj.sx2.smoothangle)


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
            if 'sxWeightedNormal' in obj.modifiers:
                if not obj.sx2.weightednormals:
                    bpy.ops.object.modifier_remove(modifier='sxWeightedNormal')
                else:
                    bpy.ops.object.modifier_apply(modifier='sxWeightedNormal')


    def remove_modifiers(self, objs, reset=False):
        modifiers = ['sxMirror', 'sxTiler', 'sxGeometryNodes', 'sxSubdivision', 'sxBevel', 'sxWeld', 'sxDecimate', 'sxDecimate2', 'sxEdgeSplit', 'sxWeightedNormal']
        if reset and ('sx_tiler' in bpy.data.node_groups):
            bpy.data.node_groups.remove(bpy.data.node_groups['sx_tiler'], do_unlink=True)

        for obj in objs:
            for modifier in modifiers:
                if modifier in obj.modifiers:
                    obj.modifiers.remove(obj.modifiers.get(modifier))

            obj.data.use_auto_smooth = True

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
        for obj in objs:
            if 'sxDecimate2' in obj.modifiers:
                count += obj.modifiers['sxDecimate2'].face_count

        return str(count)


    def __del__(self):
        print('SX Tools: Exiting modifiers')


# ------------------------------------------------------------------------
#    Export Module
# ------------------------------------------------------------------------
class SXTOOLS2_export(object):
    def __init__(self):
        return None


    def smart_separate(self, objs):
        if len(objs) > 0:
            mirror_pairs = [('_top', '_bottom'), ('_front', '_rear'), ('_left', '_right'), ('_bottom', '_top'), ('_rear', '_front'), ('_right', '_left')]
            prefs = bpy.context.preferences.addons['sxtools2'].preferences
            scene = bpy.context.scene.sx2
            view_layer = bpy.context.view_layer
            mode = objs[0].mode
            objs = objs[:]

            sepObjs = []
            for obj in objs:
                if obj.sx2.smartseparate:
                    if obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror:
                        sepObjs.append(obj)

            if 'ExportObjects' not in bpy.data.collections:
                exportObjects = bpy.data.collections.new('ExportObjects')
            else:
                exportObjects = bpy.data.collections['ExportObjects']

            if 'SourceObjects' not in bpy.data.collections:
                sourceObjects = bpy.data.collections.new('SourceObjects')
            else:
                sourceObjects = bpy.data.collections['SourceObjects']

            if len(sepObjs) > 0:
                for obj in sepObjs:
                    if (scene.exportquality == 'LO') and (obj.name not in sourceObjects.objects.keys()) and (obj.name not in exportObjects.objects.keys()) and (obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror):
                        sourceObjects.objects.link(obj)

            separatedObjs = []
            if len(sepObjs) > 0:
                active = view_layer.objects.active
                bpy.ops.object.select_all(action='DESELECT')
                for obj in sepObjs:
                    obj.select_set(True)
                    view_layer.objects.active = obj
                    refObjs = view_layer.objects[:]
                    orgname = obj.name[:]
                    xmirror = obj.sx2.xmirror
                    ymirror = obj.sx2.ymirror
                    zmirror = obj.sx2.zmirror

                    if obj.modifiers['sxMirror'].mirror_object is not None:
                        refLoc = obj.modifiers['sxMirror'].mirror_object.matrix_world.to_translation()
                    else:
                        refLoc = obj.matrix_world.to_translation()

                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    bpy.ops.object.modifier_apply(modifier='sxMirror')

                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.mesh.separate(type='LOOSE')

                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    newObjArray = [view_layer.objects[orgname], ]

                    for vlObj in view_layer.objects:
                        if vlObj not in refObjs:
                            newObjArray.append(vlObj)
                            exportObjects.objects.link(vlObj)

                    if len(newObjArray) > 1:
                        export.set_pivots(newObjArray)
                        suffixDict = {}
                        for newObj in newObjArray:
                            view_layer.objects.active = newObj
                            zstring = ystring = xstring = ''
                            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([newObj, ])

                            # 1) compare bounding box max to reference pivot
                            # 2) flip result with xor according to SX Tools prefs
                            # 3) look up matching sub-string from mirror_pairs
                            if zmirror:
                                zstring = mirror_pairs[0][int(zmax < refLoc[2])]
                            if ymirror:
                                ystring = mirror_pairs[1][int((ymax < refLoc[1]) ^ prefs.flipsmarty)]
                            if xmirror:
                                xstring = mirror_pairs[2][int((xmax < refLoc[0]) ^ prefs.flipsmartx)]

                            if len(newObjArray) > 2 ** (zmirror + ymirror + xmirror):
                                if zstring+ystring+xstring not in suffixDict:
                                    suffixDict[zstring+ystring+xstring] = 0
                                else:
                                    suffixDict[zstring+ystring+xstring] += 1
                                newObj.name = orgname + str(suffixDict[zstring+ystring+xstring]) + zstring + ystring + xstring
                            else:
                                newObj.name = orgname + zstring + ystring + xstring

                            newObj.data.name = newObj.name + '_mesh'

                    separatedObjs.extend(newObjArray)

                    # Parent new children to their matching parents
                    for obj in separatedObjs:
                        for mirror_pair in mirror_pairs:
                            if mirror_pair[0] in obj.name:
                                if mirror_pair[1] in obj.parent.name:
                                    new_parent_name = obj.parent.name.replace(mirror_pair[1], mirror_pair[0])
                                    obj.parent = view_layer.objects[new_parent_name]
                                    obj.matrix_parent_inverse = obj.parent.matrix_world.inverted()

                view_layer.objects.active = active
            bpy.ops.object.mode_set(mode=mode)
            return separatedObjs


    # LOD levels:
    # If subdivision enabled:
    #   LOD0 - Maximum subdivision and bevels
    #   LOD1 - Subdiv 1, bevels
    #   LOD2 - Control cage
    # If bevels only:
    #   LOD0 - Maximum bevel segments
    #   LOD1 - Control cage
    # NOTE: In case of bevels, prefer even-numbered segment counts!
    #       Odd-numbered bevels often generate incorrect vertex colors
    def generate_lods(self, objs):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        orgObjArray = objs[:]
        nameArray = []
        newObjArray = []
        activeObj = bpy.context.view_layer.objects.active
        scene = bpy.context.scene.sx2

        xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box(orgObjArray)
        bbxheight = zmax - zmin

        # Make sure source and export Collections exist
        if 'ExportObjects' not in bpy.data.collections:
            exportObjects = bpy.data.collections.new('ExportObjects')
        else:
            exportObjects = bpy.data.collections['ExportObjects']

        if 'SourceObjects' not in bpy.data.collections:
            sourceObjects = bpy.data.collections.new('SourceObjects')
        else:
            sourceObjects = bpy.data.collections['SourceObjects']

        lodCount = 1
        for obj in orgObjArray:
            nameArray.append((obj.name[:], obj.data.name[:]))
            if obj.sx2.subdivisionlevel >= 1:
                lodCount = min(3, obj.sx2.subdivisionlevel + 1)
            elif (obj.sx2.subdivisionlevel == 0) and ((obj.sx2.bevelsegments) > 0):
                lodCount = 2

        if lodCount > 1:
            for i in range(lodCount):
                print(f'SX Tools: Generating LOD {i}')
                if i == 0:
                    for obj in orgObjArray:
                        obj.data.name = obj.data.name + '_LOD' + str(i)
                        obj.name = obj.name + '_LOD' + str(i)
                        newObjArray.append(obj)
                        if scene.exportquality == 'LO':
                            sourceObjects.objects.link(obj)
                else:
                    for j, obj in enumerate(orgObjArray):
                        newObj = obj.copy()
                        newObj.data = obj.data.copy()

                        newObj.data.name = nameArray[j][1] + '_LOD' + str(i)
                        newObj.name = nameArray[j][0] + '_LOD' + str(i)

                        newObj.location += Vector((0.0, 0.0, (bbxheight+prefs.lodoffset)*i))

                        bpy.context.scene.collection.objects.link(newObj)
                        exportObjects.objects.link(newObj)

                        newObj.parent = bpy.context.view_layer.objects[obj.parent.name]

                        bpy.ops.object.select_all(action='DESELECT')
                        newObj.select_set(True)
                        bpy.context.view_layer.objects.active = newObj

                        if i == 1:
                            if obj.sx2.subdivisionlevel > 0:
                                newObj.sx2.subdivisionlevel = 1
                                if obj.sx2.bevelsegments > 0:
                                    newObj.sx2.bevelsegments = obj.sx2.bevelsegments
                                else:
                                    newObj.sx2.bevelsegments = 0
                            elif obj.sx2.subdivisionlevel == 0:
                                newObj.sx2.subdivisionlevel = 0
                                newObj.sx2.bevelsegments = 0
                                newObj.sx2.weldthreshold = 0
                        else:
                            newObj.sx2.subdivisionlevel = 0
                            newObj.sx2.bevelsegments = 0
                            newObj.sx2.weldthreshold = 0

                        newObjArray.append(newObj)

        # activeObj.select_set(True)
        bpy.context.view_layer.objects.active = activeObj

        return newObjArray


    def generate_mesh_colliders(self, objs):
        org_objs = objs[:]
        nameArray = []
        new_objs = []
        active_obj = bpy.context.view_layer.objects.active
        scene = bpy.context.scene.sx2

        if 'ExportObjects' not in bpy.data.collections:
            exportObjects = bpy.data.collections.new('ExportObjects')
        else:
            exportObjects = bpy.data.collections['ExportObjects']

        if 'SXColliders' not in bpy.data.collections:
            colliders = bpy.data.collections.new('SXColliders')
        else:
            colliders = bpy.data.collections['SXColliders']

        if 'CollisionSourceObjects' not in bpy.data.collections:
            collisionSourceObjects = bpy.data.collections.new('CollisionSourceObjects')
        else:
            collisionSourceObjects = bpy.data.collections['CollisionSourceObjects']

        if colliders.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(colliders)

        for obj in org_objs:
            nameArray.append((obj.name[:], obj.data.name[:]))

        for j, obj in enumerate(org_objs):
            newObj = obj.copy()
            newObj.data = obj.data.copy()

            newObj.data.name = nameArray[j][1] + '_collision'
            newObj.name = nameArray[j][0] + '_collision'

            bpy.context.scene.collection.objects.link(newObj)
            collisionSourceObjects.objects.link(newObj)

            newObj.parent = bpy.context.view_layer.objects[obj.parent.name]

            newObj.sx2.subdivisionlevel = scene.sourcesubdivision

            new_objs.append(newObj)

        tools.apply_modifiers(new_objs)

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

        for src_obj in collisionSourceObjects.objects:
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

        #     tools.apply_modifiers([obj, ])

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


    def export_to_linear(self, objs):
        for obj in objs:
            vcolors = layers.get_colors(obj, 'VertexColor0')
            count = len(vcolors)//4
            for i in range(count):
                vcolors[(0+i*4):(4+i*4)] = convert.srgb_to_linear(vcolors[(0+i*4):(4+i*4)])
            layers.set_colors(obj, 'VertexColor0', vcolors)


    def export_to_srgb(self, objs):
        for obj in objs:
            vcolors = layers.get_colors(obj, 'VertexColor0')
            count = len(vcolors)//4
            for i in range(count):
                vcolors[(0+i*4):(4+i*4)] = convert.linear_to_srgb(vcolors[(0+i*4):(4+i*4)])
            layers.set_colors(obj, 'VertexColor0', vcolors)


    def remove_exports(self):
        if 'ExportObjects' in bpy.data.collections:
            exportObjects = bpy.data.collections['ExportObjects'].objects
            for obj in exportObjects:
                bpy.data.objects.remove(obj, do_unlink=True)

        if 'SourceObjects' in bpy.data.collections:
            sourceObjects = bpy.data.collections['SourceObjects'].objects
            tags = sxglobals.keywords
            for obj in sourceObjects:
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
                sourceObjects.unlink(obj)



    # Generate 1-bit layer masks for color layers
    # so the faces can be re-colored in a game engine
    # Brush tools may leave low alpha values that break
    # palettemasks, alpha_tolerance can be used to fix this.
    # The default setting accepts only fully opaque values.
    def generate_palette_masks(self, objs, alpha_tolerance=1.0):
        channels = {'R': 0, 'G': 1, 'B': 2, 'A': 3, 'U': 0, 'V': 1}

        for obj in objs:
            layers = utils.find_color_layers(obj)

            for layer in layers:
                uvmap = obj.sx2.uvmap_palette_masks
                target_channel = obj.sx2.uvchannel_palette_masks
                for poly in obj.data.polygons:
                    for idx in poly.loop_indices:
                        for i, layer in enumerate(layers):
                            if i == 0:
                                obj.data.uv_layers[uvmap].data[idx].uv[channels[target_channel]] = 1.0
                            else:
                                vertex_alpha = obj.data.color_attributes[layer.color_attribute].data[idx].color[3]
                                if vertex_alpha >= alpha_tolerance:
                                    if layer.paletted:
                                        obj.data.uv_layers[uvmap].data[idx].uv[channels[target_channel]] = layer.palette_index + 1
                                    else:
                                        obj.data.uv_layers[uvmap].data[idx].uv[channels[target_channel]] = 0


    def flatten_alphas(self, objs):
        for obj in objs:
            vertexUVs = obj.data.uv_layers
            channels = {'U': 0, 'V': 1}
            for layer in obj.sx2layers:
                alpha = layer.opacity
                if (layer.name == 'Gradient 1') or (layer.name == 'Gradient 2'):
                    for poly in obj.data.polygons:
                        for idx in poly.loop_indices:
                            vertexUVs[layer.uvLayer0].data[idx].uv[channels[layer.uvChannel0]] *= alpha
                    layer.opacity = 1.0
                elif layer.name == 'Overlay':
                    if layer.blend_mode == 'OVR':
                        for poly in obj.data.polygons:
                            for idx in poly.loop_indices:
                                base = [0.5, 0.5, 0.5, 1.0]
                                top = [
                                    vertexUVs[layer.uvLayer0].data[idx].uv[channels[layer.uvChannel0]],
                                    vertexUVs[layer.uvLayer1].data[idx].uv[channels[layer.uvChannel1]],
                                    vertexUVs[layer.uvLayer2].data[idx].uv[channels[layer.uvChannel2]],
                                    vertexUVs[layer.uvLayer3].data[idx].uv[channels[layer.uvChannel3]]][:]
                                for j in range(3):
                                    base[j] = (top[j] * (top[3] * alpha) + base[j] * (1 - (top[3] * alpha)))

                                vertexUVs[layer.uvLayer0].data[idx].uv[channels[layer.uvChannel0]] = base[0]
                                vertexUVs[layer.uvLayer1].data[idx].uv[channels[layer.uvChannel1]] = base[1]
                                vertexUVs[layer.uvLayer2].data[idx].uv[channels[layer.uvChannel2]] = base[2]
                                vertexUVs[layer.uvLayer3].data[idx].uv[channels[layer.uvChannel3]] = base[3]
                    else:
                        for poly in obj.data.polygons:
                            for idx in poly.loop_indices:
                                vertexUVs[layer.uvLayer3].data[idx].uv[channels[layer.uvChannel3]] *= alpha
                    layer.opacity = 1.0


    def create_sxcollection(self):
        if 'SXObjects' not in bpy.data.collections:
            sxObjects = bpy.data.collections.new('SXObjects')
        else:
            sxObjects = bpy.data.collections['SXObjects']
        for obj in bpy.data.objects:
            if (len(obj.sx2.keys()) > 0) and (obj.name not in sxObjects.objects) and (obj.type == 'MESH'):
                sxObjects.objects.link(obj)

        if sxObjects.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(sxObjects)


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

        utils.mode_manager(objs, revert=True, mode_id='group_objects')


    # pivotmodes: 0 == no change, 1 == center of mass, 2 == center of bbox,
    # 3 == base of bbox, 4 == world origin, 5 == pivot of parent, force == set mirror axis to mirrorobj
    def set_pivots(self, objs, pivotmode=None, force=False):
        viewlayer = bpy.context.view_layer
        active = viewlayer.objects.active
        selected = viewlayer.objects.selected[:]
        modedict = {'OFF': 0, 'MASS': 1, 'BBOX': 2, 'ROOT': 3, 'ORG': 4, 'PAR': 5}

        for sel in viewlayer.objects.selected:
            sel.select_set(False)

        for obj in objs:
            if pivotmode is None:
                mode = modedict[obj.sx2.pivotmode]
            else:
                mode = pivotmode

            viewlayer.objects.active = obj
            obj.select_set(True)

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
            else:
                pass

            obj.select_set(False)

        for sel in selected:
            sel.select_set(True)
        viewlayer.objects.active = active


    def create_uvset0(self, objs):
        active = bpy.context.view_layer.objects.active
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            setCount = len(obj.data.uv_layers)
            if setCount == 0:
                print(f'SX Tools Error: {obj.name} has no UV Sets')
            elif setCount > 6:
                base = obj.data.uv_layers[0]
                if base.name != 'UVSet0':
                    print(f'SX Tools Error: {obj.name} does not have enough free UV Set slots for the operation (2 free slots needed)')
            else:
                base = obj.data.uv_layers[0]
                if 'UVSet' not in base.name:
                    base.name = 'UVSet0'
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


    def revert_objects(self, objs):
        modifiers.remove_modifiers(objs)

        layers.clear_layers(objs, objs[0].sx2layers['Overlay'])
        layers.clear_layers(objs, objs[0].sx2layers['Occlusion'])
        layers.clear_layers(objs, objs[0].sx2layers['Metallic'])
        layers.clear_layers(objs, objs[0].sx2layers['Roughness'])
        layers.clear_layers(objs, objs[0].sx2layers['Transmission'])

        active = bpy.context.view_layer.objects.active
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

        bpy.context.view_layer.objects.active = active


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

        # self.validate_sxmaterial(objs)

        ok1 = self.validate_palette_layers(objs)
        ok2 = self.validate_names(objs)
        ok3 = self.validate_loose(objs)
        # ok4 = self.validate_uv_sets(objs)

        utils.mode_manager(objs, revert=True, mode_id='validate_objects')

        if ok1 and ok2 and ok3 and ok4:
            print('SX Tools: Selected objects passed validation tests')
            return True
        else:
            print('SX Tools Error: Selected objects failed validation tests')
            return False


    def validate_sxmaterial(self, objs):
        try:
            sxmaterial = bpy.data.materials['SX2Material'].node_tree
        except:
            print(f'SX Tools Error: Invalid SX Material on {objs}. \nPerforming SX Material reset.')
            bpy.ops.sx2.resetmaterial('EXEC_DEFAULT')


    # Check for proper UV set configuration
    def validate_uv_sets(self, objs):
        for obj in objs:
            if len(obj.data.uv_layers) != 7:
                message_box('Objects contain an invalid number of UV Sets!')
                print(f'SX Tools Error: {obj.name} has invalid UV Sets')
                return False

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

        for obj in objs:
            obj.update_from_editmode()
            mesh = obj.data
            selection = [None] * len(mesh.vertices)
            mesh.vertices.foreach_get('select', selection)

            if True in selection:
                message_box('Objects contain loose geometry!')
                print(f'SX Tools Error: {obj.name} contains loose geometry')
                return False

        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
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
    # TODO: update to sx2 -> check all
    def validate_palette_layers(self, objs):
        for obj in objs:
            if obj.sx2.category != 'DEFAULT':
                mesh = obj.data
                for i in range(5):
                    colorArray = []
                    layer = utils.find_layer_by_stack_index(obj, i+1)
                    vertexColors = mesh.attributes[layer.vertexColorLayer].data

                    for poly in mesh.polygons:
                        for loop_idx in poly.loop_indices:
                            colorArray.append(vertexColors[loop_idx].color[:])

                    colorSet = set(colorArray)

                    if i == 0:
                        if len(colorSet) > 1:
                            print(f'SX Tools Error: Multiple colors in {obj.name} layer {i+1}')
                            message_box('Multiple colors in ' + obj.name + ' layer' + str(i+1))
                            return False
                    else:
                        if len(colorSet) > 2:
                            print(f'SX Tools Error: Multiple colors in {obj.name} layer {i+1}')
                            message_box('Multiple colors in ' + obj.name + ' layer' + str(i+1))
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
        scene = bpy.context.scene.sx2
        viewlayer = bpy.context.view_layer
        orgObjNames = {}

        org_toolmode = scene.toolmode
        org_toolopacity = scene.toolopacity
        org_toolblend = scene.toolblend
        org_rampmode = scene.rampmode
        org_ramplist = scene.ramplist
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
        if 'ExportObjects' not in bpy.data.collections:
            exportObjects = bpy.data.collections.new('ExportObjects')
        else:
            exportObjects = bpy.data.collections['ExportObjects']

        if 'SourceObjects' not in bpy.data.collections:
            sourceObjects = bpy.data.collections.new('SourceObjects')
        else:
            sourceObjects = bpy.data.collections['SourceObjects']

        # Make sure objects are in groups
        for obj in objs:
            if obj.parent is None:
                obj.hide_viewport = False
                # viewlayer.objects.active = obj
                export.group_objects([obj, ])

            # Make sure auto-smooth is on
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(obj.sx2.smoothangle)
            if '_mesh' not in obj.data.name:
                obj.data.name = obj.name + '_mesh'

        # Make sure all objects have UVSet0
        # tools.create_uvset0(objs)

        # Remove empties from selected objects
        for sel in viewlayer.objects.selected:
            if sel.type != 'MESH':
                sel.select_set(False)

        # Create modifiers
        modifiers.add_modifiers(objs)

        # Create high-poly bake meshes
        if scene.exportquality == 'HI':
            newObjs = []
            lodObjs = []
            sepObjs = []
            partObjs = []

            for obj in objs:
                if obj.sx2.smartseparate:
                    if obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror:
                        sepObjs.append(obj)

            if len(sepObjs) > 0:
                partObjs = export.smart_separate(sepObjs)
                for obj in partObjs:
                    if obj not in objs:
                        objs.append(obj)

            groups = utils.find_groups(objs)
            for group in groups:
                orgGroup = bpy.data.objects.new('empty', None)
                bpy.context.scene.collection.objects.link(orgGroup)
                orgGroup.empty_display_size = 2
                orgGroup.empty_display_type = 'PLAIN_AXES'
                orgGroup.location = group.location
                orgGroup.name = group.name + '_org'
                orgGroup.hide_viewport = True
                sourceObjects.objects.link(orgGroup)
                exportObjects.objects.link(group)

            for obj in objs:
                if obj.name not in sourceObjects.objects:
                    sourceObjects.objects.link(obj)
                orgObjNames[obj] = [obj.name, obj.data.name][:]
                obj.data.name = obj.data.name + '_org'
                obj.name = obj.name + '_org'

                newObj = obj.copy()
                newObj.data = obj.data.copy()
                newObj.name = orgObjNames[obj][0]
                newObj.data.name = orgObjNames[obj][1]
                bpy.context.scene.collection.objects.link(newObj)
                exportObjects.objects.link(newObj)

                if obj.sx2.lodmeshes:
                    lodObjs.append(newObj)
                else:
                    newObjs.append(newObj)

                obj.parent = viewlayer.objects[obj.parent.name + '_org']

            if len(lodObjs) > 0:
                newObjArray = export.generate_lods(lodObjs)
                for newObj in newObjArray:
                    newObjs.append(newObj)

            objs = newObjs

        # Place pivots
        export.set_pivots(objs, force=True)

        # Fix transforms
        utils.clear_parent_inverse_matrix(objs)

        for obj in objs:
            obj.select_set(False)

        viewlayer.objects.active = objs[0]

        # Begin category-specific compositing operations
        # Hide modifiers for performance
        for obj in objs:
            obj.sx2.modifiervisibility = False
            if obj.sx2.category == '':
                obj.sx2.category == 'DEFAULT'

        for obj in viewlayer.objects:
            if obj.type == 'MESH' and obj.hide_viewport is False:
                obj.hide_viewport = True

        # Mandatory to update visibility?
        viewlayer.update()

        categoryList = list(sxglobals.categoryDict.keys())
        categories = []
        for category in categoryList:
            categories.append(category.replace(" ", "_").upper())
        for category in categories:
            categoryObjs = []
            for obj in objs:
                if obj.sx2.category == category:
                    categoryObjs.append(obj)

            if len(categoryObjs) > 0:
                groupList = utils.find_groups(categoryObjs)
                for group in groupList:
                    createLODs = False
                    groupObjs = utils.find_children(group, categoryObjs)
                    filter_list = []
                    for groupObj in groupObjs:
                        if groupObj.type == 'MESH':
                            filter_list.append(groupObj)
                    groupObjs = filter_list
                    viewlayer.objects.active = groupObjs[0]
                    for obj in groupObjs:
                        if obj.sx2.lodmeshes:
                            createLODs = True
                        obj.hide_viewport = False
                        obj.select_set(True)

                    if category == 'DEFAULT':
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_default(groupObjs)
                    elif category == 'PALETTED':
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_paletted(groupObjs)
                    elif category == 'VEHICLES':
                        for obj in groupObjs:
                            if ('wheel' in obj.name) or ('tire' in obj.name):
                                scene.occlusionblend = 0.0
                            else:
                                scene.occlusionblend = 0.5
                            if obj.name.endswith('_roof') or obj.name.endswith('_frame') or obj.name.endswith('_dash') or obj.name.endswith('_hood') or ('bumper' in obj.name):
                                obj.modifiers['sxDecimate2'].use_symmetry = True
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        if (scene.exportquality == 'HI') and (createLODs is True):
                            for i in range(3):
                                lodObjs = []
                                for obj in groupObjs:
                                    if '_LOD'+str(i) in obj.name:
                                        lodObjs.append(obj)
                                        # obj.select_set(True)
                                        obj.hide_viewport = False
                                    else:
                                        # obj.select_set(False)
                                        obj.hide_viewport = True
                                viewlayer.objects.active = lodObjs[0]
                                self.process_vehicles(lodObjs)
                        else:
                            self.process_vehicles(groupObjs)
                    elif category == 'BUILDINGS':
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_buildings(groupObjs)
                    elif category == 'TREES':
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_trees(groupObjs)
                    elif category == 'CHARACTERS':
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_characters(groupObjs)
                    elif category == 'ROADTILES':
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_roadtiles(groupObjs)
                    elif category == 'TRANSPARENT':
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_default(groupObjs)
                    else:
                        if scene.exportquality == 'HI':
                            tools.apply_modifiers(groupObjs)
                        self.process_default(groupObjs)

                    for obj in groupObjs:
                        obj.select_set(False)
                        obj.hide_viewport = True

                now = time.perf_counter()
                print(f'SX Tools: {category} / {len(groupList)} groups duration: {now-then} seconds')

        for obj in viewlayer.objects:
            if (scene.exportquality == 'HI') and ('_org' in obj.name):
                obj.hide_viewport = True
            elif obj.type == 'MESH':
                obj.hide_viewport = False

        # Bake Overlay for export
        # export.flatten_alphas(objs)

        # LOD mesh generation for low-detail
        if scene.exportquality == 'LO':
            nonLodObjs = []
            for obj in objs:
                if obj.sx2.lodmeshes is True:
                    if '_LOD' not in obj.name:
                        nonLodObjs.append(obj)
                else:
                    obj.select_set(True)

            if len(nonLodObjs) > 0:
                lodObjs = export.generate_lods(nonLodObjs)
                bpy.ops.object.select_all(action='DESELECT')
                for obj in nonLodObjs:
                    obj.select_set(True)
                for obj in lodObjs:
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

        now = time.perf_counter()
        print(f'SX Tools: Mesh processing duration: {now-then} seconds')

        scene.toolmode = org_toolmode
        scene.toolopacity = org_toolopacity
        scene.toolblend = org_toolblend
        scene.rampmode = org_rampmode
        scene.ramplist = org_ramplist
        scene.dirAngle = org_dirangle
        scene.dirInclination = org_dirinclination
        scene.dirCone = org_dircone

        # Enable modifier stack for LODs
        for obj in objs:
            obj.sx2.modifiervisibility = True

        now = time.perf_counter()
        print(f'SX Tools: Modifier stack duration: {now-then} seconds')

        utils.mode_manager(objs, revert=True, mode_id='process_objects')
        sxglobals.refresh_in_progress = False


    def process_default(self, objs):
        print('SX Tools: Processing Default')
        scene = bpy.context.scene.sx2
        obj = objs[0]

        # Apply occlusion
        layer = obj.sx2layers['Occlusion']
        scene.toolmode = 'OCC'
        scene.noisemono = True
        scene.occlusionblend = 0.5
        scene.occlusionrays = 1000
        scene.occlusiondistance = 10.0

        tools.apply_tool(objs, layer)

        # Apply custom overlay
        layer = obj.sx2layers['Overlay']
        scene.toolmode = 'CRV'
        scene.curvaturenormalize = True

        tools.apply_tool(objs, layer)

        scene.noisemono = False
        scene.toolmode = 'NSE'
        scene.toolopacity = 0.01
        scene.toolblend = 'MUL'

        tools.apply_tool(objs, layer)
        for obj in objs:
            obj.sx2layers['Overlay'].blend_mode = 'OVR'
            # obj.sx2layers['Overlay'].opacity = obj.sx2.overlaystrength

        # Emissives are smooth
        color = (0.0, 0.0, 0.0, 1.0)
        masklayer = obj.sx2layers['Emission']
        layer = obj.sx2layers['Roughness']
        tools.apply_tool(objs, layer, masklayer=masklayer, color=color)


    def process_roadtiles(self, objs):
        print('SX Tools: Processing Roadtiles')
        scene = bpy.context.scene.sx2

        # Apply occlusion
        layer = objs[0].sx2layers['Occlusion']
        scene.toolmode = 'OCC'
        scene.noisemono = True
        scene.occlusionblend = 0.0
        scene.occlusionrays = 1000
        scene.occlusiondistance = 50.0

        tools.apply_tool(objs, layer)

        # Apply Gradient 1 to roughness and metallic
        for obj in objs:
            obj.sx2layers['Gradient 1'].opacity = 1.0
            colors = layers.get_layer(obj, obj.sx2layers['Gradient 1'])
            if colors is not None:
                layers.set_layer(obj, colors, obj.sx2layers['Metallic'])
                layers.set_layer(obj, colors, obj.sx2layers['Roughness'])

        # Painted roads are smooth-ish
        color = (0.5, 0.5, 0.5, 1.0)
        masklayer = utils.find_color_layers(objs[0], 2)
        layer = objs[0].sx2layers['Metallic']
        tools.apply_tool(objs, layer, masklayer=masklayer, color=color)
        layer = objs[0].sx2layers['Roughness']
        tools.apply_tool(objs, layer, masklayer=masklayer, color=color)

        # Emissives are smooth
        color = (0.0, 0.0, 0.0, 1.0)
        masklayer = objs[0].sx2layers['Emission']
        layer = objs[0].sx2layers['Roughness']
        tools.apply_tool(objs, layer, masklayer=masklayer, color=color)

        # Tone down Gradient 1 contribution
        for obj in objs:
            obj.sx2layers['Gradient 1'].opacity = 0.15


    def process_characters(self, objs):
        print('SX Tools: Processing Characters')
        scene = bpy.context.scene.sx2
        obj = objs[0]

        # Apply occlusion
        layer = obj.sx2layers['Occlusion']
        scene.toolmode = 'OCC'
        scene.noisemono = True
        scene.occlusionblend = 0.2
        scene.occlusionrays = 1000
        scene.occlusiondistance = 1.0

        tools.apply_tool(objs, layer)

        # Apply custom overlay
        layer = obj.sx2layers['Overlay']
        scene.toolmode = 'CRV'
        scene.curvaturenormalize = True
        scene.toolopacity = 0.2
        scene.toolblend = 'ALPHA'

        tools.apply_tool(objs, layer)

        scene.noisemono = False
        scene.toolmode = 'NSE'
        scene.toolopacity = 0.01
        scene.toolblend = 'MUL'

        tools.apply_tool(objs, layer)
        for obj in objs:
            obj.sx2layers['Overlay'].blend_mode = 'OVR'
            # obj.sx2layers['Overlay'].opacity = obj.sx2.overlaystrength

        # Emissives are smooth
        color = (0.0, 0.0, 0.0, 1.0)
        masklayer = obj.sx2layers['Emission']
        layer = obj.sx2layers['Roughness']
        tools.apply_tool(objs, layer, masklayer=masklayer, color=color)


    def process_paletted(self, objs):
        print('SX Tools: Processing Paletted')
        scene = bpy.context.scene.sx2
        obj = objs[0]

        # Apply occlusion masked by emission
        scene.occlusionblend = 0.5
        scene.occlusionrays = 1000
        scene.occlusiondistance = 10.0

        for obj in objs:
            layer = obj.sx2layers['Occlusion']
            colors0 = generate.occlusion_list(obj, scene.occlusionrays, scene.occlusionblend, scene.occlusiondistance, scene.occlusiongroundplane)
            colors1 = layers.get_layer(obj, obj.sx2layers['Emission'], single_as_alpha=True)
            colors = tools.blend_values(colors1, colors0, 'ALPHA', 1.0)

            layers.set_layer(obj, colors, layer)

        # Apply custom overlay
        layer = obj.sx2layers['Overlay']
        scene.toolmode = 'CRV'
        scene.curvaturenormalize = True

        tools.apply_tool(objs, layer)

        scene.toolmode = 'NSE'
        scene.toolopacity = 0.01
        scene.toolblend = 'MUL'
        scene.noisemono = False

        tools.apply_tool(objs, layer)

        scene.toolopacity = 1.0
        scene.toolblend = 'ALPHA'

        for obj in objs:
            obj.sx2layers['Overlay'].blend_mode = 'OVR'
            # obj.sx2layers['Overlay'].opacity = obj.sx2.overlaystrength

        # Clear metallic, roughness, and transmission
        layers.clear_layers(objs, obj.sx2layers['Metallic'])
        layers.clear_layers(objs, obj.sx2layers['Roughness'])
        layers.clear_layers(objs, obj.sx2layers['Transmission'])

        material = 'Iron'
        palette = [
            bpy.context.scene.sx2materials[material].color0,
            bpy.context.scene.sx2materials[material].color1,
            bpy.context.scene.sx2materials[material].color2]

        inverted_color = [None, None, None, 1.0]
        for i in range(3):
            inverted_color[i] = 1 - palette[2][i]
        palette[2] = inverted_color

        for obj in objs:
            # Construct layer1-7 roughness base mask
            color = (obj.sx2.roughness0, obj.sx2.roughness0, obj.sx2.roughness0, 1.0)
            colors = generate.color_list(obj, color)
            color = (obj.sx2.roughness4, obj.sx2.roughness4, obj.sx2.roughness4, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 3))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 4))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            color = (1.0, 1.0, 1.0, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 5))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Combine with roughness from PBR material
            colors1 = generate.color_list(obj, palette[2], utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Noise for variance
            colors1 = generate.noise_list(obj, 0.01, True)
            colors = tools.blend_values(colors1, colors, 'OVR', 1.0)
            # Combine roughness base mask with custom curvature gradient
            scene.curvaturenormalize = True
            scene.ramplist = 'CURVATUREROUGHNESS'
            colors1 = generate.curvature_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
            # Combine previous mix with directional dust
            scene.ramplist = 'DIRECTIONALDUST'
            scene.dirAngle = 0.0
            scene.dirInclination = 90.0
            scene.dirCone = 30
            colors1 = generate.direction_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
            # Emissives are smooth
            color = (0.0, 0.0, 0.0, 1.0)
            colors1 = generate.color_list(obj, color, obj.sx2layers['Emission'])
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Write roughness
            layer = obj.sx2layers['Roughness']
            layers.set_layer(obj, colors, layer)
            # layer.locked = True

            # Apply PBR metal based on layer7
            # noise = 0.01
            # mono = True
            scene.toolmode = 'COL'
            scene.toolopacity = 1.0
            scene.toolblend = 'ALPHA'

            # Mix metallic with occlusion (dirt in crevices)
            colors = generate.color_list(obj, color=palette[1], masklayer=utils.find_color_layers(obj, 6))
            colors1 = layers.get_layer(obj, obj.sx2layers['Occlusion'], single_as_alpha=True)
            if colors is not None:
                colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
                layers.set_layer(obj, colors, obj.sx2layers['Metallic'])


    def process_vehicles(self, objs):
        print('SX Tools: Processing Vehicles')
        scene = bpy.context.scene.sx2

        # Apply occlusion masked by emission
        scene.occlusionblend = 0.5
        scene.occlusionrays = 1000
        scene.occlusiongroundplane = True
        scene.occlusiondistance = 10.0

        for obj in objs:
            layer = obj.sx2layers['Occlusion']
            colors0 = generate.occlusion_list(obj, scene.occlusionrays, scene.occlusionblend, scene.occlusiondistance, scene.occlusiongroundplane)
            colors1 = layers.get_layer(obj, obj.sx2layers['Emission'])
            colors = tools.blend_values(colors1, colors0, 'ALPHA', 1.0)
            layers.set_layer(obj, colors, layer)

        obj = objs[0]
        # Apply custom overlay
        layer = obj.sx2layers['Overlay']
        scene.toolmode = 'CRV'
        scene.curvaturenormalize = True

        tools.apply_tool(objs, layer)

        scene.toolmode = 'NSE'
        scene.toolopacity = 0.01
        scene.toolblend = 'MUL'
        scene.noisemono = False

        tools.apply_tool(objs, layer)

        scene.toolopacity = 1.0
        scene.toolblend = 'ALPHA'

        # for obj in objs:
        #     obj.sx2layers['Overlay'].blend_mode = 'OVR'

        # Clear metallic, roughness, and transmission
        layers.clear_layers(objs, obj.sx2layers['Metallic'])
        layers.clear_layers(objs, obj.sx2layers['Roughness'])
        layers.clear_layers(objs, obj.sx2layers['Transmission'])

        material = 'Iron'
        palette = [
            bpy.context.scene.sx2materials[material].color0,
            bpy.context.scene.sx2materials[material].color1,
            bpy.context.scene.sx2materials[material].color2]

        inverted_color = [None, None, None, 1.0]
        for i in range(3):
            inverted_color[i] = 1 - palette[2][i]
        palette[2] = inverted_color

        for obj in objs:
            # Construct layer1-7 roughness base mask
            color = (obj.sx2.roughness0, obj.sx2.roughness0, obj.sx2.roughness0, 1.0)
            colors = generate.color_list(obj, color)
            color = (obj.sx2.roughness4, obj.sx2.roughness4, obj.sx2.roughness4, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 3))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 4))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            color = (1.0, 1.0, 1.0, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 5))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Combine with roughness from PBR material
            colors1 = generate.color_list(obj, palette[2], utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Noise for variance
            colors1 = generate.noise_list(obj, 0.01, True)
            colors = tools.blend_values(colors1, colors, 'OVR', 1.0)
            # Combine roughness base mask with custom curvature gradient
            scene.curvaturenormalize = True
            scene.ramplist = 'CURVATUREROUGHNESS'
            colors1 = generate.curvature_list(obj)
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
            colors = tools.blend_values(colors1, colors, 'ADD', 0.1)
            # Emissives are smooth
            color = (0.0, 0.0, 0.0, 1.0)
            colors1 = generate.color_list(obj, color, obj.sx2layers['Emission'])
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Write roughness
            layer = obj.sx2layers['Roughness']
            layers.set_layer(obj, colors, layer)
            # layer.locked = True

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
        print('SX Tools: Processing Buildings')
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

        # Apply occlusion
        scene.toolmode = 'OCC'
        scene.noisemono = True
        scene.occlusionblend = 0.0
        scene.occlusionrays = 1000
        scene.occlusiongroundplane = False
        scene.occlusiondistance = 0.5
        color = (1.0, 1.0, 1.0, 1.0)
        mask = utils.find_color_layers(obj, 6)

        for obj in objs:
            layer = obj.sx2layers['Occlusion']
            colors = generate.occlusion_list(obj, scene.occlusionrays, scene.occlusionblend, scene.occlusiondistance, scene.occlusiongroundplane)
            colors1 = generate.color_list(obj, color=color, masklayer=mask)
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)

            layers.set_layer(obj, colors, layer)

        # Apply normalized curvature with luma remapping to overlay, clear windows
        scene.curvaturenormalize = False
        scene.ramplist = 'WEARANDTEAR'
        for obj in objs:
            layer = obj.sx2layers['Overlay']
            colors = generate.curvature_list(obj)
            values = layers.get_luminances(obj, colors=colors)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            color = (0.5, 0.5, 0.5, 1.0)
            colors1 = generate.color_list(obj, color=color, masklayer=mask)
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            layers.set_layer(obj, colors, layer)
            obj.sx2layers['Overlay'].blend_mode = 'OVR'
            # obj.sx2layers['Overlay'].opacity = obj.sx2.overlaystrength


        # Clear metallic, roughness, and transmission
        layers.clear_layers(objs, obj.sx2layers['Metallic'])
        layers.clear_layers(objs, obj.sx2layers['Roughness'])
        layers.clear_layers(objs, obj.sx2layers['Transmission'])

        material = 'Silver'
        palette = [
            bpy.context.scene.sx2materials[material].color0,
            bpy.context.scene.sx2materials[material].color1,
            bpy.context.scene.sx2materials[material].color2]

        for obj in objs:
            # Construct layer1-7 roughness base mask
            color = (obj.sx2.roughness0, obj.sx2.roughness0, obj.sx2.roughness0, 1.0)
            colors = generate.color_list(obj, color)
            color = (obj.sx2.roughness4, obj.sx2.roughness4, obj.sx2.roughness4, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 4))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 5))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            color = (1.0, 1.0, 1.0, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Combine with roughness from PBR material
            colors1 = generate.color_list(obj, palette[2], utils.find_color_layers(obj, 6))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Noise for variance
            # colors1 = generate.noise_list(obj, 0.01, True)
            # colors = tools.blend_values(colors1, colors, 'OVR', 1.0)
            # Combine roughness base mask with custom curvature gradient
            scene.curvaturenormalize = True
            scene.ramplist = 'CURVATUREROUGHNESS'
            colors1 = generate.curvature_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
            # Combine previous mix with directional dust
            scene.ramplist = 'DIRECTIONALDUST'
            scene.dirAngle = 0.0
            scene.dirInclination = 90.0
            scene.dirCone = 30
            colors1 = generate.direction_list(obj)
            values = layers.get_luminances(obj, colors=colors1)
            colors1 = generate.luminance_remap_list(obj, values=values)
            colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
            # Emissives are smooth
            color = (0.0, 0.0, 0.0, 1.0)
            colors1 = generate.color_list(obj, color, obj.sx2layers['Emission'])
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            # Write roughness
            layer = obj.sx2layers['Roughness']
            layers.set_layer(obj, colors, layer)
            # layer.locked = True

            # Apply PBR metal based on layer7
            # noise = 0.01
            # mono = True
            scene.toolmode = 'COL'
            scene.toolopacity = 1.0
            scene.toolblend = 'ALPHA'

            colors = generate.color_list(obj, color=palette[0], masklayer=utils.find_color_layers(obj, 6))
            if colors is not None:
                layers.set_layer(obj, colors, obj.sx2layers['Metallic'])

            # Windows are smooth and emissive
            colors = generate.color_list(obj, color=(1.0, 1.0, 1.0, 1.0), masklayer=utils.find_color_layers(obj, 6))
            base = layers.get_layer(obj, obj.sx2layers['Roughness'])
            colors = tools.blend_values(colors, base, 'ALPHA', 1.0)
            if colors is not None:
                layers.set_layer(obj, colors, obj.sx2layers['Roughness'])
            emission = generate.color_list(obj, color=(0.1, 0.1, 0.1, 0.1), masklayer=utils.find_color_layers(obj, 7))
            if emission is not None:
                layers.set_layer(obj, emission, obj.sx2layers['Emission'])


        for obj in objs_windows:
            obj.hide_viewport = False


    def process_trees(self, objs):
        print('SX Tools: Processing Trees')
        scene = bpy.context.scene.sx2
        obj = objs[0]

        # Apply occlusion
        scene.toolmode = 'OCC'
        scene.noisemono = True
        scene.occlusionblend = 0.5
        scene.occlusionrays = 500
        color = (1.0, 1.0, 1.0, 1.0)

        for obj in objs:
            layer = obj.sx2layers['Occlusion']
            colors = generate.occlusion_list(obj, scene.occlusionrays, scene.occlusionblend, scene.occlusiondistance, scene.occlusiongroundplane)
            layers.set_layer(obj, colors, layer)

        # Apply overlay
        scene.curvaturenormalize = True
        for obj in objs:
            layer = obj.sx2layers['Overlay']
            colors = generate.curvature_list(obj)
            # Noise for variance
            colors1 = generate.noise_list(obj, 0.03, False)
            colors = tools.blend_values(colors1, colors, 'OVR', 1.0)
            layers.set_layer(obj, colors, layer)
            obj.sx2layers['Overlay'].blend_mode = 'OVR'
            # obj.sx2layers['Overlay'].opacity = obj.sx2.overlaystrength

        # Clear metallic, roughness, and transmission
        layers.clear_layers(objs, obj.sx2layers['Metallic'])
        layers.clear_layers(objs, obj.sx2layers['Roughness'])
        layers.clear_layers(objs, obj.sx2layers['Transmission'])

        # Apply thickness to transmission
        layer = obj.sx2layers['Transmission']
        scene.ramplist = 'FOLIAGERAMP'
        color = (1.0, 1.0, 1.0, 1.0)

        for obj in objs:
            colors = generate.thickness_list(obj, 500, masklayer=None)
            values = layers.get_luminances(obj, colors=colors)
            colors = generate.luminance_remap_list(obj, values=values)
            colors1 = generate.color_list(obj, color=color, masklayer=utils.find_color_layers(obj, 3))
            colors2 = generate.color_list(obj, color=color, masklayer=utils.find_color_layers(obj, 4))
            colors1 = tools.blend_values(colors2, colors1, 'ALPHA', 1.0)
            colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
            layers.set_layer(obj, colors, layer)
            # Construct layer1-7 roughness base mask
            color = (obj.sx2.roughness0, obj.sx2.roughness0, obj.sx2.roughness0, 1.0)
            colors = generate.color_list(obj, color)
            color = (obj.sx2.roughness4, obj.sx2.roughness4, obj.sx2.roughness4, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 3))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 4))
            colors = tools.blend_values(colors1, colors, 'ALPHA', 1.0)
            color = (1.0, 1.0, 1.0, 1.0)
            colors1 = generate.color_list(obj, color, utils.find_color_layers(obj, 5))
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
            colors = tools.blend_values(colors1, colors, 'MUL', 1.0)
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


    def start_modal(self):
        bpy.ops.sx2.selectionmonitor('EXEC_DEFAULT')
        bpy.ops.sx2.keymonitor('EXEC_DEFAULT')
        sxglobals.modal_status = True


    def create_tiler(self):
        nodetree = bpy.data.node_groups.new(type='GeometryNodeTree', name='sx_tiler')
        group_in = nodetree.nodes.new(type='NodeGroupInput')
        group_in.name = 'group_input'
        group_in.location = (-800, 0)
        group_out = nodetree.nodes.new(type='NodeGroupOutput')
        group_out.name = 'group_output'
        group_out.location = (1200, 0)

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
        output = group_in.outputs[0]
        input = nodetree.nodes['bbx'].inputs['Geometry']
        nodetree.links.new(input, output)
        output = group_in.outputs[0]
        input = nodetree.nodes['flip'].inputs[0]
        nodetree.links.new(input, output)

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
            output = offset.outputs[0]
            input = multiply.inputs[1]
            nodetree.links.new(input, output)

            # link bbx bounds to multipliers
            output = bbx.outputs[i+1]
            input = multiply.inputs[0]
            nodetree.links.new(input, output)

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
        nodetree.inputs.new('NodeSocketFloat', 'New')
        output = group_in.outputs[1]
        input = combine_offset.inputs[0]
        nodetree.links.new(output, input)
        output = group_in.outputs[1]
        input = combine_offset.inputs[1]
        nodetree.links.new(output, input)
        output = group_in.outputs[1]
        input = combine_offset.inputs[2]
        nodetree.links.new(output, input)

        output = combine_offset.outputs[0]
        input = bbx_subtract.inputs[1]
        nodetree.links.new(output, input)
        input = bbx_add.inputs[1]
        nodetree.links.new(output, input)

        # link multipliers to bbx offsets
        output = nodetree.nodes['multiply0'].outputs[0]
        input = nodetree.nodes['bbx_subtract'].inputs[0]
        nodetree.links.new(input, output)
        output = nodetree.nodes['multiply1'].outputs[0]
        input = nodetree.nodes['bbx_add'].inputs[0]
        nodetree.links.new(input, output)

        # bbx to separate min and max
        output = nodetree.nodes['bbx_subtract'].outputs[0]
        input = nodetree.nodes['separate0'].inputs[0]
        nodetree.links.new(input, output)
        output = nodetree.nodes['bbx_add'].outputs[0]
        input = nodetree.nodes['separate1'].inputs[0]
        nodetree.links.new(input, output)

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
            output = combine.outputs[0]
            input = transform.inputs[1]
            nodetree.links.new(input, output)

            # expose axis enable switches
            nodetree.inputs.new('NodeSocketBool', 'New')
            output = group_in.outputs[2+i]
            input = switch.inputs[1]
            nodetree.links.new(output, input)

            # link flipped mesh to switch input
            output = flip.outputs[0]
            input = switch.inputs[15]
            nodetree.links.new(input, output)

            # link switch mesh output to transform
            output = switch.outputs[6]
            input = transform.inputs[0]
            nodetree.links.new(input, output)

            # link mirror mesh output to join node
            output = transform.outputs[0]
            input = join.inputs[0]
            nodetree.links.new(input, output)

        # link group in as first in join node for working auto-smooth
        output = group_in.outputs[0]
        input = join.inputs[0]
        nodetree.links.new(input, output)

        # link combined geometry to group output
        output = nodetree.nodes['join'].outputs['Geometry']
        input = group_out.inputs[0]
        nodetree.links.new(input, output)

        # min max pairs to combiners in -x, x, -y, y, -z, z order
        output = nodetree.nodes['separate0'].outputs[0]
        input = nodetree.nodes['combine0'].inputs[0]
        nodetree.links.new(input, output)
        output = nodetree.nodes['separate1'].outputs[0]
        input = nodetree.nodes['combine1'].inputs[0]
        nodetree.links.new(input, output)

        output = nodetree.nodes['separate0'].outputs[1]
        input = nodetree.nodes['combine2'].inputs[1]
        nodetree.links.new(input, output)
        output = nodetree.nodes['separate1'].outputs[1]
        input = nodetree.nodes['combine3'].inputs[1]
        nodetree.links.new(input, output)

        output = nodetree.nodes['separate0'].outputs[2]
        input = nodetree.nodes['combine4'].inputs[2]
        nodetree.links.new(input, output)
        output = nodetree.nodes['separate1'].outputs[2]
        input = nodetree.nodes['combine5'].inputs[2]
        nodetree.links.new(input, output)

        nodetree.nodes['transform0'].inputs[3].default_value[0] = -3.0
        nodetree.nodes['transform1'].inputs[3].default_value[0] = -3.0
        nodetree.nodes['transform2'].inputs[3].default_value[1] = -3.0
        nodetree.nodes['transform3'].inputs[3].default_value[1] = -3.0
        nodetree.nodes['transform4'].inputs[3].default_value[2] = -3.0
        nodetree.nodes['transform5'].inputs[3].default_value[2] = -3.0


    def create_sxtoolmaterial(self):
        sxmaterial = bpy.data.materials.new(name='SXToolMaterial')
        sxmaterial.use_nodes = True

        sxmaterial.node_tree.nodes.remove(sxmaterial.node_tree.nodes['Principled BSDF'])
        sxmaterial.node_tree.nodes['Material Output'].location = (-200, 0)

        # Gradient tool color ramp
        sxmaterial.node_tree.nodes.new(type='ShaderNodeValToRGB')
        sxmaterial.node_tree.nodes['ColorRamp'].location = (-1000, 0)


    def create_sx2material(self, objs):
        blend_mode_dict = {'ALPHA': 'MIX', 'OVR': 'OVERLAY', 'MUL': 'MULTIPLY', 'ADD': 'ADD'}
        scene = bpy.context.scene.sx2

        for obj in objs:
            if len(obj.sx2layers) > 0:
                sxmaterial = bpy.data.materials.new(name='SX2Material_'+obj.name)
                sxmaterial.use_nodes = True
                sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission'].default_value = [0.0, 0.0, 0.0, 1.0]
                sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = [0.0, 0.0, 0.0, 1.0]
                sxmaterial.node_tree.nodes['Principled BSDF'].location = (1000, 0)
                sxmaterial.node_tree.nodes['Material Output'].location = (1300, 0)

                if obj.sx2.shadingmode == 'FULL':
                    sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
                    prev_color = None
                    rgba_mats = ['SSS', 'EMI']
                    alpha_mats = ['OCC', 'MET', 'RGH', 'TRN']
                    color_layers = []
                    material_layers = []
                    if len(sxglobals.layer_stack_dict.get(obj.name, {})) > 0:
                        color_stack = [sxglobals.layer_stack_dict[obj.name][key] for key in sorted(sxglobals.layer_stack_dict[obj.name].keys())]
                        layer_ids = {}
                        for key, values in sxglobals.layer_stack_dict[obj.name].items():
                            layer_ids[values[0]] = key

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

                        output = input_vector.outputs['Vector']
                        input = input_splitter.inputs['Vector']
                        sxmaterial.node_tree.links.new(input, output)   


                    # Create color layers
                    if (len(color_layers) > 0) and obj.sx2layers[color_layers[0][0]].visibility:
                        base_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                        base_color.name = 'BaseColor'
                        base_color.label = 'BaseColor'
                        base_color.layer_name = color_layers[0][1]
                        base_color.location = (-1000, 0)

                        base_palette = sxmaterial.node_tree.nodes.new(type="ShaderNodeRGB")
                        base_palette.name = 'PaletteColor' + color_layers[0][0]
                        base_palette.label = 'PaletteColor' + color_layers[0][0]
                        base_palette.outputs[0].default_value = getattr(scene, 'newpalette'+str(obj.sx2layers[color_layers[0][0]].palette_index))
                        base_palette.location = (-1000, -150)

                        palette_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                        palette_blend.inputs[0].default_value = 0
                        palette_blend.use_clamp = True
                        palette_blend.location = (-800, -150)

                        output = base_color.outputs['Color']
                        input = palette_blend.inputs['Color1']
                        sxmaterial.node_tree.links.new(input, output)

                        output = base_palette.outputs['Color']
                        input = palette_blend.inputs['Color2']
                        sxmaterial.node_tree.links.new(input, output)

                        if obj.sx2layers[color_layers[0][0]].paletted:
                            output = paletted_shading.outputs[2]
                            input = palette_blend.inputs[0]
                            sxmaterial.node_tree.links.new(input, output)   

                        prev_color = palette_blend.outputs['Color']

                    # Layer groups
                    splitter_index = 0
                    count = 1
                    for i in range(len(color_layers) - 1):
                        if obj.sx2layers[color_layers[i+1][0]].visibility:
                            source_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                            source_color.name = 'LayerColor' + str(i + 1)
                            source_color.label = 'LayerColor' + str(i + 1)
                            source_color.layer_name = color_layers[i+1][1]
                            source_color.location = (-1000, i*400+400)

                            layer_palette = sxmaterial.node_tree.nodes.new(type="ShaderNodeRGB")
                            layer_palette.name = 'PaletteColor' + color_layers[i+1][0]
                            layer_palette.label = 'PaletteColor' + color_layers[i+1][0]
                            layer_palette.outputs[0].default_value = getattr(scene, 'newpalette'+str(obj.sx2layers[color_layers[i+1][0]].palette_index))
                            layer_palette.location = (-1000, i*400+250)

                            palette_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                            palette_blend.inputs[0].default_value = 0
                            palette_blend.use_clamp = True
                            palette_blend.location = (-800, i*400+250)

                            output = source_color.outputs['Color']
                            input = palette_blend.inputs['Color1']
                            sxmaterial.node_tree.links.new(input, output)

                            output = layer_palette.outputs['Color']
                            input = palette_blend.inputs['Color2']
                            sxmaterial.node_tree.links.new(input, output)

                            if obj.sx2layers[color_layers[i+1][0]].paletted:
                                output = paletted_shading.outputs[2]
                                input = palette_blend.inputs[0]
                                sxmaterial.node_tree.links.new(input, output)

                            opacity_and_alpha = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                            opacity_and_alpha.name = 'Opacity ' + str(i + 1)
                            opacity_and_alpha.label = 'Opacity ' + str(i + 1)
                            opacity_and_alpha.operation = 'MULTIPLY'
                            opacity_and_alpha.use_clamp = True
                            opacity_and_alpha.inputs[0].default_value = 1
                            opacity_and_alpha.location = (-800, i*400+400)

                            layer_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                            layer_blend.name = 'Mix ' + str(i + 1)
                            layer_blend.label = 'Mix ' + str(i + 1)
                            layer_blend.inputs[0].default_value = 1
                            layer_blend.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                            layer_blend.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                            layer_blend.blend_type = blend_mode_dict[obj.sx2layers[color_layers[i+1][0]].blend_mode]
                            layer_blend.use_clamp = True
                            layer_blend.location = (-600, i*400+400)

                            # Group connections
                            output = palette_blend.outputs['Color']
                            input = layer_blend.inputs['Color2']
                            sxmaterial.node_tree.links.new(input, output)

                            output = sxmaterial.node_tree.nodes['Input Splitter ' + str(splitter_index)].outputs[(i+1) % 3]
                            input = opacity_and_alpha.inputs[0]
                            sxmaterial.node_tree.links.new(input, output)
                            count += 1
                            if count == 3:
                                splitter_index += 1
                                count = 0

                            output = source_color.outputs['Alpha']
                            input = opacity_and_alpha.inputs[1]
                            sxmaterial.node_tree.links.new(input, output)

                            output = opacity_and_alpha.outputs[0]
                            input = layer_blend.inputs[0]
                            sxmaterial.node_tree.links.new(input, output)

                            if prev_color is not None:
                                output = prev_color
                                input = layer_blend.inputs['Color1']
                                sxmaterial.node_tree.links.new(input, output)

                            prev_color = layer_blend.outputs['Color']

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

                        output = input_vector.outputs['Vector']
                        input = input_splitter.inputs['Vector']
                        sxmaterial.node_tree.links.new(input, output)   

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

                                output = source_mats.outputs['Color']
                                input = source_alphas.inputs['Vector']
                                sxmaterial.node_tree.links.new(input, output)

                            elif (material_layers[i][2] not in alpha_mats):
                                name = material_layers[i][0]

                                source_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                                source_color.name = name
                                source_color.label = name
                                source_color.layer_name = material_layers[i][1]
                                source_color.location = (0, -i*400-800)

                                source_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                                source_blend.inputs[0].default_value = 0
                                source_blend.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                                source_blend.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                                source_blend.use_clamp = True
                                source_blend.location = (200, -i*400-800)

                                material_palette = sxmaterial.node_tree.nodes.new(type="ShaderNodeRGB")
                                material_palette.name = 'PaletteColor' + name
                                material_palette.label = 'PaletteColor' + name
                                material_palette.outputs[0].default_value = getattr(scene, 'newpalette'+str(obj.sx2layers[material_layers[i][0]].palette_index))
                                material_palette.location = (0, -i*400-950)

                                palette_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                                palette_blend.inputs[0].default_value = 0
                                palette_blend.use_clamp = True
                                palette_blend.location = (200, -i*400-950)

                                output = source_color.outputs['Alpha']
                                input = source_blend.inputs[0]
                                sxmaterial.node_tree.links.new(input, output)

                                output = material_palette.outputs['Color']
                                input = source_blend.inputs['Color2']
                                sxmaterial.node_tree.links.new(input, output)

                                output = source_color.outputs['Color']
                                input = palette_blend.inputs['Color1']
                                sxmaterial.node_tree.links.new(input, output)

                                output = source_blend.outputs['Color']
                                input = palette_blend.inputs['Color2']
                                sxmaterial.node_tree.links.new(input, output)

                                if obj.sx2layers[material_layers[i][0]].paletted:
                                    output = paletted_shading.outputs[2]
                                    input = palette_blend.inputs[0]
                                    sxmaterial.node_tree.links.new(input, output)   

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
                            output = sxmaterial.node_tree.nodes['Material Input Splitter ' + str(splitter_index)].outputs[i % 3]
                            input = layer_blend.inputs[0]   
                            sxmaterial.node_tree.links.new(input, output)

                            count += 1
                            if count == 3:
                                splitter_index += 1
                                count = 0

                            if (material_layers[i][2] == 'OCC'):
                                layer_blend.blend_type = 'MULTIPLY'
                                if prev_color is not None:
                                    output = prev_color
                                    input = layer_blend.inputs['Color1']
                                    sxmaterial.node_tree.links.new(input, output)

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
                            input = layer_blend.inputs[2]
                            sxmaterial.node_tree.links.new(input, output)

                            output = layer_blend.outputs[0]
                            if material_layers[i][2] == 'MET':
                                input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Metallic']
                                sxmaterial.node_tree.links.new(input, output)

                            if material_layers[i][2] == 'RGH':
                                input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Roughness']
                                sxmaterial.node_tree.links.new(input, output)

                            if material_layers[i][2] == 'TRN':
                                input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Transmission']
                                sxmaterial.node_tree.links.new(input, output)

                            if material_layers[i][2] == 'SSS':
                                input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Subsurface']
                                sxmaterial.node_tree.links.new(input, output)

                                output = palette_blend.outputs['Color']
                                input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Subsurface Color']
                                sxmaterial.node_tree.links.new(input, output)

                            if material_layers[i][2] == 'EMI':
                                sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 100
                                input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission']
                                sxmaterial.node_tree.links.new(input, output)
                                bpy.context.scene.eevee.use_bloom = True

                    if prev_color is not None:
                        output = prev_color
                        input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color']
                        sxmaterial.node_tree.links.new(input, output)

                elif (obj.sx2.shadingmode == 'DEBUG') or (obj.sx2.shadingmode == 'ALPHA'):
                    sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.0
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

                        output = base_color.outputs['Color']
                        input = source_alphas.inputs['Vector']
                        sxmaterial.node_tree.links.new(input, output)

                    if layer.layer_type not in alpha_mats:
                        opacity_and_alpha = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                        opacity_and_alpha.name = 'Opacity and Alpha'
                        opacity_and_alpha.label = 'Opacity and Alpha'
                        opacity_and_alpha.operation = 'MULTIPLY'
                        opacity_and_alpha.use_clamp = True
                        opacity_and_alpha.inputs[0].default_value = 1
                        opacity_and_alpha.inputs[1].default_value = 1
                        opacity_and_alpha.location = (-800, -400)

                        output = layer_opacity.outputs[2]
                        input = opacity_and_alpha.inputs[0]
                        sxmaterial.node_tree.links.new(input, output)

                        output = base_color.outputs['Alpha']
                        input = opacity_and_alpha.inputs[1]
                        sxmaterial.node_tree.links.new(input, output)

                        output = opacity_and_alpha.outputs[0]
                        input = debug_blend.inputs[0]
                        sxmaterial.node_tree.links.new(input, output)
                    else:
                        output = layer_opacity.outputs[2]
                        input = debug_blend.inputs[0]
                        sxmaterial.node_tree.links.new(input, output)

                    if layer.layer_type not in alpha_mats:
                        if obj.sx2.shadingmode == 'DEBUG':
                            output = base_color.outputs['Color']
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
                    input = debug_blend.inputs['Color2']
                    sxmaterial.node_tree.links.new(input, output)

                    output = debug_blend.outputs['Color']
                    input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission']
                    sxmaterial.node_tree.links.new(input, output)


    def update_sx2material(self, context):
        if not sxglobals.magic_in_progress:
            if 'SXToolMaterial' not in bpy.data.materials:
                setup.create_sxtoolmaterial()

            objs = mesh_selection_validator(self, context)
            sx_mat_objs = []
            for obj in objs:
                if 'sx2layers' in obj.keys():
                    sx_mat_objs.append(obj)

            sx_mats = []
            for obj in sx_mat_objs:
                utils.sort_stack_indices(obj)
                if obj.active_material is not None:
                    sx_mat = bpy.data.materials.get(obj.active_material.name)
                    obj.data.materials.clear()
                    if ('SX2Material_' + obj.name) == sx_mat.name:
                        sx_mats.append(sx_mat)

            if len(sx_mats) > 0:
                for sx_mat in sx_mats:
                    sx_mat.user_clear()
                    bpy.data.materials.remove(sx_mat)

            bpy.context.view_layer.update()

            setup.create_sx2material(sx_mat_objs)

            for obj in sx_mat_objs:
                if len(obj.sx2layers) > 0:
                    obj.active_material = bpy.data.materials['SX2Material_' + obj.name]


    # TODO: Update to sx2 parameters
    def reset_scene(self):
        sxglobals.refresh_in_progress = True
        objs = bpy.context.scene.objects
        scene = bpy.context.scene.sx2

        for obj in objs:
            removeColList = []
            removeUVList = []
            if obj.type == 'MESH':
                for vertexColor in obj.data.attributes:
                    if 'VertexColor' in vertexColor.name:
                        removeColList.append(vertexColor.name)
                for uvLayer in obj.data.uv_layers:
                    if 'UVSet' in uvLayer.name:
                        removeUVList.append(uvLayer.name)

            sx2layers = len(obj.sx2layers.keys())
            for i in range(sx2layers):
                obj.sx2layers.remove(0)

            obj.sx2.selectedlayer = 1

            for vertexColor in removeColList:
                obj.data.attributes.remove(obj.data.attributes[vertexColor])
            for uvLayer in removeUVList:
                obj.data.uv_layers.remove(obj.data.uv_layers[uvLayer])

        bpy.data.materials.remove(bpy.data.materials['SXMaterial'])

        sxglobals.copyLayer = None
        sxglobals.listItems = []
        sxglobals.listIndices = {}
        sxglobals.prevShadingMode = 'FULL'
        sxglobals.composite = False
        sxglobals.paletteDict = {}
        sxglobals.masterPaletteArray = []
        sxglobals.materialArray = []
        for value in sxglobals.layerInitArray:
            value[1] = False
        scene.eraseuvs = True

        sxglobals.refresh_in_progress = False


    def __del__(self):
        print('SX Tools: Exiting setup')


# ------------------------------------------------------------------------
#    Update Functions
# ------------------------------------------------------------------------
# Return value eliminates duplicates
def mesh_selection_validator(self, context):
    mesh_objs = []
    for obj in context.view_layer.objects.selected:
        if obj.type == 'MESH' and obj.hide_viewport is False:
            mesh_objs.append(obj)
        elif obj.type == 'ARMATURE':
            all_children = utils.find_children(obj, recursive=True)
            for child in all_children:
                if child.type == 'MESH':
                    mesh_objs.append(child)

    return list(set(mesh_objs))


def layer_validator(self, context):
    objs = mesh_selection_validator(self, context)
    if len(objs) > 1:
        for obj in objs:
            if obj.sx2layers.keys() != objs[0].sx2layers.keys():
                return False
        return True
    else:
        return True


def refresh_swatches(self, context):
    # if not sxglobals.refresh_in_progress:
    #     sxglobals.refresh_in_progress = True

    scene = context.scene.sx2
    objs = mesh_selection_validator(self, context)
    # mode = objs[0].sx2.shadingmode

    utils.mode_manager(objs, set_mode=True, mode_id='refresh_swatches')

    if len(objs) > 0:
        for obj in objs:
            if len(obj.sx2layers) > 0:
                layer = obj.sx2layers[obj.sx2.selectedlayer]
                colors = utils.find_colors_by_frequency([obj, ], layer, 8)

                # Refresh SX Tools UI to latest selection
                # 1) Update layer HSL elements
                hArray = []
                sArray = []
                lArray = []
                for color in colors:
                    hsl = convert.rgb_to_hsl(color)
                    hArray.append(hsl[0])
                    sArray.append(hsl[1])
                    lArray.append(hsl[2])
                hue = max(hArray)
                sat = max(sArray)
                lightness = max(lArray)

                sxglobals.hsl_update = True
                obj.sx2.huevalue = hue
                obj.sx2.saturationvalue = sat
                obj.sx2.lightnessvalue = lightness
                sxglobals.hsl_update = False

        # 2) Update layer palette elements
        layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
        colors = utils.find_colors_by_frequency(objs, layer, 8)
        for i, pcol in enumerate(colors):
            palettecolor = (pcol[0], pcol[1], pcol[2], 1.0)
            setattr(scene, 'layerpalette' + str(i + 1), palettecolor)

        # 3) update tool palettes
        if scene.toolmode == 'COL':
            update_fillcolor(self, context)
        elif scene.toolmode == 'PAL':
            layers.color_layers_to_values(objs)
        # elif (scene.toolmode == 'MAT'):
        #     layers.material_layers_to_values(objs)

    # Verify selectionMonitor is running
    if not sxglobals.modal_status:
        setup.start_modal()

    utils.mode_manager(objs, revert=True, mode_id='refresh_swatches')
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
        if objs[0].mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        if obj.sx2.shadingmode == 'FULL':
            for layer in obj.sx2layers:
                if layer.paletted:
                    color = getattr(scene, 'newpalette'+str(layer.palette_index))
                    bpy.data.materials['SX2Material_'+obj.name].node_tree.nodes['PaletteColor' + str(layer.name)].outputs[0].default_value = color[:]

    # refresh_swatches(self, context)


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

        if sxglobals.mode == 'EDIT':
            modecolor = utils.find_colors_by_frequency(objs, objs[0].sx2layers[layer_ids[index]], 1)[0]
        else:
            modecolor = utils.find_colors_by_frequency(objs, objs[0].sx2layers[layer_ids[index]], 1, masklayer=layer)[0]

        if not utils.color_compare(modecolor, pbr_values[index]):
            if sxglobals.mode == 'EDIT':
                tools.apply_tool(objs, objs[0].sx2layers[layer_ids[index]], color=pbr_values[index])
            else:
                tools.apply_tool(objs, objs[0].sx2layers[layer_ids[index]], masklayer=objs[0].sx2layers[layer_ids[0]], color=pbr_values[index])

            setattr(scene, 'newmaterial' + str(index), pbr_values[index])

        utils.mode_manager(objs, revert=True, mode_id='update_material_layer')
        sxglobals.mat_update = False
        refresh_swatches(self, context)


# TODO: This only works correctly on multi-object selection with identical layersets
def update_selected_layer(self, context):
    objs = mesh_selection_validator(self, context)
    for obj in objs:
        if len(obj.sx2layers) > 0:
            obj.data.attributes.active_color = obj.data.attributes[obj.sx2layers[obj.sx2.selectedlayer].color_attribute]

        # utils.sort_stack_indices(obj)

    areas = bpy.context.workspace.screens[0].areas
    shading_types = ['MATERIAL', 'RENDERED']  # 'WIREFRAME' 'SOLID' 'MATERIAL' 'RENDERED'
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                if space.shading.type not in shading_types:
                    space.shading.type = 'MATERIAL'

    if objs[0].sx2.shadingmode != 'FULL':
        setup.update_sx2material(context)

    if 'SXToolMaterial' not in bpy.data.materials:
        setup.create_sxtoolmaterial()

    refresh_swatches(self, context)

    # Verify selectionMonitor and keyMonitor are running
    if not sxglobals.modal_status:
        setup.start_modal()


def update_material_props(self, context):
    objs = mesh_selection_validator(self, context)
    for obj in objs:
        if len(obj.sx2layers) > 0:
            color_stack = []
            material_stack = []
            for layer in obj.sx2layers:
                if layer.layer_type == 'COLOR':
                    color_stack.append((layer.index, layer.opacity))
                elif layer.layer_type != 'CMP':
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
                setattr(obj.sx2, 'layer_opacity_list_' + str(i), opacity_vector)

            vectors = []
            for i in range(0, len(mat_list), 3):
                vectors.append(mat_list[i:i+3])

            for i, opacity_vector in enumerate(vectors):
                while len(opacity_vector) < 3:
                    opacity_vector.append(1.0)
                setattr(obj.sx2, 'material_opacity_list_' + str(i), opacity_vector)

            if 'SX2Material_'+obj.name in bpy.data.materials:
                bpy.data.materials['SX2Material_'+obj.name].blend_method = 'OPAQUE'
                bpy.data.materials['SX2Material_'+obj.name].use_backface_culling = False

                # for layer in obj.sx2layers:
                #     if (layer.index == 0) and (layer.opacity < 1.0):
                #         bpy.data.materials['SX2Material_'+obj.name].blend_method = 'BLEND'
                #         bpy.data.materials['SX2Material_'+obj.name].use_backface_culling = True


def update_modifiers(self, context, prop):
    update_obj_props(self, context, prop)
    objs = mesh_selection_validator(self, context)
    if len(objs) > 0:
        if prop == 'modifiervisibility':
            for obj in objs:
                obj.data.use_auto_smooth = True
                if 'sxMirror' in obj.modifiers:
                    obj.modifiers['sxMirror'].show_viewport = obj.sx2.modifiervisibility
                if 'sxSubdivision' in obj.modifiers:
                    if (obj.sx2.subdivisionlevel == 0):
                        obj.modifiers['sxSubdivision'].show_viewport = False
                    else:
                        obj.modifiers['sxSubdivision'].show_viewport = obj.sx2.modifiervisibility
                if 'sxDecimate' in obj.modifiers:
                    if (obj.sx2.subdivisionlevel == 0.0) or (obj.sx2.decimation == 0.0):
                        obj.modifiers['sxDecimate'].show_viewport = False
                    else:
                        obj.modifiers['sxDecimate'].show_viewport = obj.sx2.modifiervisibility
                if 'sxDecimate2' in obj.modifiers:
                    if (obj.sx2.subdivisionlevel == 0.0) or (obj.sx2.decimation == 0.0):
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
                    tiler['Input_1'] = obj.sx2.tile_offset
                    tiler['Input_2'] = obj.sx2.tile_neg_x
                    tiler['Input_3'] = obj.sx2.tile_pos_x
                    tiler['Input_4'] = obj.sx2.tile_neg_y
                    tiler['Input_5'] = obj.sx2.tile_pos_y
                    tiler['Input_6'] = obj.sx2.tile_neg_z
                    tiler['Input_7'] = obj.sx2.tile_pos_z

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
                    if obj.sx2.subdivisionlevel == 0:
                        obj.modifiers['sxDecimate'].show_viewport = False
                    else:
                        obj.modifiers['sxDecimate'].show_viewport = obj.sx2.modifiervisibility
                if 'sxDecimate2' in obj.modifiers:
                    if obj.sx2.subdivisionlevel == 0:
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

        elif prop == 'weightednormals':
            for obj in objs:
                if 'sxWeightedNormal' in obj.modifiers:
                    obj.modifiers['sxWeightedNormal'].show_viewport = obj.sx2.weightednormals


def adjust_hsl(self, context, hslmode):
    if not sxglobals.hsl_update:
        objs = mesh_selection_validator(self, context)

        if len(objs) > 0:
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
    if len(objs) > 0:
        active = context.view_layer.objects.active
        category_data = sxglobals.categoryDict[sxglobals.preset_lookup[objs[0].sx2.category]]

        for obj in objs:
            obj.select_set(False)

        for obj in objs:
            obj.select_set(True)
            context.view_layer.objects.active = obj

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
            obj.sx2.roughnessoverride = bool(category_data['roughness_override'])
            obj.sx2.roughness0 = category_data['palette_roughness'][0]
            obj.sx2.roughness1 = category_data['palette_roughness'][1]
            obj.sx2.roughness2 = category_data['palette_roughness'][2]
            obj.sx2.roughness3 = category_data['palette_roughness'][3]
            obj.sx2.roughness4 = category_data['palette_roughness'][4]
            obj.sx2.selectedlayer = 1

            # bpy.data.materials['SX2Material_' + obj.name].blend_method = category_data['blend']
            # if category_data['blend'] == 'BLEND':
            #     bpy.data.materials['SX2Material_' + obj.name].use_backface_culling = True
            # else:
            #     bpy.data.materials['SX2Material_' + obj.name].use_backface_culling = False

            utils.sort_stack_indices(obj)
            context.view_layer.objects.active = None
            obj.select_set(False)

        for obj in objs:
            obj.select_set(True)

        context.view_layer.objects.active = active


def load_ramp(self, context):
    rampName = sxglobals.preset_lookup[context.scene.sx2.ramplist]
    ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['ColorRamp'].color_ramp
    temp_dict = sxglobals.ramp_dict[rampName]

    ramp.color_mode = temp_dict['mode']
    ramp.interpolation = temp_dict['interpolation']
    ramp.hue_interpolation = temp_dict['hue_interpolation']

    rampLength = len(ramp.elements)
    for i in range(rampLength-1):
        ramp.elements.remove(ramp.elements[0])

    for i, tempElement in enumerate(temp_dict['elements']):
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

    if status1 and status2 and status3 and status4:
        message_box('Libraries loaded successfully')
        sxglobals.librariesLoaded = True


# Fillcolor is automatically converted to grayscale on specific material layers
def update_fillcolor(self, context):
    scene = context.scene.sx2
    objs = mesh_selection_validator(self, context)
    layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

    gray_types = ['MET', 'RGH', 'OCC', 'TRN']
    if layer.layer_type in gray_types:
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


def update_curvature_selection(self, context):
    if not sxglobals.curvature_update:
        sxglobals.curvature_update = True

        objs = mesh_selection_validator(self, context)
        sel_mode = context.tool_settings.mesh_select_mode[:]
        limitvalue = context.scene.sx2.curvaturelimit
        tolerance = context.scene.sx2.curvaturetolerance
        scene = context.scene.sx2
        normalize = scene.curvaturenormalize
        scene.curvaturenormalize = True
        context.tool_settings.mesh_select_mode = (True, False, False)

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        for obj in objs:
            vert_curv_dict = generate.curvature_list(obj, returndict=True)
            mesh = obj.data

            for vert in mesh.vertices:
                if math.isclose(limitvalue, vert_curv_dict[vert.index], abs_tol=tolerance):
                    vert.select = True
                else:
                    vert.select = False

        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        context.tool_settings.mesh_select_mode = sel_mode
        scene.curvaturenormalize = normalize
        sxglobals.curvature_update = False


def update_obj_props(self, context, prop):
    if not sxglobals.refresh_in_progress:
        sxglobals.refresh_in_progress = True
        objs = mesh_selection_validator(self, context)
        value = getattr(objs[0].sx2, prop)
        for obj in objs:
            if getattr(obj.sx2, prop) != value:
                setattr(obj.sx2, prop, value)

        if prop == 'shadingmode':
            setup.update_sx2material(context)

        elif prop == 'selectedlayer':
            update_selected_layer(self, context)

        elif prop == 'smoothangle':
            smooth_angle = math.radians(objs[0].sx2.smoothangle)
            for obj in objs:
                obj.data.use_auto_smooth = True
                if obj.data.auto_smooth_angle != smooth_angle:
                    obj.data.auto_smooth_angle = smooth_angle

        elif prop == 'staticvertexcolors':
            for obj in objs:
                obj['staticVertexColors'] = int(objs[0].sx2.staticvertexcolors)

        elif prop == 'category':
            load_category(self, context)

        sxglobals.refresh_in_progress = False


def update_layer_props(self, context, prop):
    objs = mesh_selection_validator(self, context)
    for obj in objs:
        if getattr(obj.sx2layers[self.name], prop) != getattr(objs[0].sx2layers[self.name], prop):
            setattr(obj.sx2layers[self.name], prop, getattr(objs[0].sx2layers[self.name], prop))

    mat_props = ['paletted', 'visibility', 'blend_mode']
    if prop in mat_props:
        setup.update_sx2material(context)
    elif prop == 'opacity':
        update_material_props(self, context)


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
        # validate.validate_objects(objs)
        root_group.select_set(True)

        export.create_sxcollection()


@persistent
def load_post_handler(dummy):
    sxglobals.layer_stack_dict.clear()
    sxglobals.librariesLoaded = False

    if bpy.data.scenes['Scene'].sx2.rampmode == '':
        bpy.data.scenes['Scene'].sx2.rampmode = 'X'

    active = bpy.context.view_layer.objects.active
    for obj in bpy.data.objects:
        bpy.context.view_layer.objects.active = obj

        if (len(obj.sx2.keys()) > 0):
            if obj.sx2.hardmode == '':
                obj.sx2.hardmode = 'SHARP'

    bpy.context.view_layer.objects.active = active

    setup.start_modal()


# Update revision IDs and save in asset catalogue
@persistent
def save_pre_handler(dummy):
    for obj in bpy.data.objects:
        if (len(obj.sx2.keys()) > 0):
            obj['sxToolsVersion'] = 'SX Tools for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))
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
    if cost == '0':
        s = bpy.context.scene.statistics(bpy.context.view_layer)
        cost = s.split("Tris:")[1].split(' ')[0].replace(',', '')


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

    selectedlayer: bpy.props.IntProperty(
        name='Selected Layer',
        min=0,
        default=0,
        update=lambda self, context: update_obj_props(self, context, 'selectedlayer'))

    shadingmode: bpy.props.EnumProperty(
        name='Shading Mode',
        description='Full: Composited shading with all channels enabled\nDebug: Display selected layer only, with alpha as black\nAlpha: Display layer alpha or UV values in grayscale',
        items=[
            ('FULL', 'Full', ''),
            ('DEBUG', 'Debug', ''),
            ('ALPHA', 'Alpha', '')],
        default='FULL',
        update=lambda self, context: update_obj_props(self, context, 'shadingmode'))

    palettedshading: bpy.props.FloatProperty(
        name='Paletted Shading',
        description='Enable palette in SXMaterial',
        min=0.0,
        max=1.0,
        default=0.0,
        update=lambda self, context: update_obj_props(self, context, 'shadingmode'))

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
        items=lambda self, context: dict_lister(self, context, sxglobals.categoryDict),
        update=lambda self, context: update_obj_props(self, context, 'category'))

    staticvertexcolors: bpy.props.EnumProperty(
        name='Vertex Color Processing Mode',
        description='Choose how to export vertex colors to a game engine\nStatic bakes all color layers to VertexColor0\nPaletted leaves overlays in alpha channels',
        items=[
            ('1', 'Static', ''),
            ('0', 'Paletted', '')],
        default='1',
        update=lambda self, context: update_obj_props(self, context, 'staticvertexcolors'))

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

    lodmeshes: bpy.props.BoolProperty(
        name='Generate LOD Meshes',
        description='NOTE: Subdivision and Bevel settings affect the end result',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'lodmeshes'))

    pivotmode: bpy.props.EnumProperty(
        name='Pivot Mode',
        description='Auto pivot placement mode',
        items=[
            ('OFF', 'No Change', ''),
            ('MASS', 'Center of Mass', ''),
            ('BBOX', 'Bbox Center', ''),
            ('ROOT', 'Bbox Base', ''),
            ('ORG', 'Origin', ''),
            ('PAR', 'Parent', '')],
        default='OFF',
        update=lambda self, context: update_obj_props(self, context, 'pivotmode'))

    collideroffset: bpy.props.BoolProperty(
        name='Collision Mesh Auto-Offset',
        default=False,
        update=lambda self, context: update_obj_props(self, context, 'collideroffset'))

    collideroffsetfactor: bpy.props.FloatProperty(
        name='Auto-Offset Factor',
        min=0.0,
        max=1.0,
        default=0.25,
        update=lambda self, context: update_obj_props(self, context, 'collideroffsetfactor'))

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
            ('NSE', 'Noise', ''),
            ('CRV', 'Curvature', ''),
            ('OCC', 'Ambient Occlusion', ''),
            ('THK', 'Mesh Thickness', ''),
            ('DIR', 'Directional', ''),
            ('LUM', 'Luminance Remap', ''),
            ('PAL', 'Palette', ''),
            ('MAT', 'Material', '')],
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
            ('ADD', 'Additive', ''),
            ('MUL', 'Multiply', ''),
            ('OVR', 'Overlay', '')],
        default='ALPHA')

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
        update=load_ramp)

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
        max=5000,
        step=50,
        default=500)

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

    curvaturenormalize: bpy.props.BoolProperty(
        name='Normalize Curvature',
        description='Normalize convex and concave ranges\nfor improved artistic control',
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
            ('CMP', 'Composite', '')],
        default='COLOR')

    export_type: bpy.props.EnumProperty(
        name='Export Type',
        items=[
            ('COLOR', 'Color', ''),
            ('UV', 'UV', ''),
            ('UV4', 'UV4', '')],
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
            ('ADD', 'Additive', ''),
            ('MUL', 'Multiply', ''),
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
        min = 0,
        max = 4,
        default=0,
        update=lambda self, context: update_paletted_layers(self, context, None))


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

        if len(objs) > 0:
            if not layer_validator(self, context):
                col = layout.column()
                col.label(text='Mismatching layers sets!')
                col.label(text='Multi-editing requires')
                col.label(text='objects with identical layer sets.')
            else:
                obj = objs[0]
                mode = obj.mode
                sx2 = obj.sx2
                scene = context.scene.sx2

                if len(obj.sx2layers) == 0:
                    layer = None
                else:
                    layer = obj.sx2layers[sx2.selectedlayer]

                row_shading = layout.row()
                row_shading.prop(sx2, 'shadingmode', expand=True)
                row_shading.operator("wm.url_open", text='', icon='URL').url = 'https://www.notion.so/SX-Tools-for-Blender-Documentation-9ad98e239f224624bf98246822a671a6'

                # Layer Controls -----------------------------------------------
                box_layer = layout.box()
                row_layer = box_layer.row()

                if layer is None:
                    box_layer.enabled = False
                    row_layer.label(text='No layers')
                elif not layer_validator(self, context):
                    box_layer.enabled = False
                    row_layer.label(text='Mismatching layers sets!')
                else:
                    row_layer.prop(
                        scene, 'expandlayer',
                        icon='TRIA_DOWN' if scene.expandlayer else 'TRIA_RIGHT',
                        icon_only=True, emboss=False)

                    split_basics = row_layer.split(factor=0.3)
                    split_basics.prop(layer, 'blend_mode', text='')
                    split_basics.prop(layer, 'opacity', slider=True, text='Layer Opacity')

                    if scene.expandlayer:
                        if obj.mode == 'OBJECT':
                            hue_text = 'Layer Hue'
                            saturation_text = 'Layer Saturation'
                            lightness_text = 'Layer Lightness'
                        else:
                            hue_text = 'Selection Hue'
                            saturation_text = 'Selection Saturation'
                            lightness_text = 'Selection Lightness'

                        col_hsl = box_layer.column(align=True)
                        row_hue = col_hsl.row(align=True)
                        row_hue.prop(sx2, 'huevalue', slider=True, text=hue_text)
                        row_sat = col_hsl.row(align=True)
                        row_sat.prop(sx2, 'saturationvalue', slider=True, text=saturation_text)
                        row_lightness = col_hsl.row(align=True)
                        row_lightness.prop(sx2, 'lightnessvalue', slider=True, text=lightness_text)

                        gray_channels = ['OCC', 'MET', 'RGH', 'TRN']
                        if layer.layer_type in gray_channels:
                            row_hue.enabled = False
                            row_sat.enabled = False

                row_palette = layout.row(align=True)
                for i in range(8):
                    row_palette.prop(scene, 'layerpalette' + str(i+1), text='')

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
                copy_text = 'Copy'
                clr_text = 'Clear'
                paste_text = 'Paste'
                sel_text = 'Select Mask'

                if scene.shift:
                    clr_text = 'Reset All'
                    paste_text = 'Paste'
                    sel_text = 'Select Inverse'
                elif scene.alt:
                    paste_text = 'Paste'
                elif scene.ctrl:
                    clr_text = 'Flatten'
                    sel_text = 'Select Color'
                    paste_text = 'Paste Alpha'

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
                    icon='TRIA_DOWN' if scene.expandfill else 'TRIA_RIGHT',
                    icon_only=True, emboss=False)
                row_fill.prop(scene, 'toolmode', text='')
                if scene.toolmode != 'PAL' and scene.toolmode != 'MAT':
                    row_fill.operator('sx2.applytool', text='Apply')

                # Color Tool --------------------------------------------------------
                if scene.toolmode == 'COL':
                    split1_fill = box_fill.split(factor=0.33)
                    if (layer is None) or (layer.layer_type == 'COLOR') or (layer.layer_type == 'EMI') or (layer.layer_type == 'SSS'):
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
                elif scene.toolmode == 'OCC' or scene.toolmode == 'THK':
                    if scene.expandfill:
                        col_fill = box_fill.column(align=True)
                        if scene.toolmode == 'OCC' or scene.toolmode == 'THK':
                            col_fill.prop(scene, 'occlusionrays', slider=True, text='Ray Count')
                        if scene.toolmode == 'OCC':
                            col_fill.prop(scene, 'occlusionblend', slider=True, text='Local/Global Mix')
                            col_fill.prop(scene, 'occlusiondistance', slider=True, text='Ray Distance')
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
                        col_fill.prop(scene, 'curvaturenormalize', text='Normalize')
                        col_fill.prop(sx2, 'tiling', text='Tiling Object')

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
                        box_fill.template_color_ramp(bpy.data.materials['SXToolMaterial'].node_tree.nodes['ColorRamp'], 'color_ramp', expand=True)
                        if mode == 'OBJECT':
                            box_fill.prop(scene, 'rampbbox', text='Use Combined Bounding Box')

                # Luminance Remap Tool -----------------------------------------------
                elif scene.toolmode == 'LUM':
                    if scene.expandfill:
                        row3_fill = box_fill.row(align=True)
                        row3_fill.prop(scene, 'ramplist', text='')
                        row3_fill.operator('sx2.addramp', text='', icon='ADD')
                        row3_fill.operator('sx2.delramp', text='', icon='REMOVE')
                        box_fill.template_color_ramp(bpy.data.materials['SXToolMaterial'].node_tree.nodes['ColorRamp'], 'color_ramp', expand=True)

                # Master Palettes -----------------------------------------------
                elif scene.toolmode == 'PAL':
                    palettes = context.scene.sx2palettes
                    col_preview = box_fill.column(align=True)
                    col_preview.prop(sx2, 'palettedshading', slider=True)
                    col_preview.operator('sx2.applypalette', text='Apply Palette')
                    row_newpalette = box_fill.row(align=True)
                    for i in range(5):
                        row_newpalette.prop(scene, 'newpalette'+str(i), text='')
                    row_newpalette.operator('sx2.addpalette', text='', icon='ADD')

                    if scene.expandfill:
                        row_lib = box_fill.row()
                        row_lib.prop(
                            scene, 'expandpal',
                            icon='TRIA_DOWN' if scene.expandpal else 'TRIA_RIGHT',
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
                    if sx2.selectedlayer > 7:
                        if scene.expandfill:
                            box_fill.label(text='Select Layer 1-7 to apply PBR materials')
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
                                icon='TRIA_DOWN' if scene.expandmat else 'TRIA_RIGHT',
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
                    box_crease = layout.box()
                    row_crease = box_crease.row()
                    row_crease.prop(
                        scene, 'expandcrease',
                        icon='TRIA_DOWN' if scene.expandcrease else 'TRIA_RIGHT',
                        icon_only=True, emboss=False)
                    row_crease.prop(scene, 'creasemode', expand=True)
                    if scene.creasemode != 'SDS':
                        if scene.expandcrease:
                            row_sets = box_crease.row(align=True)
                            for i in range(4):
                                setbutton = row_sets.operator('sx2.setgroup', text=str(25*(i+1))+'%')
                                setbutton.setmode = scene.creasemode
                                setbutton.setvalue = 0.25 * (i+1)
                            col_sel = box_crease.column(align=True)
                            col_sel.prop(scene, 'curvaturelimit', slider=True, text='Curvature Selector')
                            col_sel.prop(scene, 'curvaturetolerance', slider=True, text='Curvature Tolerance')
                            col_sets = box_crease.column(align=True)
                            if (bpy.context.tool_settings.mesh_select_mode[0]) and (scene.creasemode == 'CRS'):
                                clear_text = 'Clear Vertex Weights'
                            else:
                                clear_text = 'Clear Edge Weights'
                            setbutton = col_sets.operator('sx2.setgroup', text=clear_text)
                            setbutton.setmode = scene.creasemode
                            setbutton.setvalue = -1.0
                    elif scene.creasemode == 'SDS':
                        if scene.expandcrease:
                            row_mod1 = box_crease.row()
                            row_mod1.prop(
                                scene, 'expandmirror',
                                icon='TRIA_DOWN' if scene.expandmirror else 'TRIA_RIGHT',
                                icon_only=True, emboss=False)
                            row_mod1.label(text='Mirror Modifier Settings')
                            if scene.expandmirror:
                                row_mirror = box_crease.row(align=True)
                                row_mirror.label(text='Axis:')
                                row_mirror.prop(sx2, 'xmirror', text='X', toggle=True)
                                row_mirror.prop(sx2, 'ymirror', text='Y', toggle=True)
                                row_mirror.prop(sx2, 'zmirror', text='Z', toggle=True)
                                row_mirrorobj = box_crease.row()
                                row_mirrorobj.prop_search(sx2, 'mirrorobject', context.scene, 'objects')

                            row_mod2 = box_crease.row()
                            row_mod2.prop(
                                scene, 'expandbevel',
                                icon='TRIA_DOWN' if scene.expandbevel else 'TRIA_RIGHT',
                                icon_only=True, emboss=False)
                            row_mod2.label(text='Bevel Modifier Settings')
                            if scene.expandbevel:
                                col2_sds = box_crease.column(align=True)
                                col2_sds.prop(scene, 'autocrease', text='Auto Hard-Crease Bevels')
                                split_sds = col2_sds.split()
                                split_sds.label(text='Max Crease Mode:')
                                split_sds.prop(sx2, 'hardmode', text='')
                                split2_sds = col2_sds.split()
                                split2_sds.label(text='Bevel Type:')
                                split2_sds.prop(sx2, 'beveltype', text='')
                                col2_sds.prop(sx2, 'bevelsegments', text='Bevel Segments')
                                col2_sds.prop(sx2, 'bevelwidth', text='Bevel Width')

                            col3_sds = box_crease.column(align=True)
                            col3_sds.prop(sx2, 'subdivisionlevel', text='Subdivision Level')
                            col3_sds.prop(sx2, 'smoothangle', text='Normal Smoothing Angle')
                            col3_sds.prop(sx2, 'weldthreshold', text='Weld Threshold')
                            col3_sds.prop(sx2, 'weightednormals', text='Weighted Normals')
                            if obj.sx2.subdivisionlevel > 0:
                                col3_sds.prop(sx2, 'decimation', text='Decimation Limit Angle')
                                col3_sds.label(text='Selection Tri Count: ' + modifiers.calculate_triangles(objs))
                            # col3_sds.separator()
                            col4_sds = box_crease.column(align=True)
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
                    box_export = layout.box()          
                    row_export = box_export.row()
                    row_export.prop(
                        scene, 'expandexport',
                        icon='TRIA_DOWN' if scene.expandexport else 'TRIA_RIGHT',
                        icon_only=True, emboss=False)
                    row_export.prop(scene, 'exportmode', expand=True)
                    if scene.exportmode == 'MAGIC':
                        if scene.expandexport:
                            col_export = box_export.column(align=True)
                            row_cat = col_export.row(align=True)
                            row_cat.label(text='Category:')
                            row_cat.prop(sx2, 'category', text='')

                            col_export.separator()
                            col_export.prop(sx2, 'roughnessoverride', text='Override Palette Roughness', toggle=True)
                            if obj.sx2.roughnessoverride:
                                row_override = col_export.row(align=True)
                                row_override.prop(sx2, 'roughness0', text='Palette Color 0 Roughness')
                                row_override.prop(sx2, 'roughness1', text='Palette Color 1 Roughness')
                                row_override.prop(sx2, 'roughness2', text='Palette Color 2 Roughness')
                                row_override.prop(sx2, 'roughness3', text='Palette Color 3 Roughness')
                                row_override.prop(sx2, 'roughness4', text='Palette Color 4 Roughness')

                            row_static = col_export.row(align=True)
                            row_static.label(text='Vertex Colors:')
                            row_static.prop(sx2, 'staticvertexcolors', text='')

                            row_quality = col_export.row(align=True)
                            row_quality.label(text='Bake Detail:')
                            row_quality.prop(scene, 'exportquality', text='')

                            row_pivot = col_export.row(align=True)
                            row_pivot.label(text='Auto-Pivot:')
                            row_pivot.prop(sx2, 'pivotmode', text='')

                            row_separate = col_export.row(align=True)
                            row_separate.label(text='Smart Separate on Export:')
                            row_separate.prop(sx2, 'smartseparate', text='', toggle=False)

                            row_lod = col_export.row(align=True)
                            row_lod.label(text='Generate LOD Meshes:')
                            row_lod.prop(sx2, 'lodmeshes', text='', toggle=False)

                            if hasattr(bpy.types, bpy.ops.object.vhacd.idname()):
                                col_export.separator()
                                col_export.prop(scene, 'exportcolliders', text='Generate Mesh Colliders (V-HACD)')
                                if scene.exportcolliders:
                                    box_colliders = box_export.box()
                                    col_collideroffset = box_colliders.column(align=True)
                                    col_collideroffset.prop(sx2, 'collideroffset', text='Auto-Shrink Collision Mesh')
                                    col_collideroffset.prop(sx2, 'collideroffsetfactor', text='Shrink Factor', slider=True)
                                    row_colliders = box_colliders.row()
                                    row_colliders.prop(
                                        scene, 'expandcolliders',
                                        icon='TRIA_DOWN' if scene.expandcolliders else 'TRIA_RIGHT',
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
                            if scene.shift:
                                col_export.operator('sx2.revertobjects', text='Revert to Control Cages')
                            else:
                                col_export.operator('sx2.macro', text='Magic Button')
                            if ('ExportObjects' in bpy.data.collections.keys()) and (len(bpy.data.collections['ExportObjects'].objects) > 0):
                                col_export.operator('sx2.removeexports', text='Remove LODs and Parts')

                    elif scene.exportmode == 'UTILS':
                        if scene.expandexport:
                            col_utils = box_export.column(align=False)
                            col_utils.operator('sx2.test_button', text='Composite and Export Atlas')
                            col_utils.operator('sx2.sxtosx2')
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
                            row_debug = box_export.row()
                            row_debug.prop(
                                scene, 'expanddebug',
                                icon='TRIA_DOWN' if scene.expanddebug else 'TRIA_RIGHT',
                                icon_only=True, emboss=False)
                            row_debug.label(text='Debug Tools')
                            if scene.expanddebug:
                                col_debug = box_export.column(align=True)
                                col_debug.operator('sx2.smart_separate', text='Debug: Smart Separate sxMirror')
                                col_debug.operator('sx2.create_sxcollection', text='Debug: Update SXCollection')
                                col_debug.operator('sx2.applymodifiers', text='Debug: Apply Modifiers')
                                col_debug.operator('sx2.generatemasks', text='Debug: Generate Masks')
                                col_debug.operator('sx2.createuv0', text='Debug: Create UVSet0')
                                col_debug.operator('sx2.generatelods', text='Debug: Create LOD Meshes')
                                col_debug.operator('sx2.resetscene', text='Debug: Reset scene (warning!)')

                    elif scene.exportmode == 'EXPORT':
                        if scene.expandexport:
                            if len(prefs.cataloguepath) > 0:
                                row_batchexport = box_export.row()
                                row_batchexport.prop(
                                    scene, 'expandbatchexport',
                                    icon='TRIA_DOWN' if scene.expandbatchexport else 'TRIA_RIGHT',
                                    icon_only=True, emboss=False)
                                row_batchexport.label(text='Batch Export Settings')
                                if scene.expandbatchexport:
                                    col_batchexport = box_export.column(align=True)
                                    col_cataloguebuttons = box_export.column(align=True)
                                    groups = utils.find_groups(objs, all_groups=True)
                                    if len(groups) == 0:
                                        col_batchexport.label(text='No Groups Selected')
                                    else:
                                        col_batchexport.label(text='Groups to Batch Process:')
                                        for group in groups:
                                            row_group = col_batchexport.row(align=True)
                                            row_group.prop(group.sx2, 'exportready', text='')
                                            row_group.label(text='  ' + group.name)

                                    col_cataloguebuttons.operator('sx2.catalogue_add', text='Add to Catalogue', icon='ADD')
                                    col_cataloguebuttons.operator('sx2.catalogue_remove', text='Remove from Catalogue', icon='REMOVE')
                                    export_ready_groups = False
                                    for group in groups:
                                        if group.sx2.exportready:
                                            export_ready_groups = True

                                    if not export_ready_groups:
                                        col_cataloguebuttons.enabled = False

                            col2_export = box_export.column(align=True)
                            col2_export.label(text='Export Folder:')
                            col2_export.prop(scene, 'exportfolder', text='')

                            split_export = box_export.split(factor=0.1)
                            split_export.operator('sx2.checklist', text='', icon='INFO')

                            if scene.shift:
                                exp_text = 'Export All'
                            else:
                                exp_text = 'Export Selected'
                            split_export.operator('sx2.exportfiles', text=exp_text)

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
            if sxglobals.librariesLoaded:
                col.label(text='Select a mesh to continue')
            else:
                col.label(text='Libraries not loaded')
                col.label(text='Check Add-on Preferences')
                if len(prefs.libraryfolder) > 0:
                    col.operator('sx2.loadlibraries', text='Reload Libraries')


class SXTOOLS2_UL_layerlist(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index, flt_flag):
        scene = context.scene.sx2
        objs = mesh_selection_validator(self, context)
        hide_icon = {False: 'HIDE_ON', True: 'HIDE_OFF'}
        lock_icon = {False: 'UNLOCKED', True: 'LOCKED'}
        paletted_icon = {False: 'RESTRICT_COLOR_OFF', True: 'RESTRICT_COLOR_ON'}

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row_item = layout.row(align=True)

            if objs[0].sx2.shadingmode == 'FULL':
                row_item.prop(item, 'visibility', text='', icon=hide_icon[item.visibility])
            else:
                row_item.label(icon=hide_icon[(utils.find_layer_index_by_name(objs[0], item.name) == objs[0].sx2.selectedlayer)])

            if sxglobals.mode == 'OBJECT':
                if scene.toolmode == 'PAL':
                    row_item.label(text='', icon='LOCKED')
                else:
                    row_item.prop(item, 'locked', text='', icon=lock_icon[item.locked])
            else:
                row_item.label(text='', icon='UNLOCKED')

            row_item.label(text='   ' + item.name + '   ')

            if ((item.layer_type == 'COLOR') or (item.layer_type == 'EMI') or (item.layer_type == 'SSS')) and (scene.toolmode == 'PAL'):
                row_item.prop(item, 'paletted', text='', icon=paletted_icon[item.paletted])
                row_item.prop(item, 'palette_index', text='')

        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text='', icon=hide_icon[item.visibility])


    # Called once to draw filtering/reordering options.
    def draw_filter(self, context, layout):
        pass


    def filter_items(self, context, data, propname):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            flt_flags = []
            flt_neworder = []
            flt_flags = [self.bitflag_filter_item] * len(objs[0].sx2layers)
            for layer in objs[0].sx2layers:
                flt_neworder.append(layer.index)

            return flt_flags, flt_neworder


class SXTOOLS2_preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    libraryfolder: bpy.props.StringProperty(
        name='Library Folder',
        description='Folder containing SX Tools data files\n(materials.json, palettes.json, gradients.json)',
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

    exportspace: bpy.props.EnumProperty(
        name='Color Space for Exports',
        description='Color space for exported vertex colors',
        items=[
            ('SRGB', 'sRGB', ''),
            ('LIN', 'Linear', '')],
        default='LIN')

    removelods: bpy.props.BoolProperty(
        name='Remove LOD Meshes After Export',
        description='Remove LOD meshes from the scene after exporting to FBX',
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
        layout_split4 = layout.split()
        layout_split4.label(text='FBX Export Color Space:')
        layout_split4.prop(self, 'exportspace', text='')
        layout_split5 = layout.split()
        layout_split5.label(text='LOD Mesh Preview Z-Offset')
        layout_split5.prop(self, 'lodoffset', text='')
        layout_split6 = layout.split()
        layout_split6.label(text='Clear LOD Meshes After Export')
        layout_split6.prop(self, 'removelods', text='')
        layout_split7 = layout.split()
        layout_split7.label(text='Reverse Smart Mirror Naming:')
        layout_split8 = layout_split7.split()
        layout_split8.prop(self, 'flipsmartx', text='X-Axis')
        layout_split8.prop(self, 'flipsmarty', text='Y-Axis')
        layout_split9 = layout.split()
        layout_split9.label(text='Use Absolute Paths')
        layout_split9.prop(self, 'absolutepaths', text='')
        layout_split10 = layout.split()
        layout_split10.label(text='Library Folder:')
        layout_split10.prop(self, 'libraryfolder', text='')
        layout_split11 = layout.split()
        layout_split11.label(text='Catalogue File (Optional):')
        layout_split11.prop(self, 'cataloguepath', text='')

        bpy.context.preferences.filepaths.use_relative_paths = not self.absolutepaths


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
            return context.active_object is not None


    def modal(self, context, event):
        if not context.area:
            print('Selection Monitor: Context Lost')
            sxglobals.modal_status = False
            return {'CANCELLED'}

        if (len(sxglobals.masterPaletteArray) == 0) or (len(sxglobals.materialArray) == 0) or (len(sxglobals.ramp_dict) == 0) or (len(sxglobals.categoryDict) == 0):
            sxglobals.librariesLoaded = False
            load_libraries(self, context)
            return {'PASS_THROUGH'}

        objs = mesh_selection_validator(self, context)
        if (len(objs) == 0) and (context.active_object is not None) and (context.object.mode == 'EDIT'):
            objs = context.objects_in_mode
            if len(objs) > 0:
                for obj in objs:
                    obj.select_set(True)
                context.view_layer.objects.active = objs[0]

        if len(objs) > 0:
            mode = objs[0].mode
            if mode != sxglobals.prev_mode:
                # print('selectionmonitor: mode change')
                sxglobals.prev_mode = mode
                sxglobals.mode = mode
                refresh_swatches(self, context)
                return {'PASS_THROUGH'}

            if (objs[0].mode == 'EDIT'):
                objs[0].update_from_editmode()
                mesh = objs[0].data
                selection = [None] * len(mesh.vertices)
                mesh.vertices.foreach_get('select', selection)
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
                        refresh_swatches(self, context)
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
            sxglobals.modal_status = False
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
        message_box('SPECIAL LAYERS:\nGradient1, Gradient2\nGrayscale layers that receive color from Palette Color 4 and 5\n(i.e. the primary colors of Layer 4 and 5)\n\nOcclusion, Metallic, Roughness, Transmission, Emission\nGrayscale layers, filled with the luminance of the selected color.')
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
        for i, categoryDict in enumerate(sxglobals.masterPaletteArray):
            for category in categoryDict:
                if category == self.paletteCategoryName.replace(" ", ""):
                    message_box('Palette Category Exists!')
                    found = True
                    break

        if not found:
            categoryName = self.paletteCategoryName.replace(" ", "")
            categoryEnum = categoryName.upper()
            categoryDict = {categoryName: {}}

            sxglobals.masterPaletteArray.append(categoryDict)
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

        for i, categoryDict in enumerate(sxglobals.masterPaletteArray):
            for category in categoryDict:
                if category == categoryName:
                    sxglobals.masterPaletteArray.remove(categoryDict)
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
        for i, categoryDict in enumerate(sxglobals.materialArray):
            for category in categoryDict:
                if category == self.materialCategoryName.replace(" ", ""):
                    message_box('Palette Category Exists!')
                    found = True
                    break

        if not found:
            categoryName = self.materialCategoryName.replace(" ", "")
            categoryEnum = categoryName.upper()
            categoryDict = {categoryName: {}}

            sxglobals.materialArray.append(categoryDict)
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

        for i, categoryDict in enumerate(sxglobals.materialArray):
            for category in categoryDict:
                if category == categoryName:
                    sxglobals.materialArray.remove(categoryDict)
                    del sxglobals.preset_lookup[categoryEnum]
                    break

        files.save_file('materials')
        files.load_file('materials')
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
            for categoryDict in sxglobals.masterPaletteArray:
                for i, category in enumerate(categoryDict.keys()):
                    if category.casefold() == context.scene.sx2.palettecategories.casefold():
                        categoryDict[category][paletteName] = [
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

        for i, categoryDict in enumerate(sxglobals.masterPaletteArray):
            if category in categoryDict:
                del categoryDict[category][paletteName]

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
        if materialName in context.scene.sx2materials:
            message_box('Material Name Already in Use!')
        else:
            for categoryDict in sxglobals.materialArray:
                for i, category in enumerate(categoryDict.keys()):
                    if category.casefold() == context.scene.sx2.materialcategories.casefold():
                        categoryDict[category][materialName] = [
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

        for i, categoryDict in enumerate(sxglobals.materialArray):
            if category in categoryDict:
                del categoryDict[category][materialName]

        files.save_file('materials')
        files.load_file('materials')
        return {'FINISHED'}


class SXTOOLS2_OT_applytool(bpy.types.Operator):
    bl_idname = 'sx2.applytool'
    bl_label = 'Apply Tool'
    bl_description = 'Applies the selected mode fill\nto the selected components or objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            if len(objs[0].sx2layers) == 0:
                layers.add_layer(objs)
                setup.update_sx2material(context)

            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            fillcolor = convert.srgb_to_linear(context.scene.sx2.fillcolor)

            if objs[0].mode == 'EDIT':
                context.scene.sx2.rampalpha = True

            if context.scene.sx2.toolmode == 'COL':
                tools.apply_tool(objs, layer, color=fillcolor)
                tools.update_recent_colors(fillcolor)
            else:
                tools.apply_tool(objs, layer)

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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].data.color_attributes) < 14:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        utils.mode_manager(objs, set_mode=True, mode_id='add_layer')
        layers.add_layer(objs)
        setup.update_sx2material(context)
        utils.mode_manager(objs, revert=True, mode_id='add_layer')
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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            utils.mode_manager(objs, set_mode=True, mode_id='del_layer')
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            layers.del_layer(objs, layer)
            setup.update_sx2material(context)
            utils.mode_manager(objs, revert=True, mode_id='del_layer')
            if len(objs[0].sx2layers) > 0:
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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if (layer.index < (len(sxglobals.layer_stack_dict.get(mesh_objs[0].name, [])) - 1)):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            idx = objs[0].sx2.selectedlayer
            updated = False
            for obj in objs:
                if len(obj.sx2layers) > 0:
                    utils.mode_manager(objs, set_mode=True, mode_id='layer_up')
                    new_index = obj.sx2layers[idx].index + 1 if obj.sx2layers[idx].index + 1 < len(obj.sx2layers) else obj.sx2layers[idx].index
                    obj.sx2layers[idx].index = utils.insert_layer_at_index(obj, obj.sx2layers[idx], new_index)
                    updated = True
                    utils.mode_manager(objs, revert=True, mode_id='layer_up')

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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if (layer.index > 0):
                enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            idx = objs[0].sx2.selectedlayer
            updated = False
            for obj in objs:
                if len(obj.sx2layers) > 0:
                    utils.mode_manager(objs, set_mode=True, mode_id='layer_down')
                    new_index = obj.sx2layers[idx].index - 1 if obj.sx2layers[idx].index -1 >= 0 else 0
                    obj.sx2layers[idx].index = utils.insert_layer_at_index(obj, obj.sx2layers[idx], new_index)
                    updated = True
                    utils.mode_manager(objs, revert=True, mode_id='layer_down')

            if updated:
                setup.update_sx2material(context)

        return {'FINISHED'}


class SXTOOLS2_OT_layer_props(bpy.types.Operator):
    bl_idname = 'sx2.layer_props'
    bl_label = 'Layer Properties'
    bl_description = 'Edit layer properties'
    bl_options = {'UNDO'}

    layer_name: bpy.props.StringProperty(
        name = 'Ramp Name')

    layer_type: bpy.props.EnumProperty(
        name = 'Layer Type',
        items = [
            ('COLOR', 'Color', ''),
            ('OCC', 'Occlusion', ''),
            ('MET', 'Metallic', ''),
            ('RGH', 'Roughness', ''),
            ('TRN', 'Transmission', ''),
            ('SSS', 'Subsurface', ''),
            ('EMI', 'Emission', ''),
            ('CMP', 'Composite', '')],
        default = 'COLOR')


    @classmethod
    def poll(cls, context):
        enabled = False
        objs = context.view_layer.objects.selected
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if (len(objs) > 0) and (len(objs[0].sx2layers) > 0):
            idx = objs[0].sx2.selectedlayer
            self.layer_name = objs[0].sx2layers[idx].name
            self.layer_type = objs[0].sx2layers[idx].layer_type

            return context.window_manager.invoke_props_dialog(self)
        else:
            return {'FINISHED'}


    def draw(self, context):
        layout = self.layout
        row_name = layout.row()
        row_name.prop(self, 'layer_name', text='Layer Name')
        row_type = layout.row()
        row_type.prop(self, 'layer_type', text='Layer Type')
        if self.layer_type != 'COLOR':
            row_name.enabled = False
        return None


    def execute(self, context):
        alpha_mats = ['OCC', 'MET', 'RGH', 'TRN']
        objs = mesh_selection_validator(self, context)
        for obj in objs:
            layer = obj.sx2layers[obj.sx2.selectedlayer]

            for sx2layer in obj.sx2layers:
                if (self.layer_type == sx2layer.layer_type) and (self.layer_type != 'COLOR'):
                    message_box('Material channel already exists!')
                    return {'FINISHED'}

            if (len(obj.data.color_attributes) == 14) and (layer.layer_type in alpha_mats) and (self.layer_type not in alpha_mats):
                message_box('Layer stack at max, delete a color layer and try again.')
                return {'FINISHED'}

            if len(obj.sx2layers) > 0:
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
                else:
                    name = self.layer_name

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
                if layer.layer_type in ['OCC', 'MET', 'RGH', 'TRN', 'CMP']:
                    layer.paletted = False
                else:
                    layer.paletted = True
                layer.default_color = sxglobals.default_colors[self.layer_type]

                # Create a color attribute for alpha materials if not in data
                if ('Alpha Materials' not in obj.data.color_attributes.keys()) and (layer.layer_type in alpha_mats):
                    obj.data.color_attributes.new(name='Alpha Materials', type='FLOAT_COLOR', domain='CORNER')
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
                        obj.data.color_attributes.new(name=layer.name, type='FLOAT_COLOR', domain='CORNER')
                        layer.color_attribute = layer.name
                        layers.set_layer(obj, source_colors, layer)

                utils.sort_stack_indices(obj)

        setup.update_sx2material(context)

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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if (layer.layer_type == 'COLOR') and (layer.index < (len(sxglobals.layer_stack_dict.get(mesh_objs[0].name, [])) - 1)):
                enabled = True
        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            baseLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            topLayer = utils.find_layer_by_stack_index(objs[0], baseLayer.index + 1)
            layers.merge_layers(objs, topLayer, baseLayer, topLayer)
            layers.del_layer(objs, baseLayer)

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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if layer.index > 0:
                nextLayer = utils.find_layer_by_stack_index(mesh_objs[0], layer.index - 1)
                if (nextLayer.layer_type == 'COLOR') and (layer.layer_type == 'COLOR'):
                    enabled = True
        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            topLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            baseLayer = utils.find_layer_by_stack_index(objs[0], topLayer.index - 1)
            layers.merge_layers(objs, topLayer, baseLayer, baseLayer)
            layers.del_layer(objs, topLayer)
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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        utils.mode_manager(objs, set_mode=True, mode_id='copy_layer')
        if len(objs) > 0:
            for obj in objs:
                colors = layers.get_layer(obj, obj.sx2layers[objs[0].sx2.selectedlayer])
                sxglobals.copy_buffer[obj.name] = generate.mask_list(obj, colors)
        utils.mode_manager(objs, revert=True, mode_id='copy_layer')
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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            for obj in objs:
                target_layer = obj.sx2layers[objs[0].sx2.selectedlayer]

                if event.ctrl:
                    mode = 'mask'
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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            utils.mode_manager(objs, set_mode=True, mode_id='clearlayers')
            if event.ctrl:
                for obj in objs:
                    comp_layers = utils.find_color_layers(obj)
                    layer_attributes = []
                    for layer in comp_layers:
                        layer_attributes.append((layer.color_attribute, layer.name))

                    layers.add_layer([obj, ], name='Flattened', layer_type='COLOR')
                    layers.blend_layers([obj, ], comp_layers, comp_layers[0], obj.sx2layers['Flattened'])

                    for attribute in layer_attributes:
                        obj.data.attributes.remove(obj.data.attributes[attribute[0]])
                        idx = utils.find_layer_index_by_name(obj, attribute[1])
                        obj.sx2layers.remove(idx)

                    utils.sort_stack_indices(obj)

                objs[0].sx2.selectedlayer = len(objs[0].sx2layers) - 1
                setup.update_sx2material(context)
            else:
                if event.shift:
                    layer = None
                else:
                    layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

                layers.clear_layers(objs, layer)

            utils.mode_manager(objs, revert=True, mode_id='clearlayers')
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
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            enabled = True

        return enabled


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            context.view_layer.objects.active = objs[0]

            if event.shift:
                inverse = True
            else:
                inverse = False

            if event.ctrl:
                color = context.scene.sx2.fillcolor
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
        if len(objs) > 0:
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
        if len(objs) > 0:
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
    bl_description = 'Applies the selected material to selected objects\nAlbedo color goes to the layer7\nmetallic and roughness values are automatically applied\nto the selected material channels'
    bl_options = {'UNDO'}

    label: bpy.props.StringProperty()


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            material = self.label
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            tools.apply_material(objs, layer, material)
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
        if len(objs) > 0:
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
        if len(objs) > 0:
            modifiers.apply_modifiers(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_removemodifiers(bpy.types.Operator):
    bl_idname = 'sx2.removemodifiers'
    bl_label = 'Remove Modifiers'
    bl_description = 'Remove SX Tools modifiers from selected objects\nand clear their settings'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
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
        if len(objs) > 0:
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

        if ((event is not None) and (event.shift)) or bpy.app.background:
            export.create_sxcollection()
            selected = bpy.data.collections['SXObjects'].all_objects
        else:
            selected = mesh_selection_validator(self, context)

        if len(selected) > 0:
            groups = utils.find_groups(selected)

            if context.scene.sx2.exportquality == 'LO':
                export.smart_separate(selected)
                export.create_sxcollection()

            # Make sure objects are in groups
            if ((event is not None) and (event.shift)) or bpy.app.background:
                selected = bpy.data.collections['SXObjects'].all_objects

            for obj in selected:
                if obj.parent is None:
                    obj.hide_viewport = False
                    if obj.type == 'MESH':
                        export.group_objects([obj, ])

            # Batch export only those that are marked ready
            if bpy.app.background:
                exportready_groups = []
                for group in groups:
                    if (group is not None) and group.sx2.exportready:
                        exportready_groups.append(group)
                groups = exportready_groups

            files.export_files(groups)

            if prefs.removelods:
                export.remove_exports()
            # sxglobals.composite = True
            # refresh_actives(self, context)
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
        all_children = utils.find_children(group, recursive=True)
        export.smart_separate(all_children)
        files.export_files([group, ])
        if prefs.removelods:
            export.remove_exports()
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
        if len(objs) > 0:
            export.set_pivots(objs, pivotmode, force=True)

        return {'FINISHED'}


class SXTOOLS2_OT_removeexports(bpy.types.Operator):
    bl_idname = 'sx2.removeexports'
    bl_label = 'Remove LODs and Separated Parts'
    bl_description = 'Deletes generated LOD meshes\nand smart separations'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        scene = bpy.context.scene.sx2
        export.remove_exports()
        scene.shadingmode = 'FULL'
        return {'FINISHED'}


class SXTOOLS2_OT_createuv0(bpy.types.Operator):
    bl_idname = 'sx2.createuv0'
    bl_label = 'Create UVSet0'
    bl_description = 'Checks if UVSet0 is missing\nand adds it at the top of the UV sets'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            export.create_uvset0(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_groupobjects(bpy.types.Operator):
    bl_idname = 'sx2.groupobjects'
    bl_label = 'Group Objects'
    bl_description = 'Groups objects under an empty.\nDefault pivot placed at the bottom center\nShift-click to place pivot in world origin'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        origin = event.shift
        if len(objs) > 0:
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

        if len(objs) > 0:
            for obj in objs:
                if '_LOD' not in obj.name:
                    org_objs.append(obj)

        if len(org_objs) > 0:
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
        scene = bpy.context.scene.sx2
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            utils.mode_manager(objs, set_mode=True, mode_id='revertobjects')
            export.revert_objects(objs)

            bpy.ops.object.shade_flat()
            scene.shadingmode = 'FULL'
            sxglobals.composite = True
            refresh_swatches(self, context)

            utils.mode_manager(objs, revert=True, mode_id='revertobjects')
        return {'FINISHED'}


class SXTOOLS2_OT_zeroverts(bpy.types.Operator):
    bl_idname = 'sx2.zeroverts'
    bl_label = 'Set Vertices to Zero'
    bl_description = 'Sets the mirror axis position of\nselected vertices to zero'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            export.zero_verts(objs)
        return {'FINISHED'}


class SXTOOLS2_OT_macro(bpy.types.Operator):
    bl_idname = 'sx2.macro'
    bl_label = 'Process Exports'
    bl_description = 'Calculates material channels\naccording to category.\nApplies modifiers in High Detail mode'
    bl_options = {'UNDO'}

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
            modeString += 'Default batch process:\n1) Overlay bake\n2) Occlusion bake\nNOTE: Overwrites existing overlay and occlusion'
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
        elif category == 'TRANSPARENT':
            modeString += 'Transparent batch process:\n1) Overlay bake\n2) Occlusion bake\nNOTE: Overwrites existing overlay and occlusion'
        else:
            modeString += 'Batch process:\nCalculates material channels according to category'
        return modeString


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        scene = context.scene.sx2
        objs = mesh_selection_validator(self, context)

        if bpy.app.background:
            filtered_objs = []
            groups = utils.find_groups(objs)
            for group in groups:
                if group is not None and group.sx2.exportready:
                    all_children = utils.find_children(group, recursive=True)
                    for child in all_children:
                        if child.type == 'MESH':
                            filtered_objs.append(child)

            bpy.ops.object.select_all(action='DESELECT')
            for obj in filtered_objs:
                obj.select_set(True)

            objs = filtered_objs

        if len(objs) > 0:
            bpy.context.view_layer.objects.active = objs[0]
            # check = export.validate_objects(objs)
            if True:  # check:
                sxglobals.magic_in_progress = True
                magic.process_objects(objs)
                if hasattr(bpy.types, bpy.ops.object.vhacd.idname()) and scene.exportcolliders:
                    export.generate_mesh_colliders(objs)

                sxglobals.magic_in_progress = False
                setup.update_sx2material(context)
                scene.shadingmode = 'FULL'
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
            obj['sxToolsVersion'] = 'SX Tools for Blender ' + str(sys.modules['sxtools2'].bl_info.get('version'))

        asset_category = objs[0].sx2.category.lower()
        asset_tags = self.assetTags.split(' ')

        groups = []
        export_groups = utils.find_groups(objs, all_groups=True)
        if len(export_groups) > 0:
            for export_group in export_groups:
                if export_group.sx2.exportready:
                    groups.append(export_group.name)

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
        asset_dict['objects'] = groups
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
        sep_objs = []
        for obj in objs:
            if obj.sx2.smartseparate:
                if obj.sx2.xmirror or obj.sx2.ymirror or obj.sx2.zmirror:
                    sep_objs.append(obj)

        if len(sep_objs) > 0:
            for obj in sep_objs:
                if obj.parent is None:
                    obj.hide_viewport = False
                    if obj.type == 'MESH':
                        export.group_objects([obj, ])
            export.smart_separate(sep_objs)
        else:
            message_box('No objects selected with Smart Separate enabled!')

        return {'FINISHED'}


class SXTOOLS2_OT_generatemasks(bpy.types.Operator):
    bl_idname = 'sx2.generatemasks'
    bl_label = 'Create Palette Masks'
    bl_description = 'Bakes masks of color layers into a UV channel\nfor exporting to game engine\nif dynamic palettes are used'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:
            export.generate_palette_masks(objs)
            # export.flatten_alphas(objs)
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


    def invoke(self, context, event):
        test_objs = mesh_selection_validator(self, context)
        objs = []
        for obj in test_objs:
            if 'sxtools' in obj:
                objs.append(obj)

        if len(objs) > 0:
            if not sxglobals.refresh_in_progress:
                sxglobals.refresh_in_progress = True

                utils.mode_manager(objs, set_mode=True, mode_id='sxtosx2')
                for obj in objs:
                    obj.select_set(False)

                for obj in objs:
                    context.view_layer.objects.active = obj
                    obj.select_set(True)

                    count = len(obj.data.color_attributes)
                    for i in range(count):
                        # omit legacy composite layer
                        if i > 0:
                            layers.add_layer([obj, ], name='Layer '+str(i), layer_type='COLOR', color_attribute='VertexColor'+str(i), clear=False)

                    layers.add_layer([obj, ], name='Gradient 1', layer_type='COLOR')
                    layers.add_layer([obj, ], name='Gradient 2', layer_type='COLOR')
                    layers.add_layer([obj, ], name='Overlay', layer_type='COLOR')

                    layers.add_layer([obj, ], name='Occlusion', layer_type='OCC')
                    layers.add_layer([obj, ], name='Metallic', layer_type='MET')
                    layers.add_layer([obj, ], name='Roughness', layer_type='RGH')
                    layers.add_layer([obj, ], name='Transmission', layer_type='TRN')
                    layers.add_layer([obj, ], name='Emission', layer_type='EMI')

                    for uv in obj.data.uv_layers:
                        print('uv name:', uv.name)
                        if uv.name == 'UVSet0':
                            pass
                        elif uv.name == 'UVSet1':
                            # grab occlusion from V
                            colors = convert.uvs_to_colors(obj, uv.name, 'V')
                            layers.set_layer(obj, colors, obj.sx2layers['Occlusion'])
                        elif uv.name == 'UVSet2':
                            # grab transmission, emission
                            colors = convert.uvs_to_colors(obj, uv.name, 'U')
                            layers.set_layer(obj, colors, obj.sx2layers['Transmission'])
                            colors = convert.uvs_to_colors(obj, uv.name, 'V', with_mask=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Emission'])
                        elif uv.name == 'UVSet3':
                            # grab metallic, smoothness
                            colors = convert.uvs_to_colors(obj, uv.name, 'U')
                            layers.set_layer(obj, colors, obj.sx2layers['Metallic'])
                            colors = convert.uvs_to_colors(obj, uv.name, 'V', invert=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Roughness'])
                        elif uv.name == 'UVSet4':
                            # grab gradient 1, 2
                            colors = convert.uvs_to_colors(obj, uv.name, 'U', as_alpha=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Gradient 1'])
                            colors = convert.uvs_to_colors(obj, uv.name, 'V', as_alpha=True)
                            layers.set_layer(obj, colors, obj.sx2layers['Gradient 2'])
                        elif uv.name == 'UVSet5':
                            # grab and store overlay RG
                            pass
                        elif uv.name == 'UVSet6':
                            # grab overlay BA
                            pass

                    obj.select_set(False)

                for obj in objs:
                    obj.select_set(True)

                modifiers.remove_modifiers(objs)
                if ('sx_tiler' in bpy.data.node_groups):
                    bpy.data.node_groups.remove(bpy.data.node_groups['sx_tiler'], do_unlink=True)

                utils.mode_manager(objs, revert=True, mode_id='sxtosx2')
                sxglobals.refresh_in_progress = False

                refresh_swatches(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_test_button(bpy.types.Operator):
    bl_idname = 'sx2.test_button'
    bl_label = 'Test Button'
    bl_description = 'Run test functions'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = mesh_selection_validator(self, context)
        if len(objs) > 0:

        # ---- LINEAR ALPHA CONVERSION ----
        #     layers.linear_alphas(objs)
        #     refresh_swatches(self, context)


        # ---- COMPOSITE LAYERS ----
            utils.mode_manager(objs, set_mode=True, mode_id='composite_layers')
            layers.add_layer(objs, name='Composite', layer_type='CMP')

            for obj in objs:
                comp_layers = utils.find_color_layers(obj)
                layers.blend_layers([obj, ], comp_layers, comp_layers[0], obj.sx2layers['Composite'])

            palette = utils.find_colors_by_frequency(objs, objs[0].sx2layers['Composite'])
            grid = 1 if len(palette) == 0 else 2**math.ceil(math.log2(math.sqrt(len(palette))))

            for obj in objs:
                if len(obj.data.uv_layers) == 0:
                    obj.data.uv_layers.new(name='UVMap')

                color_uv_coords = {}
                offset = 1.0/grid * 0.5
                j = -1
                for i, color in enumerate(palette):
                    a = i%grid
                    if a == 0:
                        j += 1
                    u = a*offset*2 + offset
                    v = j*offset*2 + offset
                    color_uv_coords[color] = [u, v]

                colors = layers.get_layer(obj, obj.sx2layers['Composite'], as_tuple=True)
                uvs = layers.get_uvs(obj, obj.data.uv_layers.active.name)
                count = len(colors)

                for i in range(count):
                    color = colors[i]
                    for key in color_uv_coords.keys():
                        if utils.color_compare(color, key, 0.01):
                            uvs[i*2:i*2+2] = color_uv_coords[color]

                layers.set_uvs(obj, obj.data.uv_layers.active, uvs)

            files.export_palette_texture(palette)
            utils.mode_manager(objs, revert=True, mode_id='composite_layers')
            refresh_swatches(self, context)

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
    SXTOOLS2_OT_mergeup,
    SXTOOLS2_OT_mergedown,
    SXTOOLS2_OT_copylayer,
    SXTOOLS2_OT_pastelayer,
    SXTOOLS2_OT_clearlayers,
    SXTOOLS2_OT_selmask,
    SXTOOLS2_OT_add_ramp,
    SXTOOLS2_OT_del_ramp,
    SXTOOLS2_OT_addpalette,
    SXTOOLS2_OT_delpalette,
    SXTOOLS2_OT_addpalettecategory,
    SXTOOLS2_OT_delpalettecategory,
    SXTOOLS2_OT_addmaterial,
    SXTOOLS2_OT_delmaterial,
    SXTOOLS2_OT_addmaterialcategory,
    SXTOOLS2_OT_delmaterialcategory)

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
    SXTOOLS2_OT_macro,
    SXTOOLS2_OT_checklist,
    SXTOOLS2_OT_catalogue_add,
    SXTOOLS2_OT_catalogue_remove,
    SXTOOLS2_OT_create_sxcollection,
    SXTOOLS2_OT_smart_separate,
    SXTOOLS2_OT_generatemasks,
    SXTOOLS2_OT_resetscene,
    SXTOOLS2_OT_sxtosx2,
    SXTOOLS2_OT_test_button)

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


if __name__ == '__main__':
    try:
        unregister()
    except:
        pass
    register()

# TODO:
# - reset scene
# - keymonitor does not start without creating an empty layer
# - Atlas export settings:
#   - Emission atlas -> emissive color in composite
#   - Smoothness vs. roughness
#   - Metallic
#   - Write any pbr atlas by iterating over color UVs
# - magic processing
#   - fix gradients for roughness
# - rewrite validation functions according to current data structures, validate based on category
# - import modifier settings
# - fix auto-smooth errors (?!!)
# - generate masks updated to dynamic layer stack
# - save-pre-handler, save-post-handler
# - load category in a non-destructive way
# - vertex bevel modifier?

