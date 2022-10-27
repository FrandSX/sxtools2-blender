bl_info = {
    'name': 'SX Tools 2',
    'author': 'Jani Kahrama / Secret Exit Ltd.',
    'version': (0, 0, 1),
    'blender': (3, 3, 0),
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
        self.refreshInProgress = False
        self.hslUpdate = False
        self.matUpdate = False
        self.curvatureUpdate = False
        self.modalStatus = False
        self.composite = False
        self.copyLayer = None
        self.prevMode = 'OBJECT'
        self.prevShadingMode = 'FULL'
        self.mode = None
        self.modeID = None

        self.randomseed = 42

        # Brush tools may leave low alpha values that break
        # palettemasks, alphaTolerance can be used to fix this.
        # The default setting accepts only fully opaque values.
        self.alphaTolerance = 1.0

        # Use absolute paths
        bpy.context.preferences.filepaths.use_relative_paths = False


    def __del__(self):
        print('SX Tools: Exiting sxglobals')


# ------------------------------------------------------------------------
#    Useful Miscellaneous Functions
# ------------------------------------------------------------------------
class SXTOOLS2_utils(object):
    def __init__(self):
        return None


    def find_layer_from_index(self, obj, index):
        for layer in obj.sx2layers:
            if layer.index == index:
                return layer
        return None


    def __del__(self):
        print('SX Tools: Exiting utils')


# ------------------------------------------------------------------------
#    Settings and properties
# ------------------------------------------------------------------------
class SXTOOLS2_objectprops(bpy.types.PropertyGroup):

    selectedlayer: bpy.props.IntProperty(
        name='Selected Layer',
        min=0,
        default=0)
        # update=refresh_actives)

    # Running index for unique layer names
    layercount: bpy.props.IntProperty(
        name='Layer Count',
        min=0,
        default=0)
        # update=refresh_actives)

    activeLayerAlpha: bpy.props.FloatProperty(
        name='Opacity',
        min=0.0,
        max=1.0,
        default=1.0)
        # update=update_layers)

    activeLayerBlendMode: bpy.props.EnumProperty(
        name='Blend Mode',
        items=[
            ('ALPHA', 'Alpha', ''),
            ('ADD', 'Additive', ''),
            ('MUL', 'Multiply', ''),
            ('OVR', 'Overlay', '')],
        default='ALPHA')
        # update=update_layers))


class SXTOOLS2_sceneprops(bpy.types.PropertyGroup):

    shadingmode: bpy.props.EnumProperty(
        name='Shading Mode',
        description='Full: Composited shading with all channels enabled\nDebug: Display selected layer only, with alpha as black\nAlpha: Display layer alpha or UV values in grayscale',
        items=[
            ('FULL', 'Full', ''),
            ('DEBUG', 'Debug', ''),
            ('ALPHA', 'Alpha', '')],
        default='FULL')
        # update=update_layers)

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

    huevalue: bpy.props.FloatProperty(
        name='Hue',
        description='The mode hue in the selection',
        min=0.0,
        max=1.0,
        default=0.0)
        # update=lambda self, context: adjust_hsl(self, context, 0))

    saturationvalue: bpy.props.FloatProperty(
        name='Saturation',
        description='The max saturation in the selection',
        min=0.0,
        max=1.0,
        default=0.0)
        # update=lambda self, context: adjust_hsl(self, context, 1))

    lightnessvalue: bpy.props.FloatProperty(
        name='Lightness',
        description='The max lightness in the selection',
        min=0.0,
        max=1.0,
        default=0.0)
        # update=lambda self, context: adjust_hsl(self, context, 2))

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
        default=(1.0, 1.0, 1.0, 1.0))
        # update=update_fillcolor)

    mergemode: bpy.props.EnumProperty(
        name='Merge Mode',
        items=[
            ('UP', 'Up', ''),
            ('DOWN', 'Down', '')],
        default='UP')

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


class SXTOOLS2_layer(bpy.types.PropertyGroup):
    # name: from PropertyGroup

    enabled: bpy.props.BoolProperty(
        name='Layer Enabled',
        default=True)

    index: bpy.props.IntProperty(
        name='Layer Index',
        min=0,
        max=100,
        default=0)

    layer_type: bpy.props.EnumProperty(
        name='Layer Type',
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
        default=True)

    # Int-type duplicate of the above for shader network
    int_visibility: bpy.props.IntProperty(
        name='Layer Visibility',
        min=0,
        max=1,
        default=1)

    opacity: bpy.props.FloatProperty(
        name='Layer Opacity',
        min=0.0,
        max=1.0,
        default=1.0)

    blend_mode: bpy.props.EnumProperty(
        name='Layer Blend Mode',
        items=[
            ('ALPHA', 'Alpha', ''),
            ('ADD', 'Additive', ''),
            ('MUL', 'Multiply', ''),
            ('OVR', 'Overlay', '')],
        default='ALPHA')

    color_attribute: bpy.props.StringProperty(
        name='Vertex Color Layer',
        description='Maps a list item to a vertex color layer',
        default='')

    locked: bpy.props.BoolProperty(
        name='Lock Layer Opacity',
        default=False)


# Return value eliminates duplicates
def selection_validator(self, context):
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


def message_box(message='', title='SX Tools', icon='INFO'):
    messageLines = message.splitlines()


    def draw(self, context):
        for line in messageLines:
            self.layout.label(text=line)

    if not bpy.app.background:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


# ------------------------------------------------------------------------
#    UI Panel
# ------------------------------------------------------------------------
class SXTOOLS2_PT_panel(bpy.types.Panel):

    bl_idname = 'SXTOOLS2_PT_panel'
    bl_label = 'SX Tools 2'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SX Tools 2'


    def draw(self, context):
        objs = selection_validator(self, context)
        layout = self.layout

        if len(objs) > 0:
            obj = objs[0]

            mode = obj.mode
            sx2 = obj.sx2
            scene = context.scene.sx2
            
            sel_idx = objs[0].sx2.selectedlayer
            layer = utils.find_layer_from_index(obj, sel_idx)
            # if layer is None:
            #     message_box('Invalid layer selected!', 'SX Tools Error', 'ERROR')

            row_shading = layout.row()
            row_shading.prop(scene, 'shadingmode', expand=True)
            row_shading.operator("wm.url_open", text='', icon='URL').url = 'https://www.notion.so/SX-Tools-for-Blender-Documentation-9ad98e239f224624bf98246822a671a6'

            # Layer Controls -----------------------------------------------
            box_layer = layout.box()
            row_layer = box_layer.row()

            if layer is None:
                box_layer.enabled = False
                row_layer.label(text='No layers')
            else:
                row_layer.prop(
                    scene, 'expandlayer',
                    icon='TRIA_DOWN' if scene.expandlayer else 'TRIA_RIGHT',
                    icon_only=True, emboss=False)

                split_basics = row_layer.split(factor=0.3)
                split_basics.prop(layer, 'blend_mode', text='')
                split_basics.prop(layer, 'opacity', slider=True, text='Layer Opacity')

                if ((layer.name == 'occlusion') or
                    (layer.name == 'smoothness') or
                    (layer.name == 'metallic') or
                    (layer.name == 'transmission') or
                    (layer.name == 'emission')):
                    split_basics.enabled = False

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
                    row_hue.prop(scene, 'huevalue', slider=True, text=hue_text)
                    row_sat = col_hsl.row(align=True)
                    row_sat.prop(scene, 'saturationvalue', slider=True, text=saturation_text)
                    row_lightness = col_hsl.row(align=True)
                    row_lightness.prop(scene, 'lightnessvalue', slider=True, text=lightness_text)

                    if ((layer.name == 'occlusion') or
                        (layer.name == 'smoothness') or
                        (layer.name == 'metallic') or
                        (layer.name == 'transmission') or
                        (layer.name == 'emission') or
                        (layer.index == 8) or
                        (layer.index == 9)):
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
            col_list.operator('sx2.applytool', text='', icon='TRIA_UP')
            col_list.operator('sx2.applytool', text='', icon='TRIA_DOWN')

            # Fill Tools --------------------------------------------------------
            box_fill = layout.box()
            row_fill = box_fill.row()
            if scene.toolmode != 'PAL' and scene.toolmode != 'MAT':
                row_fill.operator('sx2.applytool', text='Apply')

            # Color Tool --------------------------------------------------------
            if scene.toolmode == 'COL':
                split1_fill = box_fill.split(factor=0.33)
                fill_text = 'Fill Color'
                split1_fill.label(text=fill_text)
                split2_fill = split1_fill.split(factor=0.8)
                split2_fill.prop(scene, 'fillcolor', text='')

                row_fpalette = box_fill.row(align=True)
                for i in range(8):
                    row_fpalette.prop(scene, 'fillpalette' + str(i+1), text='')

        else:
            col = layout.column()
            col.label(text='Select a mesh to continue')


class SXTOOLS2_UL_layerlist(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index, flt_flag):
        scene = context.scene.sx2
        objs = selection_validator(self, context)
        hide_icon = {False: 'HIDE_ON', True: 'HIDE_OFF'}
        lock_icon = {False: 'UNLOCKED', True: 'LOCKED'}

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row_item = layout.row(align=True)

            if scene.shadingmode == 'FULL':
                row_item.prop(item, 'visibility', text='', icon=hide_icon[item.visibility])
            else:
                row_item.label(icon=hide_icon[(item.index == objs[0].sx2.selectedlayer)])

            if sxglobals.mode == 'OBJECT':
                if scene.toolmode == 'PAL':
                    row_item.label(text='', icon='LOCKED')
                else:
                    row_item.prop(item, 'locked', text='', icon=lock_icon[item.locked])
            else:
                row_item.label(text='', icon='UNLOCKED')

            row_item.label(text='  ' + item.name)

        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text='', icon=hide_icon[item.visibility])


    # Called once to draw filtering/reordering options.
    def draw_filter(self, context, layout):
        pass


    def filter_items(self, context, data, propname):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            flt_flags = []
            flt_neworder = []
            flt_flags = [self.bitflag_filter_item] * len(objs[0].sx2layers)
            for i in range(len(objs[0].sx2layers)):
                flt_neworder.append(i)

                # if not layer.enabled:
                #     flt_flags[idx] |= ~self.bitflag_filter_item

            return flt_flags, flt_neworder


# ------------------------------------------------------------------------
#   Operators
# ------------------------------------------------------------------------
class SXTOOLS2_OT_applytool(bpy.types.Operator):
    bl_idname = 'sx2.applytool'
    bl_label = 'Apply Tool'
    bl_description = 'Applies the selected mode fill\nto the selected components or objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        # if len(objs) > 0:
        #     idx = objs[0].sx2.selectedlayer
        #     layer = utils.find_layer_from_index(objs[0], idx)
        #     color = context.scene.sx2.fillcolor

        #     tools.apply_tool(objs, layer, color=color)
        #     if context.scene.sx2.toolmode == 'COL':
        #         tools.update_recent_colors(color)

        #     sxglobals.composite = True
        #     refresh_actives(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_add_layer(bpy.types.Operator):
    bl_idname = 'sx2.add_layer'
    bl_label = 'Add Layer'
    bl_description = 'Add a new layer'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            for obj in objs:
                item = obj.sx2layers.add()
                item.name = 'Layer' + str(obj.sx2.layercount)
                item.layer_type = 'COLOR'
                item.default_color = (0.0, 0.0, 0.0, 0.0)
                item.visibility = 1
                item.opacity = 1.0
                item.blend_mode = 'ALPHA'
                item.color_attribute = item.name
                item.locked = False

                obj.data.attributes.new(name=item.name, type='FLOAT_COLOR', domain='CORNER')

                item.index = len(obj.sx2layers) - 1
                obj.sx2.selectedlayer = item.index
                obj.sx2.layercount += 1

            # tools.apply_tool(objs, layer, color=color)
            # if context.scene.sx2.toolmode == 'COL':
            #     tools.update_recent_colors(color)
            # update_sx2material()

            # sxglobals.composite = True
            # refresh_actives(self, context)

        return {'FINISHED'}


class SXTOOLS2_OT_del_layer(bpy.types.Operator):
    bl_idname = 'sx2.del_layer'
    bl_label = 'Remove Layer'
    bl_description = 'Remove layer'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            idx = objs[0].sx2.selectedlayer
            for obj in objs:
                if len(obj.sx2layers) > 0:
                    obj.sx2.selectedlayer = 0 if (idx - 1 < 0) else idx - 1
                    obj.data.attributes.remove(obj.data.attributes[idx])
                    obj.sx2layers.remove(idx)

        return {'FINISHED'}



# ------------------------------------------------------------------------
#    Registration and initialization
# ------------------------------------------------------------------------
sxglobals = SXTOOLS2_sxglobals()
utils = SXTOOLS2_utils()

classes = (
    SXTOOLS2_objectprops,
    SXTOOLS2_sceneprops,
    SXTOOLS2_layer,
    SXTOOLS2_PT_panel,
    SXTOOLS2_UL_layerlist,
    SXTOOLS2_OT_applytool,
    SXTOOLS2_OT_add_layer,
    SXTOOLS2_OT_del_layer)

addon_keymaps = []


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Object.sx2 = bpy.props.PointerProperty(type=SXTOOLS2_objectprops)
    bpy.types.Object.sx2layers = bpy.props.CollectionProperty(type=SXTOOLS2_layer)
    bpy.types.Scene.sx2 = bpy.props.PointerProperty(type=SXTOOLS2_sceneprops)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Object.sx2
    del bpy.types.Object.sx2layers
    del bpy.types.Scene.sx2


if __name__ == '__main__':
    try:
        unregister()
    except:
        pass
    register()
