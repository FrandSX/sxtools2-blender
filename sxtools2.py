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

        # {stack_layer_index: (layer.name, layer.color_attribute, layer.layer_type)}
        self.layer_stack_dict = {}


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


    def mode_manager(self, objs, set_mode=False, revert=False, mode_id=None):
        if set_mode:
            if sxglobals.modeID is None:
                sxglobals.modeID = mode_id
                sxglobals.mode = objs[0].mode
                if objs[0].mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
            else:
                if objs[0].mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        elif revert:
            if sxglobals.modeID == mode_id:
                bpy.ops.object.mode_set(mode=sxglobals.mode)
                sxglobals.modeID = None


    def find_layer_by_stack_index(self, obj, index):
        if len(obj.sx2layers) > 0:
            if len(sxglobals.layer_stack_dict) == 0:
                sxglobals.layer_stack_dict.clear()

                layers = []
                for layer in obj.sx2layers:
                    layers.append((layer.index, layer.color_attribute))

                layers.sort(key=lambda y: y[0])
                for i, layer_tuple in enumerate(layers):
                    for layer in obj.sx2layers:
                        if layer.color_attribute == layer_tuple[1]:
                            sxglobals.layer_stack_dict[i] = (layer.name, layer.color_attribute, layer.layer_type)

            return obj.sx2layers[sxglobals.layer_stack_dict[index][0]]
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


    def find_root_pivot(self, objs):
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_object_bounding_box(objs)
        pivot = ((xmax + xmin)*0.5, (ymax + ymin)*0.5, zmin)

        return pivot


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
        sxglobals.layer_stack_dict.clear()
        inserted_layer.index = 100

        if len(obj.sx2layers) == 1:
            index = 0

        layers = []
        for layer in obj.sx2layers:
            layers.append((layer.index, layer.color_attribute))

        layers.sort(key=lambda y: y[0])
        layers.remove((100, inserted_layer.color_attribute))
        layers.insert(index, (index, inserted_layer.color_attribute))

        for i, layer_tuple in enumerate(layers):
            for layer in obj.sx2layers:
                if layer.color_attribute == layer_tuple[1]:
                    layer.index = i
                    sxglobals.layer_stack_dict[i] = (layer.name, layer.color_attribute, layer.layer_type)

        return index


    def sort_stack_indices(self, obj):
        sxglobals.layer_stack_dict.clear()

        if len(obj.sx2layers) > 0:
            layers = []
            for layer in obj.sx2layers:
                layers.append((layer.index, layer.color_attribute))

            layers.sort(key=lambda y: y[0])
            print('sorted:', layers)
            for i, layer_tuple in enumerate(layers):
                for layer in obj.sx2layers:
                    if layer.color_attribute == layer_tuple[1]:
                        layer.index = i
                        sxglobals.layer_stack_dict[i] = (layer.name, layer.color_attribute, layer.layer_type)


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
                    modifier.show_viewport = obj.sx2.modifiervisibility

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
        count = len(obj.data.attributes[0].data)
        colors = [color[0], color[1], color[2], color[3]] * count

        return self.mask_list(obj, colors, masklayer, as_tuple)


    def ramp_list(self, obj, objs, rampmode, masklayer=None, mergebbx=True):
        ramp = bpy.data.materials['SXMaterial'].node_tree.nodes['ColorRamp']

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
        ramp = bpy.data.materials['SXMaterial'].node_tree.nodes['ColorRamp']
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
        count = len(mesh.uv_layers[0].data)
        mask = [0.0] * count

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


    def add_layer(self, objs):
        if len(objs) > 0:
            for obj in objs:
                item = obj.sx2layers.add()
                item.name = 'Layer ' + str(obj.sx2.layercount)
                item.color_attribute = item.name

                obj.data.attributes.new(name=item.name, type='FLOAT_COLOR', domain='CORNER')
                item.index = utils.insert_layer_at_index(obj, item, obj.sx2layers[obj.sx2.selectedlayer].index + 1)
                print('added at:', item.index)

                colors = generate.color_list(obj, item.default_color)
                layers.set_layer(obj, colors, obj.sx2layers[len(obj.sx2layers) - 1])

                obj.sx2.selectedlayer = len(obj.sx2layers) - 1
                obj.sx2.layercount += 1

            # refresh_actives(self, context)


    # wrapper for low-level functions, always returns layerdata in RGBA
    def get_layer(self, obj, sourcelayer, as_tuple=False, uv_as_alpha=False, apply_layer_alpha=False, gradient_with_palette=False):
        valid_sources = ['COLOR', 'OCC', 'MET', 'RGH', 'TRN', 'EMI']
        sourceType = sourcelayer.layer_type
        # sxmaterial = bpy.data.materials['SXMaterial'].node_tree
        alpha = sourcelayer.opacity

        if sourceType in valid_sources:
            values = self.get_colors(obj, sourcelayer.color_attribute)

        elif sourceType == 'UV':
            uvs = self.get_uvs(obj, sourcelayer.uvLayer0, channel=sourcelayer.uvChannel0)
            count = len(uvs)  # len(obj.data.uv_layers[0].data)
            values = [None] * count * 4
            dv = [1.0, 1.0, 1.0, 1.0]

            # if gradient_with_palette:
            #     if sourcelayer.name == 'gradient1':
            #         dv = sxmaterial.nodes['PaletteColor3'].outputs[0].default_value
            #     elif sourcelayer.name == 'gradient2':
            #         dv = sxmaterial.nodes['PaletteColor4'].outputs[0].default_value

            if uv_as_alpha:
                for i in range(count):
                    if uvs[i] > 0.0:
                        values[(0+i*4):(4+i*4)] = [dv[0], dv[1], dv[2], uvs[i]]
                    else:
                        values[(0+i*4):(4+i*4)] = [0.0, 0.0, 0.0, uvs[i]]
            else:
                for i in range(count):
                    values[(0+i*4):(4+i*4)] = [uvs[i], uvs[i], uvs[i], 1.0]

        elif sourceType == 'UV4':
            values = layers.get_uv4(obj, sourcelayer)

        if apply_layer_alpha and alpha != 1.0:
            count = len(values)//4
            for i in range(count):
                values[3+i*4] *= alpha

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

        valid_sources = ['COLOR', 'OCC', 'MET', 'RGH', 'TRN', 'EMI']
        targetType = targetlayer.layer_type

        if targetType in valid_sources:
            layers.set_colors(obj, targetlayer.color_attribute, colors)

        elif targetType == 'UV':
            if (targetlayer.name == 'gradient1') or (targetlayer.name == 'gradient2'):
                target_uvs = colors[3::4]
                if constant_alpha_test(target_uvs):
                    target_uvs = layers.get_luminances(obj, sourcelayer=None, colors=colors, as_rgba=False)
            else:
                target_uvs = layers.get_luminances(obj, sourcelayer=None, colors=colors, as_rgba=False)

            layers.set_uvs(obj, targetlayer.uvLayer0, target_uvs, targetlayer.uvChannel0)

        elif targetType == 'UV4':
            layers.set_uv4(obj, targetlayer, colors)


    def get_layer_mask(self, obj, sourcelayer):
        layer_type = sourcelayer.layer_type

        if layer_type == 'COLOR':
            colors = self.get_colors(obj, sourcelayer.color_attribute)
            values = colors[3::4]
        elif layer_type == 'UV':
            values = self.get_uvs(obj, sourcelayer.uvLayer0, sourcelayer.uvChannel0)
        elif layer_type == 'UV4':
            values = self.get_uvs(obj, sourcelayer.uvLayer3, sourcelayer.uvChannel3)

        if any(v != 0.0 for v in values):
            return values, False
        else:
            return values, True


    def get_colors(self, obj, source):
        sourceColors = obj.data.attributes[source].data
        colors = [None] * len(sourceColors) * 4
        sourceColors.foreach_get('color', colors)
        return colors


    def set_colors(self, obj, target, colors):
        targetColors = obj.data.attributes[target].data
        targetColors.foreach_set('color', colors)


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


    def get_uvs(self, obj, sourcelayer, channel=None):
        channels = {'U': 0, 'V': 1}
        sourceUVs = obj.data.uv_layers[sourcelayer].data
        count = len(sourceUVs)
        source_uvs = [None] * count * 2
        sourceUVs.foreach_get('uv', source_uvs)

        if channel is None:
            uvs = source_uvs
        else:
            uvs = [None] * count
            sc = channels[channel]
            for i in range(count):
                uvs[i] = source_uvs[sc+i*2]

        return uvs


    # when targetchannel is None, sourceuvs is expected to contain data for both U and V
    def set_uvs(self, obj, targetlayer, sourceuvs, targetchannel=None):
        channels = {'U': 0, 'V': 1}
        targetUVs = obj.data.uv_layers[targetlayer].data

        if targetchannel is None:
            targetUVs.foreach_set('uv', sourceuvs)
        else:
            target_uvs = self.get_uvs(obj, targetlayer)
            tc = channels[targetchannel]
            count = len(sourceuvs)
            for i in range(count):
                target_uvs[tc+i*2] = sourceuvs[i]
            targetUVs.foreach_set('uv', target_uvs)


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
                setattr(obj.sx2layers[layer.name], 'alpha', 1.0)
                setattr(obj.sx2layers[layer.name], 'visibility', True)
                setattr(obj.sx2layers[layer.name], 'locked', False)
                if reset:
                    if layer == obj.sx2layers['overlay']:
                        setattr(obj.sx2layers[layer.name], 'blend_mode', 'OVR')
                        default_color = (0.5, 0.5, 0.5, 1.0)
                    else:
                        setattr(obj.sx2layers[layer.name], 'blend_mode', 'ALPHA')
                        if layer.index == 1 and obj.sx2.category != 'TRANSPARENT':
                            default_color = (0.5, 0.5, 0.5, 1.0)
                        elif layer.layer_type == 'COLOR':
                            default_color = (0.0, 0.0, 0.0, 0.0)
                colors = generate.color_list(obj, color=default_color)
                layers.set_layer(obj, colors, layer)
            else:
                if (targetlayer.name == 'gradient1') or (targetlayer.name == 'gradient2'):
                    colors = layers.get_layer(obj, layer, uv_as_alpha=True)
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
                    if sxlayer.enabled:
                        sxlayer.locked = False
                        clear_layer(obj, sxlayer, reset=True)
                obj.data.update()
        else:
            for obj in objs:
                clear_layer(obj, obj.sx2layers[targetlayer.name])
                obj.data.update()


    def composite_layers(self, objs):
        if sxglobals.composite:
            # then = time.perf_counter()
            compLayers = utils.find_comp_layers(objs[0])
            shadingmode = bpy.context.scene.sx2.shadingmode
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

            if shadingmode == 'FULL':
                layer0 = utils.find_layer_by_stack_index(objs[0], 0)
                layer1 = utils.find_layer_by_stack_index(objs[0], 1)
                self.blend_layers(objs, compLayers, layer1, layer0, uv_as_alpha=True)
            else:
                self.blend_debug(objs, layer, shadingmode)

            sxglobals.composite = False
            # now = time.perf_counter()
            # print('Compositing duration: ', now-then, ' seconds')


    def blend_debug(self, objs, layer, shadingmode):
        active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = objs[0]

        for obj in objs:
            colors = self.get_layer(obj, layer, uv_as_alpha=True)
            count = len(colors)//4

            if shadingmode == 'DEBUG':
                for i in range(count):
                    color = colors[(0+i*4):(4+i*4)]
                    a = color[3] * layer.opacity
                    colors[(0+i*4):(4+i*4)] = [color[0]*a, color[1]*a, color[2]*a, 1.0]
            elif shadingmode == 'ALPHA':
                for i in range(count):
                    a = colors[3+i*4] * layer.opacity
                    colors[(0+i*4):(4+i*4)] = [a, a, a, 1.0]

            self.set_layer(obj, colors, obj.sx2layers['composite'])
            # obj.data.update()
        bpy.context.view_layer.objects.active = active


    def blend_layers(self, objs, topLayerArray, baseLayer, resultLayer, uv_as_alpha=False):
        active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = objs[0]

        for obj in objs:
            basecolors = self.get_layer(obj, baseLayer, uv_as_alpha=uv_as_alpha)

            for layer in topLayerArray:
                if getattr(obj.sx2layers[layer.name], 'visibility'):
                    blendmode = getattr(obj.sx2layers[layer.name], 'blend_mode')
                    layeralpha = getattr(obj.sx2layers[layer.name], 'alpha')
                    topcolors = self.get_layer(obj, obj.sx2layers[layer.name], uv_as_alpha=uv_as_alpha, gradient_with_palette=True)
                    basecolors = tools.blend_values(topcolors, basecolors, blendmode, layeralpha)

            self.set_layer(obj, basecolors, resultLayer)
        bpy.context.view_layer.objects.active = active


    # Generate 1-bit layer masks for color layers
    # so the faces can be re-colored in a game engine
    def generate_masks(self, objs):
        channels = {'R': 0, 'G': 1, 'B': 2, 'A': 3, 'U': 0, 'V': 1}

        for obj in objs:
            vertexColors = obj.data.attributes
            uvValues = obj.data.uv_layers
            layers = utils.find_color_layers(obj)
            del layers[0]
            for layer in layers:
                uvmap = obj.sx2layers['masks'].uvLayer0
                targetChannel = obj.sx2layers['masks'].uvChannel0
                for poly in obj.data.polygons:
                    for idx in poly.loop_indices:
                        for i, layer in enumerate(layers):
                            i += 1
                            if i == 1:
                                uvValues[uvmap].data[idx].uv[channels[targetChannel]] = i
                            else:
                                vertexAlpha = vertexColors[layer.color_attribute].data[idx].color[3]
                                if vertexAlpha >= sxglobals.alphaTolerance:
                                    uvValues[uvmap].data[idx].uv[channels[targetChannel]] = i


    def flatten_alphas(self, objs):
        for obj in objs:
            vertexUVs = obj.data.uv_layers
            channels = {'U': 0, 'V': 1}
            for layer in obj.sx2layers:
                alpha = layer.opacity
                if (layer.name == 'gradient1') or (layer.name == 'gradient2'):
                    for poly in obj.data.polygons:
                        for idx in poly.loop_indices:
                            vertexUVs[layer.uvLayer0].data[idx].uv[channels[layer.uvChannel0]] *= alpha
                    layer.opacity = 1.0
                elif layer.name == 'overlay':
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


    def merge_layers(self, objs, toplayer, baselayer, targetlayer):
        for obj in objs:
            basecolors = self.get_layer(obj, baselayer, apply_layer_alpha=True, uv_as_alpha=True)
            topcolors = self.get_layer(obj, toplayer, apply_layer_alpha=True, uv_as_alpha=True)

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


    def paste_layer(self, objs, sourceLayer, targetLayer, fillMode):
        utils.mode_manager(objs, set_mode=True, mode_id='paste_layer')

        sourceMode = sourceLayer.layer_type
        targetMode = targetLayer.layer_type

        if (sourceMode == 'COLOR' or sourceMode == 'UV4') and (targetMode == 'COLOR' or targetMode == 'UV4'):
            for obj in objs:
                sourceBlend = getattr(obj.sx2layers[sourceLayer.name], 'blend_mode')[:]
                targetBlend = getattr(obj.sx2layers[targetLayer.name], 'blend_mode')[:]
                sourceAlpha = getattr(obj.sx2layers[sourceLayer.name], 'opacity')
                targetAlpha = getattr(obj.sx2layers[targetLayer.name], 'opacity')

                if fillMode == 'swap':
                    setattr(obj.sx2layers[sourceLayer.name], 'blend_mode', targetBlend)
                    setattr(obj.sx2layers[targetLayer.name], 'blend_mode', sourceBlend)
                    setattr(obj.sx2layers[sourceLayer.name], 'opacity', targetAlpha)
                    setattr(obj.sx2layers[targetLayer.name], 'opacity', sourceAlpha)
                else:
                    setattr(obj.sx2layers[targetLayer.name], 'blend_mode', sourceBlend)
                    setattr(obj.sx2layers[targetLayer.name], 'opacity', sourceAlpha)

        if fillMode == 'swap':
            for obj in objs:
                layer1_colors = layers.get_layer(obj, sourceLayer)
                layer2_colors = layers.get_layer(obj, targetLayer)
                layers.set_layer(obj, layer1_colors, targetLayer)
                layers.set_layer(obj, layer2_colors, sourceLayer)
        elif fillMode == 'mask':
            for obj in objs:
                colors = layers.get_layer(obj, targetLayer)
                sourcevalues = generate.mask_list(obj, colors, sourceLayer, override_mask=True)
                layers.set_layer(obj, sourcevalues, targetLayer)
        else:
            for obj in objs:
                colors = self.get_layer(obj, sourceLayer)
                if sxglobals.mode == 'EDIT':
                    targetvalues = self.get_layer(obj, targetLayer)
                    colors = generate.mask_list(obj, colors)
                    colors = tools.blend_values(colors, targetvalues, 'ALPHA', 1.0)
                layers.set_layer(obj, colors, targetLayer)

        utils.mode_manager(objs, revert=True, mode_id='paste_layer')


    def update_layer_panel(self, objs, layer):
        scene = bpy.context.scene.sx2
        colors = utils.find_colors_by_frequency(objs, layer, 8)
        # Update layer HSL elements
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

        sxglobals.hslUpdate = True
        bpy.context.scene.sx2.huevalue = hue
        bpy.context.scene.sx2.saturationvalue = sat
        bpy.context.scene.sx2.lightnessvalue = lightness
        sxglobals.hslUpdate = False

        # Update layer palette elements
        for i, pcol in enumerate(colors):
            palettecolor = (pcol[0], pcol[1], pcol[2], 1.0)
            setattr(scene, 'layerpalette' + str(i + 1), palettecolor)

        # Update SXMaterial color
        # if layer.index <= 5:
        #     bpy.data.materials['SXMaterial'].node_tree.nodes['PaletteColor'+str(layer.index - 1)].outputs[0].default_value = colors[0]


    def color_layers_to_values(self, objs):
        scene = bpy.context.scene.sx2

        for i in range(5):
            layer = objs[0].sx2layers[i+1]
            palettecolor = utils.find_colors_by_frequency(objs, layer, 1, obj_sel_override=True)[0]
            tabcolor = getattr(scene, 'newpalette' + str(i))

            if not utils.color_compare(palettecolor, tabcolor) and not utils.color_compare(palettecolor, (0.0, 0.0, 0.0, 1.0)):
                setattr(scene, 'newpalette' + str(i), palettecolor)
                bpy.data.materials['SXMaterial'].node_tree.nodes['PaletteColor'+str(i)].outputs[0].default_value = palettecolor


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

            obj.data.update()
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
#    Scene Setup
# ------------------------------------------------------------------------
class SXTOOLS2_setup(object):
    def __init__(self):
        return None


    def create_sx2material(self, objs):
        blend_mode_dict = {'ALPHA': 'MIX', 'OVR': 'OVERLAY', 'MUL': 'MULTIPLY', 'ADD': 'ADD'}

        layercount = 0
        for obj in objs:
            if len(obj.sx2layers) > layercount:
                layercount = len(obj.sx2layers)

            sxmaterial = bpy.data.materials.new(name='SX2Material_'+obj.name)
            sxmaterial.use_nodes = True
            sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission'].default_value = [0.0, 0.0, 0.0, 1.0]
            sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = [0.0, 0.0, 0.0, 1.0]
            sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
            sxmaterial.node_tree.nodes['Principled BSDF'].location = (1000, 0)

            sxmaterial.node_tree.nodes['Material Output'].location = (1300, 0)

            prev_color = None

            color_layers = []
            material_layers = []
            if len(sxglobals.layer_stack_dict) > 0:
                color_stack = [sxglobals.layer_stack_dict[key] for key in sorted(sxglobals.layer_stack_dict.keys())]
                for layer in color_stack:
                    if layer[2] == 'COLOR':
                        color_layers.append(layer)
                    else:
                        material_layers.append(layer)

            # Create color layers
            if (len(color_layers) > 0) and objs[0].sx2layers[color_layers[0][0]].visibility:
                base_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                base_color.name = 'BaseColor'
                base_color.label = 'BaseColor'
                base_color.layer_name = color_layers[0][1]
                base_color.location = (-1000, -600)

                prev_color = base_color.outputs['Color']

            # Layer groups
            for i in range(len(color_layers) - 1):
                if obj.sx2layers[color_layers[i+1][0]].visibility:
                    source_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                    source_color.name = 'LayerColor' + str(i + 1)
                    source_color.label = 'LayerColor' + str(i + 1)
                    source_color.layer_name = color_layers[i+1][1]
                    source_color.location = (-1000, i*500)

                    layer_opacity = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                    layer_opacity.name = 'Layer Opacity' + str(i + 1)
                    layer_opacity.label = 'Layer Opacity' + str(i + 1)
                    layer_opacity.attribute_name = 'sx2layers["' + color_layers[i+1][0] + '"].opacity'
                    layer_opacity.attribute_type = 'OBJECT'
                    layer_opacity.location = (-1000, i*500+200)

                    opacity_and_alpha = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                    opacity_and_alpha.name = 'Opacity ' + str(i + 1)
                    opacity_and_alpha.label = 'Opacity ' + str(i + 1)
                    opacity_and_alpha.operation = 'MULTIPLY'
                    opacity_and_alpha.use_clamp = True
                    opacity_and_alpha.inputs[0].default_value = 1
                    opacity_and_alpha.location = (-800, i*500+200)

                    layer_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                    layer_blend.name = 'Mix ' + str(i + 1)
                    layer_blend.label = 'Mix ' + str(i + 1)
                    layer_blend.inputs[0].default_value = 1
                    layer_blend.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                    layer_blend.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                    layer_blend.blend_type = blend_mode_dict[objs[0].sx2layers[color_layers[i+1][0]].blend_mode]
                    layer_blend.use_clamp = True
                    layer_blend.location = (-600, i*500+200)

                    # Group connections
                    output = source_color.outputs['Color']
                    input = layer_blend.inputs['Color2']
                    sxmaterial.node_tree.links.new(input, output)   

                    output = layer_opacity.outputs[2]
                    input = opacity_and_alpha.inputs[0]
                    sxmaterial.node_tree.links.new(input, output)

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

            # Create material layers
            for i in range(len(material_layers)):
                if obj.sx2layers[material_layers[i][0]].visibility:
                    source_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                    source_color.name = material_layers[i][0]
                    source_color.label = material_layers[i][0]
                    source_color.layer_name = material_layers[i][1]
                    source_color.location = (0, i*500)

                    layer_opacity = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                    layer_opacity.name = material_layers[i][0] + ' Opacity'
                    layer_opacity.label = material_layers[i][0] + ' Opacity'
                    layer_opacity.attribute_name = 'sx2layers["' + material_layers[i][0] + '"].opacity'
                    layer_opacity.attribute_type = 'OBJECT'
                    layer_opacity.location = (0, i*500+200)

                    opacity_and_alpha = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                    opacity_and_alpha.name = material_layers[i][0] + ' Opacity and Alpha'
                    opacity_and_alpha.label = material_layers[i][0] + ' Opacity and Alpha'
                    opacity_and_alpha.operation = 'MULTIPLY'
                    opacity_and_alpha.use_clamp = True
                    opacity_and_alpha.inputs[0].default_value = 1
                    opacity_and_alpha.location = (200, i*500+200)

                    if (material_layers[i][2] == 'OCC') or (material_layers[i][2] == 'EMI'):
                        layer_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                        layer_blend.name = material_layers[i][0] + ' Mix'
                        layer_blend.label = material_layers[i][0] + ' Mix'
                        layer_blend.inputs[0].default_value = 1
                        layer_blend.use_clamp = True
                        layer_blend.location = (400, i*500+200)

                    # Group connections
                    output = layer_opacity.outputs[2]
                    input = opacity_and_alpha.inputs[0]
                    sxmaterial.node_tree.links.new(input, output)

                    if (material_layers[i][2] == 'OCC'):
                        output = source_color.outputs['Alpha']
                    else:
                        output = source_color.outputs['Color']
                    input = opacity_and_alpha.inputs[1]
                    sxmaterial.node_tree.links.new(input, output)

                    if material_layers[i][2] == 'OCC':
                        layer_blend.blend_type = 'MULTIPLY'
                        layer_blend.inputs[1].default_value = [1.0, 1.0, 1.0, 1.0]
                        layer_blend.inputs[2].default_value = [1.0, 1.0, 1.0, 1.0]

                        output = source_color.outputs['Color']
                        input = layer_blend.inputs['Color2']
                        sxmaterial.node_tree.links.new(input, output)

                        output = opacity_and_alpha.outputs[0]
                        input = layer_blend.inputs[0]
                        sxmaterial.node_tree.links.new(input, output)

                        if prev_color is not None:
                            output = prev_color
                            input = layer_blend.inputs['Color1']
                            sxmaterial.node_tree.links.new(input, output)

                        prev_color = layer_blend.outputs['Color']

                    if material_layers[i][2] == 'MET':
                        output = opacity_and_alpha.outputs[0]
                        input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Metallic']
                        sxmaterial.node_tree.links.new(input, output)

                    if material_layers[i][2] == 'RGH':
                        output = opacity_and_alpha.outputs[0]
                        input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Roughness']
                        sxmaterial.node_tree.links.new(input, output)

                    if material_layers[i][2] == 'TRN':
                        output = opacity_and_alpha.outputs[0]
                        input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Transmission']
                        sxmaterial.node_tree.links.new(input, output)

                    if material_layers[i][2] == 'EMI':
                        layer_blend.blend_type = 'MIX'
                        layer_blend.inputs[1].default_value = [0.0, 0.0, 0.0, 0.0]
                        layer_blend.inputs[2].default_value = [0.0, 0.0, 0.0, 0.0]
                        sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 100

                        output = source_color.outputs['Color']
                        input = layer_blend.inputs['Color2']
                        sxmaterial.node_tree.links.new(input, output)

                        output = opacity_and_alpha.outputs[0]
                        input = layer_blend.inputs[0]
                        sxmaterial.node_tree.links.new(input, output)

                        output = layer_blend.outputs['Color']
                        input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission']
                        sxmaterial.node_tree.links.new(input, output)
                        bpy.context.scene.eevee.use_bloom = True


            if prev_color is not None:
                output = prev_color
                input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color']
                sxmaterial.node_tree.links.new(input, output)



    def update_sx2material(self, context):
        sx_mat_objs = []
        sx_mats = [value for key, value in bpy.data.materials.items() if 'SX2Material' in key]
        if len(sx_mats) > 0:
            sx_mats = [mat for mat in bpy.data.materials.keys() if 'SX2Material' in mat]

            for mat in sx_mats:
                sx_mat = bpy.data.materials.get(mat)

                for obj in context.scene.objects:
                    if obj.active_material == bpy.data.materials[mat]:
                        sx_mat_objs.append(obj.name)
                        obj.data.materials.clear()

                sx_mat.user_clear()
                bpy.data.materials.remove(sx_mat)

        bpy.context.view_layer.update()

        objs = selection_validator(self, context)
        setup.create_sx2material(objs)

        for obj in objs:
            sx_mat_objs.append(obj.name)

        for obj_name in sx_mat_objs:
            # context.scene.objects[obj_name].sxtools.shadingmode = 'FULL'
            context.scene.objects[obj_name].active_material = bpy.data.materials['SX2Material_' + obj_name]


    def __del__(self):
        print('SX Tools: Exiting setup')


# ------------------------------------------------------------------------
#    Update Functions
# ------------------------------------------------------------------------
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


def refresh_actives(self, context):
    if not sxglobals.refreshInProgress:
        sxglobals.refreshInProgress = True

        scene = context.scene.sx2
        mode = context.scene.sx2.shadingmode
        objs = selection_validator(self, context)

        utils.mode_manager(objs, set_mode=True, mode_id='refresh_actives')

        if len(objs) > 0:
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            vcols = layer.color_attribute

            # update Color Tool according to selected layer
            if scene.toolmode == 'COL':
                update_fillcolor(self, context)

            # update Palettes-tab color values
            if scene.toolmode == 'PAL':
                layers.color_layers_to_values(objs)
            elif (prefs.materialtype != 'SMP') and (scene.toolmode == 'MAT'):
                layers.material_layers_to_values(objs)

            for obj in objs:
                setattr(obj.sx2, 'selectedlayer', idx)
                if vcols != '':
                    obj.data.attributes.active = obj.data.attributes[vcols]

            # Refresh SX Tools UI to latest selection
            layers.update_layer_panel(objs, layer)

            # Update SX Material to latest selection
            if objs[0].sx2.category == 'TRANSPARENT':
                if bpy.data.materials['SXMaterial'].blend_method != 'BLEND':
                    bpy.data.materials['SXMaterial'].blend_method = 'BLEND'
                    bpy.data.materials['SXMaterial'].use_backface_culling = True
            else:
                if bpy.data.materials['SXMaterial'].blend_method != 'OPAQUE':
                    bpy.data.materials['SXMaterial'].blend_method = 'OPAQUE'
                    bpy.data.materials['SXMaterial'].use_backface_culling = False

        # Verify selectionMonitor is running
        if not sxglobals.modalStatus:
            setup.start_modal()

        utils.mode_manager(objs, revert=True, mode_id='refresh_actives')
        sxglobals.refreshInProgress = False


def refresh_swatches(self, context):
    if not sxglobals.refreshInProgress:
        sxglobals.refreshInProgress = True

        # mode = context.scene.sxtools.shadingmode
        objs = selection_validator(self, context)

        utils.mode_manager(objs, set_mode=True, mode_id='refresh_actives')

        if len(objs) > 0:
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

            # Refresh SX Tools UI to latest selection
            layers.update_layer_panel(objs, layer)

        utils.mode_manager(objs, revert=True, mode_id='refresh_actives')
        sxglobals.refreshInProgress = False


def message_box(message='', title='SX Tools', icon='INFO'):
    messageLines = message.splitlines()


    def draw(self, context):
        for line in messageLines:
            self.layout.label(text=line)

    if not bpy.app.background:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


# TODO: This only works with identical layersets
def update_selected_layer(self, context):
    objs = selection_validator(self, context)
    for obj in objs:
        if len(obj.sx2layers) > 0:
            obj.data.attributes.active_color = obj.data.attributes[obj.sx2.selectedlayer]

        sxglobals.layer_stack_dict.clear()
        layers = []
        for layer in obj.sx2layers:
            layers.append((layer.index, layer.color_attribute))

        layers.sort(key=lambda y: y[0])

        for i, layer_tuple in enumerate(layers):
            for layer in obj.sx2layers:
                if layer.color_attribute == layer_tuple[1]:
                    sxglobals.layer_stack_dict[i] = (layer.name, layer.color_attribute, layer.layer_type)

    refresh_swatches(self, context)


def update_layer_visibility(self, context):
    setup.update_sx2material(context)


def update_material_props(self, context):
    objs = selection_validator(self, context)
    if 'SX2Material' in bpy.data.materials:
        for obj in objs:
            bpy.data.materials['SX2Material'].blend_method = 'OPAQUE'
            bpy.data.materials['SX2Material'].use_backface_culling = False

            for layer in obj.sx2layers:
                if (layer.index == 0) and (layer.opacity < 1.0):
                    bpy.data.materials['SX2Material'].blend_method = 'BLEND'
                    bpy.data.materials['SX2Material'].use_backface_culling = True


def update_layer_blend_modes(self, context):
    setup.update_sx2material(context)

# ------------------------------------------------------------------------
#    Settings and properties
# ------------------------------------------------------------------------
class SXTOOLS2_objectprops(bpy.types.PropertyGroup):

    selectedlayer: bpy.props.IntProperty(
        name='Selected Layer',
        min=0,
        default=0,
        update=update_selected_layer)

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

    tiling: bpy.props.BoolProperty(
        name='Tiling Object',
        description='Creates duplicates during occlusion rendering to fix edge values',
        default=False)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_pos_x: bpy.props.BoolProperty(
        name='Tile X',
        description='Select required mirror directions',
        default=False)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_neg_x: bpy.props.BoolProperty(
        name='Tile -X',
        description='Select required mirror directions',
        default=False)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_pos_y: bpy.props.BoolProperty(
        name='Tile Y',
        description='Select required mirror directions',
        default=False)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_neg_y: bpy.props.BoolProperty(
        name='Tile -Y',
        description='Select required mirror directions',
        default=False)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_pos_z: bpy.props.BoolProperty(
        name='Tile Z',
        description='Select required mirror directions',
        default=False)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_neg_z: bpy.props.BoolProperty(
        name='Tile -Z',
        description='Select required mirror directions',
        default=False)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))

    tile_offset: bpy.props.FloatProperty(
        name='Tile Offset',
        description='Adjust offset of mirror copies',
        default=0.0)
        # update=lambda self, context: update_modifiers(self, context, 'tiling'))


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
        items=lambda self, context: dict_lister(self, context, sxglobals.rampDict))
        # update=load_ramp)

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
        default=0.5)
        # update=update_curvature_selection)

    curvaturetolerance: bpy.props.FloatProperty(
        name='Curvature Tolerance',
        min=0.0,
        max=1.0,
        default=0.1)
        # update=update_curvature_selection)

    palettecategories: bpy.props.EnumProperty(
        name='Category',
        description='Choose palette category',
        items=lambda self, context: ext_category_lister(self, context, 'sxpalettes'))

    materialcategories: bpy.props.EnumProperty(
        name='Category',
        description='Choose material category',
        items=lambda self, context: ext_category_lister(self, context, 'sxmaterials'))

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


class SXTOOLS2_layerprops(bpy.types.PropertyGroup):
    # name: from PropertyGroup

    enabled: bpy.props.BoolProperty(
        name='Layer Enabled',
        default=True)

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
            ('EMI', 'Emission', '')],
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
        update=lambda self, context: update_layer_visibility(self, context))

    opacity: bpy.props.FloatProperty(
        name='Layer Opacity',
        min=0.0,
        max=1.0,
        default=1.0,
        update=lambda self, context: update_material_props(self, context))

    blend_mode: bpy.props.EnumProperty(
        name='Layer Blend Mode',
        items=[
            ('ALPHA', 'Alpha', ''),
            ('ADD', 'Additive', ''),
            ('MUL', 'Multiply', ''),
            ('OVR', 'Overlay', '')],
        default='ALPHA',
        update=lambda self, context: update_layer_blend_modes(self, context))

    color_attribute: bpy.props.StringProperty(
        name='Vertex Color Layer',
        description='Maps a list item to a vertex color layer',
        default='')

    locked: bpy.props.BoolProperty(
        name='Lock Layer Opacity',
        default=False)


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

            if len(obj.sx2layers) == 0:
                layer = None
            else:
                layer = obj.sx2layers[sx2.selectedlayer]

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
                        (layer.name == 'emission')):
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
                paste_text = 'Swap'
                sel_text = 'Select Inverse'
            elif scene.alt:
                paste_text = 'Merge Into'
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
                if (layer is None) or (layer.layer_type == 'COLOR') or (layer.layer_type == 'EMI'):
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
                        row_tiling.prop(sx2, 'tiling', text='Tiling Object')
                        row_tiling.prop(sx2, 'tile_offset', text='Tile Offset')
                        row_tiling2 = col_fill.row(align=False)
                        row_tiling2.prop(sx2, 'tile_pos_x', text='+X')
                        row_tiling2.prop(sx2, 'tile_pos_y', text='+Y')
                        row_tiling2.prop(sx2, 'tile_pos_z', text='+Z')
                        row_tiling3 = col_fill.row(align=False)
                        row_tiling3.prop(sx2, 'tile_neg_x', text='-X')
                        row_tiling3.prop(sx2, 'tile_neg_y', text='-Y')
                        row_tiling3.prop(sx2, 'tile_neg_z', text='-Z')
                        if scene.occlusionblend == 0:
                            row_ground.enabled = False
                        if not sx2.tiling:
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
                    box_fill.template_color_ramp(bpy.data.materials['SX2Material'].node_tree.nodes['ColorRamp'], 'color_ramp', expand=True)
                    if mode == 'OBJECT':
                        box_fill.prop(scene, 'rampbbox', text='Use Combined Bounding Box')

            # Luminance Remap Tool -----------------------------------------------
            elif scene.toolmode == 'LUM':
                if scene.expandfill:
                    row3_fill = box_fill.row(align=True)
                    row3_fill.prop(scene, 'ramplist', text='')
                    row3_fill.operator('sx2.addramp', text='', icon='ADD')
                    row3_fill.operator('sx2.delramp', text='', icon='REMOVE')
                    box_fill.template_color_ramp(bpy.data.materials['SX2Material'].node_tree.nodes['ColorRamp'], 'color_ramp', expand=True)

            # Master Palettes -----------------------------------------------
            elif scene.toolmode == 'PAL':
                palettes = context.scene.sxpalettes

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
                                    mp_button = split2_mpalette.operator('sx2.applypalette', text='Apply')
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
                    materials = context.scene.sxmaterials

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

            # if scene.shadingmode == 'FULL':
            #     row_item.prop(item, 'visibility', text='', icon=hide_icon[item.visibility])
            # else:
            #     row_item.label(icon=hide_icon[(item.index == objs[0].sx2.selectedlayer)])

            # if sxglobals.mode == 'OBJECT':
            #     if scene.toolmode == 'PAL':
            #         row_item.label(text='', icon='LOCKED')
            #     else:
            #         row_item.prop(item, 'locked', text='', icon=lock_icon[item.locked])
            # else:
            #     row_item.label(text='', icon='UNLOCKED')

            row_item.prop(item, 'visibility', text='', icon=hide_icon[item.visibility])
            row_item.prop(item, 'locked', text='', icon=lock_icon[item.locked])
            row_item.label(text='  ' + item.name)
            # row_item.prop(item, 'name', text='')

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
            for layer in objs[0].sx2layers:
                flt_neworder.append(layer.index)

                # if not layer.enabled:
                #     flt_flags[idx] |= ~self.bitflag_filter_item

            return flt_flags, flt_neworder


# ------------------------------------------------------------------------
#   Operators
# ------------------------------------------------------------------------
class SXTOOLS_OT_selectionmonitor(bpy.types.Operator):
    bl_idname = 'sx2.selectionmonitor'
    bl_label = 'Selection Monitor'
    bl_description = 'Refreshes the UI on selection change'


    @classmethod
    def poll(cls, context):
        if not bpy.app.background:
            if context.area.type == 'VIEW_3D':
                return context.active_object is not None
        else:
            return context.active_object is not None


    def modal(self, context, event):
        if not context.area:
            print('Selection Monitor: Context Lost')
            sxglobals.modalStatus = False
            return {'CANCELLED'}

        if (len(sxglobals.masterPaletteArray) == 0) or (len(sxglobals.materialArray) == 0) or (len(sxglobals.rampDict) == 0) or (len(sxglobals.categoryDict) == 0):
            sxglobals.librariesLoaded = False
            load_libraries(self, context)
            return {'PASS_THROUGH'}

        objs = selection_validator(self, context)
        if len(objs) > 0:
            mode = objs[0].mode
            if mode != sxglobals.prevMode:
                # print('selectionmonitor: mode change')
                sxglobals.prevMode = mode
                sxglobals.mode = mode
                # refresh_actives(self, context)
                return {'PASS_THROUGH'}

            if (objs[0].mode == 'EDIT'):
                objs[0].update_from_editmode()
                mesh = objs[0].data
                selection = [None] * len(mesh.vertices)
                mesh.vertices.foreach_get('select', selection)
                # print('selectionmonitor: componentselection ', selection)

                if selection != sxglobals.prevComponentSelection:
                    # print('selectionmonitor: component selection changed')
                    sxglobals.prevComponentSelection = selection
                    # refresh_actives(self, context)
                    return {'PASS_THROUGH'}
                else:
                    return {'PASS_THROUGH'}
            else:
                selection = context.view_layer.objects.selected.keys()
                # print('selectionmonitor: selection ', selection)

                if selection != sxglobals.prevSelection:
                    # print('selectionmonitor: object selection changed')
                    sxglobals.prevSelection = selection[:]
                    # if (selection is not None) and (len(selection) > 0) and (len(context.view_layer.objects[selection[0]].sxlayers) > 0):
                    #     refresh_actives(self, context)
                    return {'PASS_THROUGH'}
                else:
                    return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        # bpy.app.timers.register(lambda: 0.01 if 'PASS_THROUGH' in self.modal(context, event) else None)
        sxglobals.prevSelection = context.view_layer.objects.selected.keys()[:]
        context.window_manager.modal_handler_add(self)
        print('SX Tools: Starting selection monitor')
        return {'RUNNING_MODAL'}


class SXTOOLS_OT_keymonitor(bpy.types.Operator):
    bl_idname = 'sx2.keymonitor'
    bl_label = 'Key Monitor'

    redraw: bpy.props.BoolProperty(default=False)
    prevShift: bpy.props.BoolProperty(default=False)
    prevAlt: bpy.props.BoolProperty(default=False)
    prevCtrl: bpy.props.BoolProperty(default=False)


    def modal(self, context, event):
        if not context.area:
            print('Key Monitor: Context Lost')
            sxglobals.modalStatus = False
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


class SXTOOLS2_OT_layerinfo(bpy.types.Operator):
    bl_idname = 'sx2.layerinfo'
    bl_label = 'Special Layer Info'
    bl_descriptions = 'Information about special layers'


    def invoke(self, context, event):
        message_box('SPECIAL LAYERS:\nGradient1, Gradient2\nGrayscale layers that receive color from Palette Color 4 and 5\n(i.e. the primary colors of Layer 4 and 5)\n\nOcclusion, Metallic, Roughness, Transmission, Emission\nGrayscale layers, filled with the luminance of the selected color.')
        return {'FINISHED'}


class SXTOOLS_OT_addramp(bpy.types.Operator):
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


class SXTOOLS_OT_delramp(bpy.types.Operator):
    bl_idname = 'sx2.delramp'
    bl_label = 'Remove Ramp Preset'
    bl_description = 'Delete ramp preset from Gradient Library'


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        rampEnum = context.scene.sx2.ramplist[:]
        rampName = sxglobals.presetLookup[context.scene.sx2.ramplist]
        del sxglobals.rampDict[rampName]
        del sxglobals.presetLookup[rampEnum]
        files.save_file('gradients')
        return {'FINISHED'}


class SXTOOLS_OT_addpalettecategory(bpy.types.Operator):
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


class SXTOOLS_OT_delpalettecategory(bpy.types.Operator):
    bl_idname = 'sx2.delpalettecategory'
    bl_label = 'Remove Palette Category'
    bl_description = 'Removes a palette category from the Palette Library'


    def invoke(self, context, event):
        categoryEnum = context.scene.sx2.palettecategories[:]
        categoryName = sxglobals.presetLookup[context.scene.sx2.palettecategories]

        for i, categoryDict in enumerate(sxglobals.masterPaletteArray):
            for category in categoryDict:
                if category == categoryName:
                    sxglobals.masterPaletteArray.remove(categoryDict)
                    del sxglobals.presetLookup[categoryEnum]
                    break

        files.save_file('palettes')
        files.load_file('palettes')
        return {'FINISHED'}


class SXTOOLS_OT_addmaterialcategory(bpy.types.Operator):
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


class SXTOOLS_OT_delmaterialcategory(bpy.types.Operator):
    bl_idname = 'sx2.delmaterialcategory'
    bl_label = 'Remove Material Category'
    bl_description = 'Removes a material category from the Material Library'


    def invoke(self, context, event):
        categoryEnum = context.scene.sx2.materialcategories[:]
        categoryName = sxglobals.presetLookup[context.scene.sx2.materialcategories]

        for i, categoryDict in enumerate(sxglobals.materialArray):
            for category in categoryDict:
                if category == categoryName:
                    sxglobals.materialArray.remove(categoryDict)
                    del sxglobals.presetLookup[categoryEnum]
                    break

        files.save_file('materials')
        files.load_file('materials')
        return {'FINISHED'}


class SXTOOLS_OT_addpalette(bpy.types.Operator):
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
        if paletteName in context.scene.sxpalettes:
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


class SXTOOLS_OT_delpalette(bpy.types.Operator):
    bl_idname = 'sx2.delpalette'
    bl_label = 'Remove Palette Preset'
    bl_description = 'Delete palette preset from Palette Library'

    label: bpy.props.StringProperty(name='Palette Name')


    def invoke(self, context, event):
        paletteName = self.label.replace(" ", "")
        category = context.scene.sxpalettes[paletteName].category

        for i, categoryDict in enumerate(sxglobals.masterPaletteArray):
            if category in categoryDict:
                del categoryDict[category][paletteName]

        files.save_file('palettes')
        files.load_file('palettes')
        return {'FINISHED'}


class SXTOOLS_OT_addmaterial(bpy.types.Operator):
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
        if materialName in context.scene.sxmaterials:
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


class SXTOOLS_OT_delmaterial(bpy.types.Operator):
    bl_idname = 'sx2.delmaterial'
    bl_label = 'Remove Material Preset'
    bl_description = 'Delete material preset from Material Library'

    label: bpy.props.StringProperty(name='Material Name')


    def invoke(self, context, event):
        materialName = self.label.replace(" ", "")
        category = context.scene.sxmaterials[materialName].category

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
        objs = selection_validator(self, context)
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
            layers.add_layer(objs)
            setup.update_sx2material(context)
            refresh_swatches(self, context)

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
                    bottom_layer_index = obj.sx2layers[idx].index - 1 if obj.sx2layers[idx].index - 1 > 0 else 0
                    obj.data.attributes.remove(obj.data.attributes[idx])
                    obj.sx2layers.remove(idx)
                    utils.sort_stack_indices(obj)
                    for i, layer in enumerate(obj.sx2layers):
                        if layer.index == bottom_layer_index:
                            obj.sx2.selectedlayer = i
                    print('del:', idx, 'sel:', obj.sx2.selectedlayer)

                l_list = []
                for layer in obj.sx2layers:
                    l_list.append(layer.index)
                print('indices:', l_list)

            setup.update_sx2material(context)

        return {'FINISHED'}


class SXTOOLS2_OT_layer_up(bpy.types.Operator):
    bl_idname = 'sx2.layer_up'
    bl_label = 'Layer Up'
    bl_description = 'Move layer up'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            idx = objs[0].sx2.selectedlayer
            updated = False
            for obj in objs:
                if len(obj.sx2layers) > 0:
                    new_index = obj.sx2layers[idx].index + 1 if obj.sx2layers[idx].index + 1 < len(obj.sx2layers) else obj.sx2layers[idx].index
                    obj.sx2layers[idx].index = utils.insert_layer_at_index(obj, obj.sx2layers[idx], new_index)
                    print('added at:', obj.sx2layers[idx].index)
                    updated = True

            if updated:
                setup.update_sx2material(context)

        return {'FINISHED'}


class SXTOOLS2_OT_layer_down(bpy.types.Operator):
    bl_idname = 'sx2.layer_down'
    bl_label = 'Layer Down'
    bl_description = 'Move layer down'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            idx = objs[0].sx2.selectedlayer
            updated = False
            for obj in objs:
                if len(obj.sx2layers) > 0:
                    new_index = obj.sx2layers[idx].index - 1 if obj.sx2layers[idx].index -1 >= 0 else 0
                    obj.sx2layers[idx].index = utils.insert_layer_at_index(obj, obj.sx2layers[idx], new_index)
                    print('added at:', obj.sx2layers[idx].index)
                    updated = True

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
            ('EMI', 'Emission', '')],
        default='COLOR')


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            idx = objs[0].sx2.selectedlayer
            self.layer_name = objs[0].sx2layers[idx].name
            self.layer_type = objs[0].sx2layers[idx].layer_type

            return context.window_manager.invoke_props_dialog(self)
        else:
            return {'FINISHED'}


    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'layer_name', text='Layer Name')
        col.prop(self, 'layer_type', text='Layer Type')
        return None


    def execute(self, context):
        objs = selection_validator(self, context)
        for obj in objs:
            for layer in obj.sx2layers:
                if (self.layer_type == layer.layer_type) and (self.layer_type != 'COLOR'):
                    message_box('Material channel already exists!')
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
                elif self.layer_type == 'EMI':
                    name = 'Emission'
                else:
                    name = self.layer_name

                obj.sx2layers[obj.sx2.selectedlayer].name = name
                obj.sx2layers[obj.sx2.selectedlayer].layer_type = self.layer_type
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
        if sxglobals.mode == 'EDIT':
            return enabled
        objs = context.view_layer.objects.selected
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            layer = utils.find_layer_by_stack_index(mesh_objs[0], mesh_objs[0].sx2.selectedlayer)
            if layer.layer_type == 'COLOR':
                enabled = True
        return enabled


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            topLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            baseLayer = utils.find_layer_by_stack_index(objs[0], topLayer.index - 1)
            layers.merge_layers(objs, topLayer, baseLayer, baseLayer)
            refresh_swatches(self, context)
            # sxglobals.composite = True
            # refresh_actives(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_mergedown(bpy.types.Operator):
    bl_idname = 'sx2.mergedown'
    bl_label = 'Merge Down'
    bl_description = 'Merges the selected color layer with the one below'
    bl_options = {'UNDO'}


    @classmethod
    def poll(cls, context):
        enabled = False
        if sxglobals.mode == 'EDIT':
            return enabled
        objs = context.view_layer.objects.selected
        mesh_objs = []
        for obj in objs:
            if obj.type == 'MESH':
                mesh_objs.append(obj)

        if len(mesh_objs[0].sx2layers) > 0:
            layer = mesh_objs[0].sx2layers[mesh_objs[0].sx2.selectedlayer]
            if layer.index < (len(sxglobals.layer_stack_dict) - 1):
                nextLayer = utils.find_layer_by_stack_index(mesh_objs[0], layer.index + 1)
                if (nextLayer.layer_type == 'COLOR') and (layer.layer_type == 'COLOR'):
                    enabled = True
        return enabled


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            baseLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            topLayer = utils.find_layer_by_stack_index(objs[0], baseLayer.index + 1)
            layers.merge_layers(objs, topLayer, baseLayer, topLayer)
            refresh_swatches(self, context)
            # sxglobals.composite = True
            # refresh_actives(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_copylayer(bpy.types.Operator):
    bl_idname = 'sx2.copylayer'
    bl_label = 'Copy Layer'
    bl_description = 'Mark the selected layer for copying'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            sxglobals.copyLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
        return {'FINISHED'}

 
class SXTOOLS2_OT_pastelayer(bpy.types.Operator):
    bl_idname = 'sx2.pastelayer'
    bl_label = 'Paste Layer'
    bl_description = 'Shift-click to swap with copied layer\nAlt-click to merge into the target layer'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            sourceLayer = sxglobals.copyLayer
            targetLayer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

            if event.alt and sourceLayer is not None:
                layers.merge_layers(objs, sourceLayer, targetLayer, targetLayer)
                refresh_swatches(self, context)
                # sxglobals.composite = True
                # refresh_actives(self, context)
                return {'FINISHED'}

            elif event.shift:
                mode = 'swap'
            elif event.ctrl:
                mode = 'mask'
            else:
                mode = False

            if sourceLayer is None:
                message_box('Nothing to paste!')
                return {'FINISHED'}
            else:
                layers.paste_layer(objs, sourceLayer, targetLayer, mode)
                refresh_swatches(self, context)
                # sxglobals.composite = True
                # refresh_actives(self, context)
                return {'FINISHED'}
        return {'FINISHED'}


class SXTOOLS2_OT_clearlayers(bpy.types.Operator):
    bl_idname = 'sx2.clear'
    bl_label = 'Clear Layer'
    bl_description = 'Shift-click to clear all layers\non selected objects or components\nCtrl-click to flatten all color layers to Layer 1'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            utils.mode_manager(objs, set_mode=True, mode_id='clearlayers')
            if event.ctrl:
                compLayers = utils.find_comp_layers(objs[0])
                layer0 = utils.find_layer_by_stack_index(objs[0], 1)
                layer1 = utils.find_layer_by_stack_index(objs[0], 1)
                layers.blend_layers(objs, compLayers, layer1, layer0, uv_as_alpha=True)
                for layer in compLayers:
                    layers.clear_layers(objs, layer)
            else:
                if event.shift:
                    layer = None
                else:
                    layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]

                layers.clear_layers(objs, layer)

            utils.mode_manager(objs, revert=True, mode_id='clearlayers')
            refresh_swatches(self, context)
            # sxglobals.composite = True
            # refresh_actives(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_selmask(bpy.types.Operator):
    bl_idname = 'sx2.selmask'
    bl_label = 'Select Layer Mask'
    bl_description = 'Click to select components with alpha\nShift-click to invert selection\nCtrl-click to select visible faces with Fill Color'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            bpy.context.view_layer.objects.active = objs[0]

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


class SXTOOLS_OT_setgroup(bpy.types.Operator):
    bl_idname = 'sx2.setgroup'
    bl_label = 'Set Value'
    bl_description = 'Click to apply modifier weight to selected edges\nShift-click to add edges to selection\nAlt-click to clear selection and select verts or edges'
    bl_options = {'UNDO'}

    setmode: bpy.props.StringProperty()
    setvalue: bpy.props.FloatProperty()

    def invoke(self, context, event):
        objs = selection_validator(self, context)
        scene = context.scene.sx2
        if len(objs) > 0:
            setmode = self.setmode
            setvalue = self.setvalue
            if event.shift:
                tools.select_set(objs, setvalue, setmode)
            elif event.alt:
                tools.select_set(objs, setvalue, setmode, True)
            else:
                tools.assign_set(objs, setvalue, setmode)
                if (scene.autocrease) and (setmode == 'BEV') and (setvalue != -1.0):
                    mode = bpy.context.tool_settings.mesh_select_mode[:]
                    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
                    tools.assign_set(objs, 1.0, 'CRS')
                    bpy.context.tool_settings.mesh_select_mode = mode
        return {'FINISHED'}


class SXTOOLS_OT_applypalette(bpy.types.Operator):
    bl_idname = 'sx2.applypalette'
    bl_label = 'Apply Palette'
    bl_description = 'Applies the selected palette to selected objects\nPalette colors are applied to layers 1-5'
    bl_options = {'UNDO'}

    label: bpy.props.StringProperty()


    def execute(self, context):
        return self.invoke(context, None)


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            palette = self.label
            tools.apply_palette(objs, palette)

            # if not bpy.app.background:
            #     sxglobals.composite = True
            #     refresh_actives(self, context)
        return {'FINISHED'}


class SXTOOLS_OT_applymaterial(bpy.types.Operator):
    bl_idname = 'sx2.applymaterial'
    bl_label = 'Apply PBR Material'
    bl_description = 'Applies the selected material to selected objects\nAlbedo color goes to the layer7\nmetallic and smoothness values are automatically applied\nto the selected material channels'
    bl_options = {'UNDO'}

    label: bpy.props.StringProperty()


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            material = self.label
            layer = objs[0].sx2layers[objs[0].sx2.selectedlayer]
            tools.apply_material(objs, layer, material)

            # refresh_actives(self, context)
        return {'FINISHED'}


class SXTOOLS2_OT_create_material(bpy.types.Operator):
    bl_idname = 'sx2.create_material'
    bl_label = 'Create Material'
    bl_description = 'Create Material'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        setup.update_sx2material(context)

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

classes = (
    SXTOOLS2_objectprops,
    SXTOOLS2_sceneprops,
    SXTOOLS2_layerprops,
    SXTOOLS2_PT_panel,
    SXTOOLS2_UL_layerlist,
    SXTOOLS2_OT_layerinfo,
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
    SXTOOLS2_OT_create_material)

addon_keymaps = []


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Object.sx2 = bpy.props.PointerProperty(type=SXTOOLS2_objectprops)
    bpy.types.Object.sx2layers = bpy.props.CollectionProperty(type=SXTOOLS2_layerprops)
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
