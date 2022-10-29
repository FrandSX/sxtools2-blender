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
        self.layer_list_dict = {}
        self.layer_order = []

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


    def find_layer_from_index(self, obj, index):
        for layer in obj.sx2layers:
            if layer.index == index:
                return layer
        return None


    def insert_layer_at_index(self, obj, inserted_layer, index):
        sxglobals.layer_list_dict.clear()

        if len(obj.sx2layers) == 1:
            index = 0

        layers = []
        for layer in obj.sx2layers:
            layers.append((layer.index, layer.color_attribute))

        layers.sort(key=lambda y: y[0])

        print(layers)
        layers.remove((100, inserted_layer.color_attribute))
        print(layers)
        layers.insert(index, (index, inserted_layer.color_attribute))
        print(layers)

        for i, layer_tuple in enumerate(layers):
            for layer in obj.sx2layers:
                if layer.color_attribute == layer_tuple[1]:
                    layer.index = i
                    sxglobals.layer_list_dict[i] = layer.color_attribute

        return index


    def sort_layer_indices(self, obj):
        sxglobals.layer_list_dict.clear()

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
                        sxglobals.layer_list_dict[i] = layer.color_attribute


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


    # wrapper for low-level functions, always returns layerdata in RGBA
    def get_layer(self, obj, sourcelayer, as_tuple=False, uv_as_alpha=False, apply_layer_alpha=False, gradient_with_palette=False):
        sourceType = sourcelayer.layer_type
        # sxmaterial = bpy.data.materials['SXMaterial'].node_tree
        alpha = sourcelayer.opacity

        if sourceType == 'COLOR':
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


        targetType = targetlayer.layer_type

        if targetType == 'COLOR':
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
            default_color = layer.defaultColor[:]
            if sxglobals.mode == 'OBJECT':
                setattr(obj.sx2layers[layer.index], 'alpha', 1.0)
                setattr(obj.sx2layers[layer.index], 'visibility', True)
                setattr(obj.sx2layers[layer.index], 'locked', False)
                if reset:
                    if layer == obj.sx2layers['overlay']:
                        setattr(obj.sx2layers[layer.index], 'blendMode', 'OVR')
                        default_color = (0.5, 0.5, 0.5, 1.0)
                    else:
                        setattr(obj.sx2layers[layer.index], 'blendMode', 'ALPHA')
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
                clear_layer(obj, obj.sx2layers[targetlayer.index])
                obj.data.update()


    def composite_layers(self, objs):
        if sxglobals.composite:
            # then = time.perf_counter()
            compLayers = utils.find_comp_layers(objs[0])
            shadingmode = bpy.context.scene.sx2.shadingmode
            idx = objs[0].sx2.selectedlayer
            layer = utils.find_layer_from_index(objs[0], idx)

            if shadingmode == 'FULL':
                layer0 = utils.find_layer_from_index(objs[0], 0)
                layer1 = utils.find_layer_from_index(objs[0], 1)
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
                layerIdx = layer.index

                if getattr(obj.sx2layers[layerIdx], 'visibility'):
                    blendmode = getattr(obj.sx2layers[layerIdx], 'blendMode')
                    layeralpha = getattr(obj.sx2layers[layerIdx], 'alpha')
                    topcolors = self.get_layer(obj, obj.sx2layers[layerIdx], uv_as_alpha=uv_as_alpha, gradient_with_palette=True)
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
                    if layer.blendMode == 'OVR':
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

            blendmode = toplayer.blendMode
            colors = tools.combine_layers(topcolors, basecolors, blendmode)
            self.set_layer(obj, colors, targetlayer)

        for obj in objs:
            setattr(obj.sx2layers[targetlayer.index], 'visibility', True)
            setattr(obj.sx2layers[targetlayer.index], 'blendMode', 'ALPHA')
            setattr(obj.sx2layers[targetlayer.index], 'alpha', 1.0)
            obj.sx2.selectedlayer = targetlayer.index

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
                sourceBlend = getattr(obj.sx2layers[sourceLayer.index], 'blendMode')[:]
                targetBlend = getattr(obj.sx2layers[targetLayer.index], 'blendMode')[:]
                sourceAlpha = getattr(obj.sx2layers[sourceLayer.index], 'alpha')
                targetAlpha = getattr(obj.sx2layers[targetLayer.index], 'alpha')

                if fillMode == 'swap':
                    setattr(obj.sx2layers[sourceLayer.index], 'blendMode', targetBlend)
                    setattr(obj.sx2layers[targetLayer.index], 'blendMode', sourceBlend)
                    setattr(obj.sx2layers[sourceLayer.index], 'alpha', targetAlpha)
                    setattr(obj.sx2layers[targetLayer.index], 'alpha', sourceAlpha)
                else:
                    setattr(obj.sx2layers[targetLayer.index], 'blendMode', sourceBlend)
                    setattr(obj.sx2layers[targetLayer.index], 'alpha', sourceAlpha)

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
        if layer.index <= 5:
            bpy.data.materials['SXMaterial'].node_tree.nodes['PaletteColor'+str(layer.index - 1)].outputs[0].default_value = colors[0]


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
                masklayer = utils.find_layer_from_index(objs[0], layers[0])
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


    def create_sx2material(self, layercount):

        sxmaterial = bpy.data.materials.new(name='SX2Material')
        sxmaterial.use_nodes = True
        sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Emission'].default_value = [0.0, 0.0, 0.0, 1.0]
        sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = [0.0, 0.0, 0.0, 1.0]
        sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
        sxmaterial.node_tree.nodes['Principled BSDF'].location = (0, 0)

        sxmaterial.node_tree.nodes['Material Output'].location = (300, 0)

        if layercount > 0:
            base_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
            base_color.name = 'BaseColor'
            base_color.label = 'BaseColor'
            base_color.layer_name = sxglobals.layer_list_dict[0]
            base_color.location = (-1000, -600)

            prev_color = base_color.outputs['Color']

            # Layer groups
            for i in range(layercount-1):
                source_color = sxmaterial.node_tree.nodes.new(type='ShaderNodeVertexColor')
                source_color.name = 'LayerColor' + str(i + 1)
                source_color.label = 'LayerColor' + str(i + 1)
                source_color.layer_name = sxglobals.layer_list_dict[i+1]
                source_color.location = (-800, i*500)

                layer_visibility = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                layer_visibility.name = 'Layer Visibility' + str(i + 1)
                layer_visibility.label = 'Layer Visibility' + str(i + 1)
                layer_visibility.attribute_name = 'sx2layers["' + sxglobals.layer_list_dict[i+1] + '"].int_visibility'
                layer_visibility.attribute_type = 'OBJECT'
                layer_visibility.location = (-1000, i*500)

                layer_opacity = sxmaterial.node_tree.nodes.new(type='ShaderNodeAttribute')
                layer_opacity.name = 'Layer Opacity' + str(i + 1)
                layer_opacity.label = 'Layer Opacity' + str(i + 1)
                layer_opacity.attribute_name = 'sx2layers["' + sxglobals.layer_list_dict[i+1] + '"].opacity'
                layer_opacity.attribute_type = 'OBJECT'
                layer_opacity.location = (-1000, i*500+200)

                vis_and_opacity = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                vis_and_opacity.name = 'Opacity ' + str(i + 1)
                vis_and_opacity.label = 'Opacity ' + str(i + 1)
                vis_and_opacity.operation = 'MULTIPLY'
                vis_and_opacity.use_clamp = True
                vis_and_opacity.inputs[0].default_value = 1
                vis_and_opacity.location = (-800, i*500+200)

                toggles_and_alpha = sxmaterial.node_tree.nodes.new(type='ShaderNodeMath')
                toggles_and_alpha.name = 'Opacity ' + str(i + 1)
                toggles_and_alpha.label = 'Opacity ' + str(i + 1)
                toggles_and_alpha.operation = 'MULTIPLY'
                toggles_and_alpha.use_clamp = True
                toggles_and_alpha.inputs[0].default_value = 1
                toggles_and_alpha.location = (-600, i*500+200)

                layer_blend = sxmaterial.node_tree.nodes.new(type='ShaderNodeMixRGB')
                layer_blend.name = 'Mix ' + str(i + 1)
                layer_blend.label = 'Mix ' + str(i + 1)
                layer_blend.inputs[0].default_value = 1
                layer_blend.inputs[2].default_value = [1.0, 1.0, 1.0, 1.0]
                layer_blend.blend_type = 'MIX'
                layer_blend.use_clamp = True
                layer_blend.location = (-400, i*500+200)

                # Group connections
                output = source_color.outputs['Color']
                input = layer_blend.inputs['Color2']
                sxmaterial.node_tree.links.new(input, output)   

                output = layer_opacity.outputs[2]
                input = vis_and_opacity.inputs[0]
                sxmaterial.node_tree.links.new(input, output)

                output = layer_visibility.outputs[2]
                input = vis_and_opacity.inputs[1]
                sxmaterial.node_tree.links.new(input, output)

                output = vis_and_opacity.outputs[0]
                input = toggles_and_alpha.inputs[0]
                sxmaterial.node_tree.links.new(input, output)

                output = source_color.outputs['Alpha']
                input = toggles_and_alpha.inputs[1]
                sxmaterial.node_tree.links.new(input, output)

                output = toggles_and_alpha.outputs[0]
                input = layer_blend.inputs[0]
                sxmaterial.node_tree.links.new(input, output)

                output = prev_color
                input = layer_blend.inputs['Color1']
                sxmaterial.node_tree.links.new(input, output)

                prev_color = layer_blend.outputs['Color']

            output = prev_color
            input = sxmaterial.node_tree.nodes['Principled BSDF'].inputs['Base Color']
            sxmaterial.node_tree.links.new(input, output)


    def update_sx2material(self, context):
        sx_mat_objs = []

        if 'SX2Material' in bpy.data.materials.keys():
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
        layercount = 0
        for obj in objs:
            if len(obj.sx2layers) > layercount:
                layercount = len(obj.sx2layers)
        setup.create_sx2material(layercount)

        for obj_name in sx_mat_objs:
            # context.scene.objects[obj_name].sxtools.shadingmode = 'FULL'
            context.scene.objects[obj_name].active_material = bpy.data.materials['SX2Material']
        for obj in objs:
            context.scene.objects[obj.name].active_material = bpy.data.materials['SX2Material']


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

        scene = context.scene.sxtools
        mode = context.scene.sxtools.shadingmode
        objs = selection_validator(self, context)

        utils.mode_manager(objs, set_mode=True, mode_id='refresh_actives')

        if len(objs) > 0:
            idx = objs[0].sxtools.selectedlayer
            layer = utils.find_layer_from_index(objs[0], idx)
            vcols = layer.vertexColorLayer

            # update Color Tool according to selected layer
            if scene.toolmode == 'COL':
                update_fillcolor(self, context)

            # update Palettes-tab color values
            if scene.toolmode == 'PAL':
                layers.color_layers_to_values(objs)
            elif (prefs.materialtype != 'SMP') and (scene.toolmode == 'MAT'):
                layers.material_layers_to_values(objs)

            for obj in objs:
                setattr(obj.sxtools, 'selectedlayer', idx)
                if vcols != '':
                    obj.data.attributes.active = obj.data.attributes[vcols]
                alphaVal = getattr(obj.sxlayers[idx], 'alpha')
                blendVal = getattr(obj.sxlayers[idx], 'blendMode')

                setattr(obj.sxtools, 'activeLayerAlpha', alphaVal)
                setattr(obj.sxtools, 'activeLayerBlendMode', blendVal)

            # Refresh SX Tools UI to latest selection
            layers.update_layer_panel(objs, layer)

            # Update SX Material to latest selection
            if objs[0].sxtools.category == 'TRANSPARENT':
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


def message_box(message='', title='SX Tools', icon='INFO'):
    messageLines = message.splitlines()


    def draw(self, context):
        for line in messageLines:
            self.layout.label(text=line)

    if not bpy.app.background:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def update_selected_layer(self, context):
    objs = selection_validator(self, context)
    print(sxglobals.layer_list_dict)
    print('selected layer:', objs[0].sx2.selectedlayer)
    for obj in objs:
        if len(obj.sx2layers) > 0:
            obj.data.attributes.active_color = obj.data.attributes[sxglobals.layer_list_dict[obj.sx2.selectedlayer]]


def update_layer_visibility(self, context):
    objs = selection_validator(self, context)
    for obj in objs:
        for layer in obj.sx2layers:
            if layer.int_visibility != int(layer.visibility):
                layer.int_visibility = int(layer.visibility)


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
        default=True,
        update=lambda self, context: update_layer_visibility(self, context))

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
            
            sel_idx = sx2.selectedlayer

            if len(obj.sx2layers) == 0:
                layer = None
            else:
                layer = obj.sx2layers[sel_idx]

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

            layout.operator('sx2.create_material')

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
class SXTOOLS2_OT_applytool(bpy.types.Operator):
    bl_idname = 'sx2.applytool'
    bl_label = 'Apply Tool'
    bl_description = 'Applies the selected mode fill\nto the selected components or objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)
        if len(objs) > 0:
            idx = objs[0].sx2.selectedlayer
            layer = objs[0].sx2layers[sxglobals.layer_list_dict[idx]]
            color = context.scene.sx2.fillcolor

            print(layer, color)

            tools.apply_tool(objs, layer, color=color)
            if context.scene.sx2.toolmode == 'COL':
                tools.update_recent_colors(color)

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
                item.name = 'Layer ' + str(obj.sx2.layercount)
                item.layer_type = 'COLOR'
                item.default_color = (0.0, 0.0, 0.0, 0.0)
                item.visibility = 1
                item.opacity = 1.0
                item.blend_mode = 'ALPHA'
                item.color_attribute = item.name
                item.locked = False
                item.index = 999

                obj.data.attributes.new(name=item.name, type='FLOAT_COLOR', domain='CORNER')
                item.index = utils.insert_layer_at_index(obj, item, obj.sx2layers[obj.sx2.selectedlayer].index + 1)
                print('added at:', item.index)

                obj.sx2.selectedlayer = len(obj.sx2layers) - 1
                obj.sx2.layercount += 1

                l_list = []
                for layer in obj.sx2layers:
                    l_list.append(layer.index)
                print('indices:', l_list)

                colors = generate.color_list(obj, item.default_color)
                layers.set_layer(obj, colors, layer)

        setup.update_sx2material(context)
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
                    obj.data.attributes.remove(obj.data.attributes[idx])
                    obj.sx2layers.remove(idx)
                    utils.sort_layer_indices(obj)
                    obj.sx2.selectedlayer = 0 if (idx - 1 < 0) else idx - 1
                    print('del:', idx, 'sel:', obj.sx2.selectedlayer)

                l_list = []
                for layer in obj.sx2layers:
                    l_list.append(layer.index)
                print('indices:', l_list)

        setup.update_sx2material(context)

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
    SXTOOLS2_OT_applytool,
    SXTOOLS2_OT_add_layer,
    SXTOOLS2_OT_del_layer,
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
