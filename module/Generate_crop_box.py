# -*- coding: utf-8 -*-
import arcpy
import os
from collections import defaultdict
from arcpy.sa import Reclassify, RemapRange
import sys  


def find_covering_tif(tif_list, cx, cy):
    for tif in tif_list:
        extent = arcpy.Describe(tif).extent
        if extent.XMin <= cx <= extent.XMax and extent.YMin <= cy <= extent.YMax:
            return tif
    return None

def del_floder(temp_list):
    for item in temp_list:
        try:
            if arcpy.Exists(item):
                arcpy.Delete_management(item)
        except Exception as e:
            print("删除失败：", item, str(e))
    temp_list = []
    return temp_list


polygon_fc = r"E:\shipdatasets\ceshi.gdb\ship"
workspace = r"E:\shipdatasets\ceshi.gdb"
output_path = r"E:\shipdatasets\crop_box\FJcrop_box.shp"

arcpy.env.workspace = workspace
arcpy.env.overwriteOutput = True

if arcpy.Exists(output_path):
    arcpy.Delete_management(output_path)

spatial_ref = arcpy.SpatialReference(4326)  # WGS 84
arcpy.CreateFeatureclass_management(
    out_path=os.path.dirname(output_path),
    out_name=os.path.basename(output_path),
    geometry_type="POLYLINE",
    spatial_reference=spatial_ref
)


temp_files = []

buffer_fc = os.path.join(workspace, "buffer_fc")
arcpy.Buffer_analysis(polygon_fc, buffer_fc, "20 Meters", dissolve_option="ALL")
temp_files.append(buffer_fc)

singlepart_fc = os.path.join(workspace, "buffer_single")
arcpy.MultipartToSinglepart_management(buffer_fc, singlepart_fc)
temp_files.append(singlepart_fc)

dissolve_fc = os.path.join(workspace, "fjdissolved")
arcpy.Dissolve_management(singlepart_fc, dissolve_fc, multi_part="SINGLE_PART")

arcpy.AddField_management(dissolve_fc, "GroupID", "LONG")
with arcpy.da.UpdateCursor(dissolve_fc, ["OID@", "GroupID"]) as cursor:
    for oid, _ in cursor:
        cursor.updateRow([oid, oid])


join_fc = os.path.join(workspace, "joined")
arcpy.SpatialJoin_analysis(polygon_fc, dissolve_fc, join_fc, join_type="KEEP_COMMON")
temp_files.append(join_fc)


group_dict = defaultdict(list)
with arcpy.da.SearchCursor(join_fc, ["GroupID", "OID@"]) as cursor:
    for group_id, oid in cursor:
        group_dict[group_id].append(oid)

meters_to_degrees = 10 / 111320.0



i = 0
totall = len(group_dict)
for group_id, oid_list in group_dict.items():
    sys.stdout.write('\r{}/{}'.format(i, totall))
    sys.stdout.flush()  
    where_clause = "OBJECTID IN ({})".format(",".join(map(str, oid_list)))
    group_layer = "group_layer_{}".format(group_id)
    arcpy.MakeFeatureLayer_management(polygon_fc, group_layer, where_clause)

    merged_fc = os.path.join(workspace, "merged_{}".format(group_id))
    arcpy.Dissolve_management(group_layer, merged_fc)
    temp_files.append(merged_fc)

    extent = arcpy.Describe(merged_fc).extent
    cx = (extent.XMin + extent.XMax) / 2.0
    cy = (extent.YMin + extent.YMax) / 2.0
    half_size = max(extent.width, extent.height) / 2.0 + meters_to_degrees
    clip_extent = arcpy.Extent(cx - half_size, cy - half_size, cx + half_size, cy + half_size)

    # target_tif = find_covering_tif(cx, cy)
    array = arcpy.Array([
        arcpy.Point(clip_extent.XMin, clip_extent.YMin),
        arcpy.Point(clip_extent.XMin, clip_extent.YMax),
        arcpy.Point(clip_extent.XMax, clip_extent.YMax),
        arcpy.Point(clip_extent.XMax, clip_extent.YMin),
        arcpy.Point(clip_extent.XMin, clip_extent.YMin)  
    ])
    polyline = arcpy.Polyline(array, spatial_ref)
    with arcpy.da.InsertCursor(output_path, ["SHAPE@"]) as cursor:
        cursor.insertRow([polyline])
    
    temp_files = del_floder(temp_files)
    i += 1
sys.stdout.write('\n')