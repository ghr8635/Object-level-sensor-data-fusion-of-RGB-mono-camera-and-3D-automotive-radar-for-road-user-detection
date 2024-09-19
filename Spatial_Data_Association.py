from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from hungarian import * 
from pprint import pprint 
from PIL import Image, ImageDraw
import os
import cv2
import torch
import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Radar_Clustering_CustomDBScan import *


# Sensor Calibration Dictionary 
def get_sensor_calibration(calibration_file):
    sensor_calibration_dict = {
        "camera_intrinsics": [],
        "camera_distcoeffs": [],
        "radar_to_camera": [],
        "radar_to_lidar": [],
        "lidar_to_ground": [],
        "camera_to_ground": []
    }

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    for item in data['calibration']:
        if item['calibration'] == 'camera_01':
            sensor_calibration_dict['camera_intrinsics'] = item['k']
            sensor_calibration_dict['camera_distcoeffs'] = item['D']
        elif item['calibration'] == 'radar_01_to_camera_01':
            sensor_calibration_dict['radar_to_camera'] = item['T']
        elif item['calibration'] == 'radar_01_to_lidar_01':
            sensor_calibration_dict['radar_to_lidar'] = item['T']
        elif item['calibration'] == 'lidar_01_to_ground':
            sensor_calibration_dict['lidar_to_ground'] = item['T']
        elif item['calibration'] == 'camera_01_to_ground_homography':
            sensor_calibration_dict['camera_to_ground'] = item['T']
    return sensor_calibration_dict


# Fetch Results from YOLO Predictions
def class_box_generator_for_pred(prediction_results):
    for result in prediction_results:
        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        detection = result.boxes.xyxy.cpu().numpy()

        list_of_pred_boxes = np.column_stack((cls, detection, conf))
    return list_of_pred_boxes


# Calibration: Radar Plane to Ground Plane
def radar_to_ground_transfomer(points_array, T_radar_to_lidar, T_lidar_to_ground):

    n_p_array = np.array(points_array).reshape(1,-1)
    tranposed_array = np.transpose(n_p_array)
   
    row_of_ones = np.ones((1, 1))                                                   #1x1
    stacked_matrix = np.vstack((tranposed_array, row_of_ones))  
  
    radar_to_lidar_matrix = np.matmul(T_radar_to_lidar, stacked_matrix)             #3x1

    new_stacked_matrix = np.vstack((radar_to_lidar_matrix, row_of_ones))            #4x1
    in_ground_data = np.matmul(T_lidar_to_ground, new_stacked_matrix)

    in_ground = np.transpose(in_ground_data)

    return in_ground[0]


# Radar Dictionary: on Ground
def radar_to_ground(radar_dict, sensor_calibration_dict):
    
    T = sensor_calibration_dict['radar_to_lidar']
    K = sensor_calibration_dict['lidar_to_ground']

    in_radar = radar_dict
    in_ground = {'clusters': [], 'noise': []}
    for key, value in in_radar.items():
        if key == 'clusters':
            for point in value:
                if point:
                    updated_centroid = radar_to_ground_transfomer(point[0], T, K)
                    updated_lowest_point = radar_to_ground_transfomer(point[1], T, K)
                    updated_velocity = point[2]
                    updated_point = [list(updated_centroid), list(updated_lowest_point), list(updated_velocity)]

                    if key in in_ground:
                        in_ground[key].append(updated_point)
                    else:
                        print('no key exist')
        else:
            for point in value:
                if point:
                    updated_centroid = radar_to_ground_transfomer(point[0], T, K)
                    updated_velocity = point[1]
                    updated_point = [list(updated_centroid), list(updated_velocity)]

                    if key in in_ground:
                        in_ground[key].append(updated_point)
                    else:
                        print('no key exist')
                    
    return in_ground


# Calibration: Radar to Image Plane
def radar_to_camera_transformer(radar_point, T, k):
   
    n_p_array = np.array(radar_point).reshape(1,-1)
    transpose_RPA = np.transpose(n_p_array)

    new_array = np.vstack([transpose_RPA, np.ones((1, 1))])             
    product_1 = np.matmul(np.array(k), np.array(T))

    product_array = np.matmul(product_1, new_array)                      #[su, sv, s] but along column

    final_array = product_array / product_array [2]                      #[u, v, 1], along column

    u_v = np.delete(final_array, 2, axis = 0)                            #[u, v], along column      
    final_u_v = np.transpose(u_v)

    return final_u_v[0]


# Radar Dictionary: on Image
def radar_to_camera(radar_output, sensor_calibration_dict):
    T =  sensor_calibration_dict['radar_to_camera']
    K = sensor_calibration_dict['camera_intrinsics']
    
    in_radar = radar_output
    in_camera = {'clusters': [], 'noise': []}
    for key, value in in_radar.items():
        if key == 'clusters':
            for point in value:
                if point:
                    updated_centroid = radar_to_camera_transformer(point[0], T, K)
                    updated_lowest_point = radar_to_camera_transformer(point[1], T, K)
                    updated_velocity = point[2]
                    updated_point = [list(updated_centroid), list(updated_lowest_point), list(updated_velocity)]

                    if key in in_camera:
                        in_camera[key].append(updated_point)
                    else:
                        print('no key exist')

        if key == 'noise':
            for point in value:
                if point:
                    updated_centroid = radar_to_camera_transformer(point[0], T, K)
                    updated_velocity = point[1]
                    updated_point = [list(updated_centroid), list(updated_velocity)]

                    if key in in_camera:
                        in_camera[key].append(updated_point)
                    else:
                        print('no key exist')

    return in_camera


# Homography: Image Plane to Ground Plane
def homography(points_on_image, sensor_calibration_dict):
    points = np.array(points_on_image).reshape(1, -1) 
    transpose_matrix = np.vstack((np.transpose(points),np.ones((1,1))))
    homogeneous_coordinates = np.matmul(sensor_calibration_dict['camera_to_ground'], transpose_matrix)
    ground_coordinates = homogeneous_coordinates / homogeneous_coordinates[-1].reshape(1, -1)
    transpose_ground_coordinates = ground_coordinates.T
    g_x1y1 = transpose_ground_coordinates[0][:2]
    return g_x1y1


# Visualization: Camera Points on the Ground
def camera_plotting(image_on_ground, my_plot):
        x_plotting_list = []
        y_plotting_list = []

        for xy in image_on_ground:
                x_coords = [xy[1][0]]
                y_coords = [xy[1][1]]

                x_plotting_list.append(x_coords)
                y_plotting_list.append(y_coords)
        
        colors = ['blue', 'green', 'orange', 'black', 'purple', 'maroon']

        for i, (x_co, y_co) in enumerate(zip(x_plotting_list, y_plotting_list)):
                my_plot.scatter(y_co, x_co, color=colors[i], label= 'camera', marker='o')

        my_plot.set_xlim(-30,30)
        my_plot.set_ylim(0,100)
        my_plot.set_xlabel('Y-axis')
        my_plot.set_ylabel('X-axis')
        my_plot.set_title('Plot of Points')
        my_plot.invert_xaxis()
        
        return my_plot


# Visualization: Radar Points on the Ground
def radar_plotting(dict, my_plot):
    clusters = dict['clusters']
    noise_points = dict['noise']

    x_lowest = []
    y_lowest = []

    x_noise = []
    y_noise = []

    for detection in clusters:
        if len(detection) != 0:
            lowest_point = detection[1]
            x_lp = lowest_point[0]
            y_lp = lowest_point[1]
            x_lowest.append(x_lp)
            y_lowest.append(y_lp)
    
    for noise in noise_points:
        if len(noise) != 0:
            lowest_point = noise[0]
            x_n = lowest_point[0]
            y_n = lowest_point[1]
            x_noise.append(x_n)
            y_noise.append(y_n)


    my_plot.scatter(y_lowest, x_lowest, color='red', label='lowest point', marker="X")
    my_plot.scatter(y_noise, x_noise, color='grey', label='Noise', marker=".")

    if not my_plot.get_legend():
        my_plot.legend()

    return my_plot


# Expand bounding boxes for Cluster tolerances
def expand_bbox(box, scale):
    # Calculate the width and height of the original box
    width = box[2] - box[0]     # x2 - x1
    height = box[3] - box[1]    # y2 - y1

    # Calculate the center of the original box
    center_x = box[0] + (width/2)
    center_y = box[1] + (height/2)

    # Calculate the increase in width and height
    new_width = width * scale
    new_height = height * scale

    # Calculate the new coordinates
    new_x1 = 0 if (center_x - new_width / 2) < 0 else (center_x - new_width / 2)
    new_y1 = 0 if (center_y - new_height / 2) < 0 else (center_y - new_height / 2)
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    return list([new_x1, new_y1, new_x2, new_y2])


# Calculate Euclidean Distance on Ground Plane
def get_euclidean_distance(clusters, images):
    d = np.sqrt(((clusters[0] - images[0])**2) + ((clusters[1] - images[1])**2))
    return d


# Form an Association Matrix of Bouding boxes and Clusters/Noise
def get_association_matrix(list_of_pred_boxes, cluster_on_image, datatype='clusters', association_list=None):

    clusters = list(cluster_on_image['clusters']) if datatype == 'clusters' else list(cluster_on_image['noise'])
    pred_boxes = list(list_of_pred_boxes)

    if len(clusters) > 0 and len(pred_boxes) > 0:
        matrix = np.zeros((len(clusters), len(pred_boxes)))

        for pred_idx, prediction in enumerate(pred_boxes):
            old_bbox = prediction[1:5]  
            bbox = expand_bbox(old_bbox, scale=1.2)

            for cluster_idx, cluster in enumerate(clusters):
                cluster_centroid = cluster[0]
                
                if datatype == 'clusters':  
                    if bbox[0] < cluster_centroid[0] < bbox[2] and bbox[1] < cluster_centroid[1] < bbox[3]:
                        matrix[cluster_idx, pred_idx] = 1
                    else: 
                        matrix[cluster_idx, pred_idx] = 0

                elif datatype == 'noise':
                    if pred_idx in association_list['non_associated_bbox']:
                        if bbox[0] < cluster_centroid[0] < bbox[2] and bbox[1] < cluster_centroid[1] < bbox[3]:
                            matrix[cluster_idx, pred_idx] = 1
                        else: 
                            matrix[cluster_idx, pred_idx] = 0

        return matrix 
    

# Fitler Cases based on Matrix
def get_filtered_cases(matrix, datatype='clusters', association_list=None):
    """
    Checks and assigns different cases of radar-image data for spatial association 

    Examples: 
    >>> import pprint
    >>> matrix = np.array([
    ...     [1, 0, 1, 0, 0, 0],
    ...     [0, 1, 1, 0, 0, 0],
    ...     [1, 1, 0, 0, 0, 0],
    ...     [0, 0, 0, 1, 0, 0],
    ...     [0, 0, 0, 0, 1, 0],
    ...     [0, 0, 0, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 0]
    ... ])
    >>> pprint.pprint(get_associations(matrix))
    {'many_radar_to_many_image': {'cols': [0, 1, 2], 'rows': [0, 1, 2]},
     'many_radar_to_one_image': {'cols': [4], 'rows': [(array([4, 5]),)]},
     'one_radar_to_many_image': {'cols': [], 'rows': []},
     'one_radar_to_one_image': {'cols': [3], 'rows': [3]}}
    """ 
    associations = {
        "many_cluster_to_many_bbox" : {"clusters": [], "bbox": []}, 
        "many_cluster_to_one_bbox"  : {"clusters": [], "bbox": []},
        "one_cluster_to_many_bbox"  : {"clusters": [], "bbox": []}, 
        "one_cluster_to_one_bbox"   : {"assigned": []},
        "unassigned_bbox" : {"bbox": []}
    }

    # MANY TO MANY CHECKS
    # -------------------
    rows_with_multiple_truths = np.where(np.sum(matrix, axis=1) > 1)[0] 
    columns_with_multiple_truths = np.where(np.sum(matrix, axis=0) > 1)[0] 
    # many_too_many = list(set(rows_with_multiple_truths) & set(columns_with_multiple_truths))
    # many_radar_to_many_image = [many_too_many, many_too_many]
    many_too_many_rows = set() 
    many_too_many_cols = set() 
    for r_id in range(matrix.shape[0]):
        for c_id in range(matrix.shape[1]):
            if r_id in rows_with_multiple_truths and c_id in columns_with_multiple_truths: 
                if matrix[r_id, c_id] == 1:
                     many_too_many_rows.add(r_id)
                     many_too_many_cols.add(c_id)

    associations['many_cluster_to_many_bbox']["clusters"] = list(many_too_many_rows)
    associations['many_cluster_to_many_bbox']["bbox"] = list(many_too_many_cols)

    # MANY TO ONE CHECKS 
    # -------------------
    many_to_one = [] 
    for c in range(matrix.shape[1]): 
        if c in columns_with_multiple_truths and c not in many_too_many_cols:
            associated_rows = np.where(matrix[:, c] > 0)[0]
            associations['many_cluster_to_one_bbox']["clusters"].append(associated_rows.tolist())
            associations['many_cluster_to_one_bbox']["bbox"].append([c])            

    # ONE TO MANY CHECKS
    # ------------------
    one_to_many = [] 
    for r in range(matrix.shape[0]): 
        if r in rows_with_multiple_truths and r not in many_too_many_rows:
            associated_cols = np.where(matrix[r] > 0)[0]
            associations['one_cluster_to_many_bbox']["clusters"].append([r])
            associations['one_cluster_to_many_bbox']["bbox"].append(associated_cols.tolist())

    # ONE TO ONE CHECKS
    # ------------------
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i,j] == 1:
                row_sum = sum(matrix[i,:])
                col_sum = sum(matrix[:,j]) 

                if row_sum == 1 and col_sum == 1:
                    associations['one_cluster_to_one_bbox']['assigned'].append([i, j]) 

    if datatype == 'clusters':
        associations["unassigned_bbox"]["bbox"].extend(list(np.where(np.sum(matrix, axis=0) == 0)[0]))
    elif datatype == 'noise':
        associations["unassigned_bbox"]["bbox"].extend(box for box in list(np.where(np.sum(matrix, axis=0) == 0)[0]) if box in association_list['non_associated_bbox']) 

    return associations


# Carry One to one Association and Unassigned Bounding boxes
def get_one_to_one_association(filtered_cases, association_list):
    association_list["associated"].extend(filtered_cases['one_cluster_to_one_bbox']['assigned']) 
    association_list["non_associated_bbox"].extend(filtered_cases["unassigned_bbox"]["bbox"]) 
    return association_list


# Association of One cluster/Noise to Many Bounding boxes
def get_one_to_many_association(filtered_cases, clusters_on_ground, image_on_ground, datatype='clusters', association_list=None):

    if datatype == 'clusters':
        lower_idx = 1
    elif datatype == 'noise':
        lower_idx = 0
    else:
        print('KeyError')

    list_of_centroid_indices = filtered_cases['one_cluster_to_many_bbox']['clusters']
    list_of_box_indices = filtered_cases['one_cluster_to_many_bbox']['bbox']

    for centroid, bboxes in zip(list_of_centroid_indices, list_of_box_indices):
        centroid = centroid[0]

        euclidean_distances = []
        for box in bboxes:
            ec_distance = get_euclidean_distance(clusters_on_ground[datatype][centroid][lower_idx], image_on_ground[box][1])
            euclidean_distances.append(ec_distance)
        
        min_bbox_idx = [idx for idx, value in enumerate(euclidean_distances) if value == min(euclidean_distances)]
        min_bbox_idx = min_bbox_idx[0]

        association_list['associated'].append([centroid, bboxes[min_bbox_idx]])
        association_list['non_associated_bbox'].extend([bbox for bbox in bboxes if bbox != bboxes[min_bbox_idx]])

    return association_list


# Association of Many cluster/Noise to One Bounding box
def nearest_point_finder(points_list, processed_radar_points_to_ground):
    x_list = []
    for point in points_list:
        x_p = processed_radar_points_to_ground['clusters'][point][1][0]
        x_list.append(x_p)
    
    return x_list.index(min(x_list))

def get_many_to_one_association(filtered_cases, association_list, clusters_on_ground, image_on_ground, datatype='clusters'):

    clusters = filtered_cases['many_cluster_to_one_bbox']['clusters']
    pre_boxes = filtered_cases['many_cluster_to_one_bbox']['bbox']
  
    for cluster, box in zip(clusters, pre_boxes):
    # if count_in_clusters == 1 and count_in_boxes == 1:
        box_data = box[0]
        clusters_data = cluster
        count = len(clusters_data)
        matrix = np.zeros((count, count))

        for index_p1, point_1 in enumerate(clusters_data):
            for index_p2, point_2 in enumerate(clusters_data):
                point_1_data = clusters_on_ground['clusters'][point_1] if datatype == 'clusters' else clusters_on_ground['noise'][point_1]
                point_2_data = clusters_on_ground['clusters'][point_2] if datatype == 'clusters' else clusters_on_ground['noise'][point_2]

                velocity_p1 = point_1_data[2][0] if datatype == 'clusters' else point_1_data[1][0]
                velocity_p2 = point_2_data[2][0] if datatype == 'clusters' else point_2_data[1][0]

                x_p1 = point_1_data[0][0]
                x_p2 = point_2_data[0][0]
                
                if abs(velocity_p1 - velocity_p2) < 0.75 and abs(x_p1 - x_p2) < 2:
                    matrix[index_p1][index_p2] = 1

        candidates = []
        for row in matrix:
            columns_with_ones = []
            for col_index, value in enumerate(row):
                if value == 1:
                    columns_with_ones.append(col_index)
        
            candidates.append(columns_with_ones)

        new_candidates = candidates
        global_merged_any = False
        while True:
            check_list = new_candidates
            for i in range(len(new_candidates)):
                local_merged_any = False
                set1 = set(new_candidates[i])
                for j in range(i + 1, len(new_candidates)):
                    set2 = set(new_candidates[j])

                    if set1 & set2:  # Check if there's any common element
                        new_element = list(set1 | set2)
                        remaining_points = [point for point in new_candidates if point not in [new_candidates[i], new_candidates[j]]]
                        new_candidates = remaining_points
                        new_candidates.insert(0, new_element) # New candidate is not emptying --no break condition 

                        local_merged_any = True 
                        global_merged_any = True
                        break
                if local_merged_any:
                    break

            # Exit condition for the while loop
            if len(new_candidates) == len(check_list):
                break
                    
        if not global_merged_any:  # If no merges occurred in this iteration, terminate the loop
            distance_data = []
            for idx, centroid in enumerate(clusters_data):
                nearest_data = clusters_on_ground['clusters'][centroid][1] if datatype == 'clusters' else clusters_on_ground['noise'][centroid][0]
                box_bottom_center = image_on_ground[box_data][1]
                ec_distance = get_euclidean_distance(nearest_data[0:2], box_bottom_center)
                distance_data.append(ec_distance)
            
            index_to_look_for_in_clusters_data = distance_data.index(min(distance_data))
            index_of_chosen_centroid = clusters_data[index_to_look_for_in_clusters_data]

            if [index_of_chosen_centroid, box_data] not in association_list['associated']:
                association_list['associated'].append([index_of_chosen_centroid, box_data])
            print('not merged case')

        new_updated_candidate = []
        for box_list in new_candidates:
            updated = []
            for box in box_list:
                updated.append(clusters_data[box]) 
            new_updated_candidate.append(updated)
            
        if len(new_updated_candidate) > 1 and global_merged_any:

            #This section removes the items which are not clustered
            for pair in new_updated_candidate[:]:
                if len(pair) < 2:
                    new_updated_candidate.remove(pair)    

            distance_comparison = []
            for item in new_updated_candidate:
                nearest_point = nearest_point_finder(item, clusters_on_ground)
                nearest_point_data = clusters_on_ground['clusters'][item[nearest_point]][1] if datatype == 'clusters' else clusters_on_ground['noise'][item[nearest_point]][0]
                box_bottom_center = image_on_ground[box_data][1]
                e_c_distance = get_euclidean_distance(nearest_point_data[0:2], box_bottom_center)
                distance_comparison.append(e_c_distance)

            final_cluster_to_associate= distance_comparison.index(min(distance_comparison))
            index_of_chosen_centroid = new_updated_candidate[final_cluster_to_associate]

            association_list['associated'].append([index_of_chosen_centroid, box_data])

        elif len(new_updated_candidate) == 1:
            association_list['associated'].append([new_updated_candidate[0], box_data])
        
    return association_list


# Make Annotations on the image (Bounding Box, Centroid of Clusters/Noise...)
def get_drawings(colour_idx, image, box_data, centroid_data, datatype='clusters'):

    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0)   # Olive
    ]

    # Calculate the Corners
    original_box = box_data[1:5]
    # pprint(original_box)
    corner_1 = tuple(map(int, [original_box[0], original_box[1]]))
    corner_2 = tuple(map(int, [original_box[2], original_box[3]]))            

    # Draw Original bounding box
    colour = colors[colour_idx]
    thickness_of_box = 3
    cv2.rectangle(image, corner_1, corner_2, colour, thickness_of_box)

    # expand bounding boxes and calculate the corners
    new_box = expand_bbox(box_data[1:5], scale=1.2) 
    corner_1 = tuple(map(int, [new_box[0], new_box[1]]))
    corner_2 = tuple(map(int, [new_box[2], new_box[3]]))
    

    # Draw a Expanded bounding box with dashed Rectangle 
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    thickness = 3
    dash_length = 10

    for i in range(corner_1[0], corner_2[0], dash_length * 2):
        for t in range(thickness):
            draw.line([(i, corner_1[1]+t), (i + dash_length, corner_1[1]+t)], fill=colour[::-1])
            draw.line([(i, corner_2[1]+t), (i + dash_length, corner_2[1]+t)], fill=colour[::-1])

    for i in range(corner_1[1], corner_2[1], dash_length * 2):
        for t in range(thickness):
            draw.line([(corner_1[0]+t, i), (corner_1[0]+t, i + dash_length)], fill=colour[::-1])
            draw.line([(corner_2[0]+t, i), (corner_2[0]+t, i + dash_length)], fill=colour[::-1])

    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    
    # Plot Centroid
    thickness = -1  # to fill the circle
    radius = 15
    
    if datatype == 'clusters':
        cluster_point = tuple(map(int, [centroid_data[0], centroid_data[1]]))
        image = cv2.circle(image, cluster_point, radius, colour, thickness)
    
    elif datatype == 'noise':     
        cluster_point_1 = tuple(map(int, [centroid_data[0]-12, centroid_data[1]-12]))
        cluster_point_2 = tuple(map(int, [centroid_data[0]+12, centroid_data[1]+12]))
        image = cv2.rectangle(image, cluster_point_1, cluster_point_2, colour, thickness=3)
    
    elif datatype == 'unassigned':
        pass

    return image


# Visualization: Associated Objects on Image Plane
def get_image_visualization(final_association_dict, list_of_pred_boxes, clusters_on_image, img):

    image = cv2.imread(img, cv2.IMREAD_COLOR)
    colour_idx = 0

    if final_association_dict['with_cluster']:
        for associated_point in final_association_dict['with_cluster']:

            # Get box and centroid data
            box_data = list_of_pred_boxes[associated_point[1]]

            if type(associated_point[0]) == int:
                point = associated_point[0]
                centroid_data = clusters_on_image['clusters'][point][0]
                
                # Get Annotations
                image = get_drawings(colour_idx, image, box_data, centroid_data, datatype='clusters')
                colour_idx += 1
            
            elif type(associated_point[0]) == list:
                for point in associated_point[0]:
                    centroid_data = clusters_on_image['clusters'][point][0]
                    
                    # Get Annotations
                    image = get_drawings(colour_idx, image, box_data, centroid_data, datatype='clusters')
                colour_idx += 1


    if final_association_dict['with_noise']:
        for associated_point in final_association_dict['with_noise']:

            # Get box and centroid data
            box_data = list_of_pred_boxes[associated_point[1]]

            if type(associated_point[0]) == int:
                point = associated_point[0]
                centroid_data = clusters_on_image['noise'][point][0]
                
                # Get Annotations
                image = get_drawings(colour_idx, image, box_data, centroid_data, datatype='noise')
                colour_idx += 1
            
            elif type(associated_point[0]) == list:
                for point in associated_point[0]:
                    centroid_data = clusters_on_image['noise'][point][0]
                    
                    # Get Annotations
                    image = get_drawings(colour_idx, image, box_data, centroid_data, datatype='noise')
                colour_idx += 1


    if final_association_dict['unassigned_bbox']:
        for non_associated_point in final_association_dict['unassigned_bbox']:
            # Get box and centroid data
            box_data = list_of_pred_boxes[non_associated_point]
            centroid_data = None

            # Get Annotations
            image = get_drawings(colour_idx, image, box_data, centroid_data, datatype='unassigned')
            colour_idx += 1

    return image


# The Function Owner
def main():    

    path_to_images = Path(r'C:\Dk\Projects\Team Project\Dataset\INFRA-3DRC-Dataset\INFRA-3DRC_scene-15\camera_01\camera_01__data')
    path_to_pcd = Path(r'C:\Dk\Projects\Team Project\Dataset\INFRA-3DRC-Dataset\INFRA-3DRC_scene-15\radar_01\radar_01__data')
    calibration_file = Path(r"C:\Dk\Projects\Team Project\Dataset\INFRA-3DRC-Dataset\INFRA-3DRC_scene-15\calibration.json")
    scene_image = sorted(list(image for image in path_to_images.iterdir()))
    scene_pcd = sorted(list(image for image in path_to_pcd.iterdir()))
    yolo_model = YOLO(r"C:\Dk\Projects\Team Project\YOLO detection\Models\Harshit_Large\large_300 epoch_batch 4_augmented\train32\weights\best.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sensor_calibration_dict = get_sensor_calibration(calibration_file)

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    fig = plt.figure(figsize=(16, 9))

    my_image = fig.add_subplot(gs[0])
    my_plot = fig.add_subplot(gs[1])

    for img, pcd in zip(scene_image, scene_pcd):

        # YOLO prediction
        results = yolo_model.predict(img)
        list_of_pred_boxes = class_box_generator_for_pred(results)

        # Cluster Formation
        db_scan = my_custom_dbscan(eps1=0.1, eps2=0.250, min_samples=2)
        clusters_on_radar = db_scan.process_pcd_files(pcd)
        
        # Bbox lower center point on the Ground Plane
        image_on_ground = []
        for result in list_of_pred_boxes:
            cls = result[0]
            bbox = list(result[1:5])
            bottom_center_point = list(((bbox[2] + bbox[0]) / 2, bbox[3]))
            image_point_on_ground = homography(bottom_center_point, sensor_calibration_dict)
            image_on_ground.append([[cls], list(image_point_on_ground)])

        # Radar Dictionaries on different Planes
        clusters_on_ground = radar_to_ground(clusters_on_radar, sensor_calibration_dict)
        clusters_on_image = radar_to_camera(clusters_on_radar, sensor_calibration_dict)

        
        # Fiter Cases with Cluster Association Matrix
        association_matrix = get_association_matrix(list_of_pred_boxes, clusters_on_image, datatype= 'clusters') 
        association_list = {"associated": [], "non_associated_bbox": []}

        if association_matrix is not None:
            filtered_cases = get_filtered_cases(association_matrix,  datatype='clusters')

            # Association: One to One
            association_list = get_one_to_one_association(filtered_cases, association_list)

            # Association: One to Many
            association_list = get_one_to_many_association(filtered_cases, clusters_on_ground, image_on_ground, association_list=association_list, datatype= 'clusters')

            # Association: Many to One
            association_list = get_many_to_one_association(filtered_cases, association_list, clusters_on_ground, image_on_ground, datatype='clusters')

            # Association: Many to Many

            
        # Fiter Cases with Noise Association Matrix
        noise_association_matrix = get_association_matrix(list_of_pred_boxes, clusters_on_image, association_list=association_list, datatype='noise')
        noise_association_list = {"associated": [], "non_associated_bbox": []}

        if noise_association_matrix is not None:
            noise_filtered_cases = get_filtered_cases(noise_association_matrix, datatype='noise', association_list=association_list)

            # Association: One to One
            noise_association_list = get_one_to_one_association(noise_filtered_cases, noise_association_list)

            # Association: One to Many
            noise_association_list = get_one_to_many_association(noise_filtered_cases, clusters_on_ground, image_on_ground, association_list=noise_association_list, datatype= 'noise')

            # Association: Many to One
            noise_association_list = get_many_to_one_association(noise_filtered_cases, noise_association_list, clusters_on_ground, image_on_ground, datatype='noise')

            # Association: Many to Many



        # Final Association List
        final_association_dict = {'with_cluster': [], 'with_noise': [], 'unassigned_bbox': []}
        final_association_dict['with_cluster'].extend(association_list['associated'])
        final_association_dict['with_noise'].extend(noise_association_list['associated'])
        final_association_dict['unassigned_bbox'].extend(noise_association_list['non_associated_bbox'])
        pprint(final_association_dict)



        # Visualization:
        my_image.clear()
        my_plot.clear()

        # Points on Ground Plane
        plot = camera_plotting(image_on_ground, my_plot)
        plot2 = radar_plotting(clusters_on_ground, plot)

        # Association Visualization on Image Plane
        image_visualize = get_image_visualization(final_association_dict, list_of_pred_boxes, clusters_on_image, img)
        # cv2.imshow('on image plane', image_visualize)
        image_visualize = cv2.cvtColor(image_visualize, cv2.COLOR_BGR2RGB)

        my_image.imshow(image_visualize)
        my_image.set_title('Image')

        fig.canvas.draw()
        plt.pause(0.001)




# GET THE SHIT GOING!!
if __name__ == '__main__': 
    main()