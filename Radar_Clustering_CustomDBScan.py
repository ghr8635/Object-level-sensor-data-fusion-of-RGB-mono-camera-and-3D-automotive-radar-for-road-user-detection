#### File management ####
import os 
from pathlib import Path
##### Data Structures #### 
import numpy as np 
import pandas as pd
from collections import defaultdict
##### Visualizations #####
import matplotlib.image as mpi 
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle
import math

############# FUSION PIPELINE CODE #################
def radar_2d_projections(radar_points, camera_T, camera_k):
    """
    Function helps with projection of radar points onto image plane
    
    Args: 
        radar_points: (x,y,z)
        camera_T: calibration info 
        camera_k: calibration info 

    Output: 
        Numpy.ndarray (x,y,1)
    """
    new_radar_points = np.hstack((radar_points, 
                                  np.ones((len(radar_points), 1))))
    project_2d = np.matmul(new_radar_points, np.matmul(camera_k , camera_T).T) 
    # project_2d = np.matmul(new_radar_points, camera_k @ camera_T.T)
    last_column = project_2d[:,-1]
    result_radar_2d = project_2d / last_column[:, None] 
    return result_radar_2d 


def associate_data():
    """ Associates the radar data - wrt - image data
        i.e. clusters are associated with bboxes in the image 
        Returns the objects in the image -- represented how? 
        Some form of using Hungarian Association matrix 
        Expected output:
            : [(bbox, radar), (bbox, radar)] -- inside a scene 
    """
    
    pass 
    

############# RADAR CLUSTERING CODE #################
class my_custom_dbscan:
    def __init__(self, eps1, eps2, min_samples):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.number_of_clusters = 0
        self.clusters_information = []


    def read_radar_pcd(self, pcd_file):
        """
        reads radar pcd_file and returns numpy array of cloud
        Args:
            pcd_file: pcd file path to read
        return:
            cloud: numpy array containing dtypes and field names
        """
        fields = []
        with open(pcd_file, "rb") as pcd:
            for line in pcd:
                ln = line.strip().decode("utf-8")
                if ln.startswith("FIELDS"):
                    fields = ln.split()[1:]
                if ln.startswith("DATA"):
                    break

            binary_data = pcd.read()
            np_dtype = np.dtype({"names" : fields, "formats": [np.dtype("float32") for _ in range(len(fields))]})
            self.cloud_data = np.frombuffer(binary_data, np_dtype)

    """Method to get neighbouring points based on distance calculation for the given radar point.
    This method will extract range values and scale the eps1 and eps 2 using the range value. Starts with one point which is
    the 'point in consideration' and then calculates the distances with other points to determine the other points are the
    neighbours of the point in consideration and returns neighbours list."""
    def region_query(self, pcd_radar_point):
        neighbours = []
        coordinates = self.cloud_data[pcd_radar_point][-3:-1]  # x, y coordinates
        range_pcd_radar_point = self.cloud_data[pcd_radar_point][0]
        
        # change of eps wrt range
        eps1 = ((self.eps1/math.sqrt(range_pcd_radar_point)) * range_pcd_radar_point) + 1.5
        eps2 = ((self.eps2/math.sqrt(range_pcd_radar_point)) * range_pcd_radar_point) + 1.5

        # eps1 = 2.5 * math.exp(range_pcd_radar_point/60)
        # eps2 = 2.5 * math.exp(range_pcd_radar_point/60)

        # eps1 = self.eps1 * range_pcd_radar_point
        # eps2 = self.eps2 * range_pcd_radar_point


        # print(f" eps {eps1, eps2}")
        
        # Calculate the neighbourhood for each other point in cloud_data
        for point_in_consideration in range(0, len(self.cloud_data)):
            if pcd_radar_point == point_in_consideration:
                continue
                
            # calculate the distance between current point and the radar point
            diff_vector = np.array(self.cloud_data[point_in_consideration][-3:-1]) - np.array(coordinates)
            scaled_diff_vector = [diff_vector[0]/eps1, diff_vector[1]/eps2]
            distance = np.sqrt(np.dot(scaled_diff_vector, scaled_diff_vector))
            
            # if the distance is less than 1, then the point is considered a neighbour
            if distance < 1:
                neighbours.append(point_in_consideration)
        return neighbours
    
    """Method to update cluster labels for a radar point and its neighbourhood. First, the initial point (`pcd_radar_point`) 
    is assigned to the current cluster (`self.number_of_clusters`). Then a `while` loop goes through each point in 
    `neighbouring_points. For each of these neighbouring points, we check if it's unassigned (i.e., label is 0) or it has 
    been marked as noise (i.e., label is -1). If that is the case, it will be assigned to the current cluster. Then the 
    neighbours of this new point are found, expanding the search. If one of these new neighbouring points has enough neighbours
    (defined by `self.min_samples`), these points are also added to the original list of neighbouring points. This way, by 
    calling the `cluster_formation` function on an initial point and its neighbours, DBSCAN clusters not only 
    these points, but also their respective neighbours, and their neighbours' neighbours, and so on. """
    def cluster_formation(self, pcd_radar_point, neighbouring_points):
        
        #assign cluster label to the radar point
        self.labels[pcd_radar_point] = self.number_of_clusters
        i = 0
        while i < len(neighbouring_points):    
            neighbouring_point = neighbouring_points[i]
            if self.labels[neighbouring_point] in [-1, 0]:
                
                # assign it to the current cluster
                self.labels[neighbouring_point] = self.number_of_clusters
                
                # get the neighbours of the neighbouring point
                add_neighbouring_points = self.region_query(neighbouring_point)
                
                # if it has enough neighbours
                if len(add_neighbouring_points) >= self.min_samples:
                    
                    # add those neighbours to the neighbourhood list
                    neighbouring_points += add_neighbouring_points
            i += 1

    """ This method first starts by looping over every point (`pcd_radar_point`) in the dataset (`self.cloud_data`). Then it 
    checks if the point has been yet assigned to a cluster. If it has it skips the rest of the loop for this point and moves 
    on to the next. Then `region_query` method is called to find all points in the dataset that are within `eps` distance from 
    the current `pcd_radar_point. Then, it checks if the number of `neighbouring_points` is less than `min_samples`. If this 
    is the case, it means that the point does not have enough density around it to form a cluster and is therefore considered as 
    noise. This is done by assigning it a label of -1. If the point does indeed have at least `min_samples` points , it is a 
    "core point" and a new cluster is started. At the end, the method returns `self.labels`, which is a list of cluster labels 
    for each point in the dataset. Points labeled as -1 are considered noise, and points labeled with a positive integer belong 
    to the cluster identified by that number."""
    
    def my_custom_dbscan(self):
        
        # loop through all data points in cloud_data
        for pcd_radar_point in range(len(self.cloud_data)):
            
            #if the point is not assigned to any cluster yet
            if not (self.labels[pcd_radar_point] == 0):
                continue
                
            # get the neighbours for the current radar point
            neighbouring_points = self.region_query(pcd_radar_point)
            
            # if the point does not have enough neighbours mark it as noise
            if len(neighbouring_points) < self.min_samples:
                self.labels[pcd_radar_point] = -1
            else:
                # increase the number of clusters
                self.number_of_clusters += 1
                # form a new cluster with the point and its neighbours
                self.cluster_formation(pcd_radar_point, neighbouring_points)
        
        # form a new cluster with the point and its neighbours
        return self.labels
    
    """For each cluster (not noise), the centroid and closest point of the cluster is calculated and appended to the `file_info` 
    list. `all_files_info[os.path.basename(pcd_file)] = file_info`: Adds the list `file_info` into the `all_files_info` 
    dictionary with the filename as the key. Finally, the function returns the `all_files_info` dictionary, which contains 
    the centroid and the closest point for each cluster in one array and another array for noise points in each scene's PCD file.
    If no clusters were detected in a file, an empty list `[]` is added for that file."""

    def process_pcd_files(self, pcd_file):
        # all_files_info = defaultdict(dict)
        #for scene in scenes_to_process:
            #print(scene)
        # radar_data_folder = Path(path_to_radar)
        # if os.path.exists(radar_data_folder):
        #     pcd_files = sorted(os.path.join(radar_data_folder, f) for f in os.listdir(radar_data_folder) if f.endswith('.pcd'))
        #     for index, pcd_file in enumerate(pcd_files):
        self.read_radar_pcd(pcd_file)
        pcd_radar_point_cloud_filtered_1 = self.cloud_data[self.cloud_data["range"] < 100]
        cloud_data = pcd_radar_point_cloud_filtered_1[abs(pcd_radar_point_cloud_filtered_1["range_rate"]) > 0.1]
        df = pd.DataFrame(cloud_data)
        pcd_np = df.to_numpy()

        file_info = {"clusters": [], "noise": []}
        if pcd_np.shape[0] > 0:
            self.labels = [0] * len(pcd_np)
            self.cloud_data = pcd_np
            labels_pred = self.my_custom_dbscan()
            df['cluster'] = labels_pred  

            clusters_detected = False
            for cluster in df["cluster"].unique():
                cluster_data = df[df["cluster"] == cluster]
                if cluster == -1:
                    for index, row in cluster_data.iterrows():
                        noise_point_position = row[["x", "y", "z"]].values.tolist()
                        noise_point_velocity = row[['range_rate']].values.tolist()
                        noise_point_data = [noise_point_position, noise_point_velocity]
                        file_info['noise'].append(noise_point_data)

                    continue
                
                clusters_detected = True
                centroid = cluster_data[["x", "y", "z"]].mean().tolist()
                closest_point = cluster_data[cluster_data['range'] == cluster_data['range'].min()]
                
                closest_point_x_position = np.array(closest_point[['x']].values.tolist()[0])
                y_mean = np.array(cluster_data[["y"]].mean().tolist())
                closest_point_z_position = np.array(closest_point[['z']].values.tolist()[0])
                closest_point_position = list(np.column_stack((closest_point_x_position, y_mean, closest_point_z_position))[0])

                # closest_point_position = [closest_point_x_position + y_mean + closest_point_z_position]
                # closest_point_position = closest_point[['x', 'y', 'z']].values.tolist()[0]
                # print(closest_point_position)

                # closest_point_velocity = closest_point['range_rate'].values.tolist()
                centroid_velocity = cluster_data[['range_rate']].mean().tolist()
                centroid_closest_point_centroid_velocity = [centroid, closest_point_position, centroid_velocity]
                file_info['clusters'].append(centroid_closest_point_centroid_velocity)

            if not clusters_detected:
                file_info["clusters"].append([]) # No clusters detected. Append an empty list.

            # all_files_info[os.path.basename(pcd_file)] = file_info

        return file_info

#
############# VISUALIZATION FUNCTIONS ###############

def visualize_radar_bbox(radar_points:list, bbox_points:list, image_path:str):
    """
    Visualize radar and bbox together for a single image
    Args:
        bbox_points: x_top, y_top, x_bottom, y_bottom 
        radar_points: x, y (could be centroid or closest point)  
    """
    img = plt.imread(image_path)
    plt.imshow(img)
    # Form a rectangle patch to be plotted 
    rectangle = Rectangle(xy=(bbox_points[0], bbox_points[1]),
                          width=bbox_points[2] - bbox_points[0],
                          height=bbox_points[3]-bbox_points[1],
                          color='red', fill=False, 
                          linewidth=0.5)
    
    # Rectangle plot for bbox 
    plt.gca().add_patch(rectangle)

    # Scatter plot for radar points 
    plt.scatter(radar_points[0], radar_points[1])
    plt.show()

# if __name__ == "__main__": 
