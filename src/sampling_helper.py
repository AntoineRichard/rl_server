import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

def gen_gaussian_dist(x, mu=0, si=0.2):            
    return (1/(si*np.sqrt(2*np.pi)))*np.exp(-(((x-mu)**2)/(2*si**2)))

def load_and_process(path):
    # LOAD MAP
    map_ = (np.load(path)*255).astype(np.uint8)
    map_visu = np.zeros((map_.shape[0],map_.shape[1],3),dtype=np.uint8)
    
    # MOVE TO CORRECT PROJECTION
    map_visu = cv2.rotate(map_visu, cv2.ROTATE_90_COUNTERCLOCKWISE)
    map_ = cv2.rotate(map_, cv2.ROTATE_90_COUNTERCLOCKWISE)
    map_visu = cv2.flip(map_visu, 0)
    map_ = cv2.flip(map_, 0)
    
    # CREATE KERNEL, AND INFLATE IMAGE (REMOVE HOLES)
    kernel = np.ones((3,3),np.uint8)
    map_dil_k3_t5 = cv2.dilate(map_, kernel, iterations=5)
    inv_dil_map = ((map_dil_k3_t5 == 0)*255).astype(np.uint8)
    
    # REMOVE THE CENTER ISLAND
    blobs = cv2.connectedComponents(inv_dil_map)
    inside = ((blobs[1]==2)*255).astype(np.uint8)
    inside_inv = ((inside == 0)*255).astype(np.uint8)
    
    # GET THE INSIDE OF THE LAKE
    blobs_inside = cv2.connectedComponents(inside_inv)
    clean_inside = (((blobs_inside[1]==1) == 0)*255).astype(np.uint8)
    
    # COMPUTE CONTOUR (ON INFLATED MAP)
    contours, hierarchy = cv2.findContours(clean_inside, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    map_visu = cv2.drawContours(map_visu, contours, 0, (0,255,0), 8)
    map_visu[:,:,0] = map_
    
    # COMPUTE CONTOUR WITH COMPENSATION FOR INFLATION OFFSET
    fine_contour_map = (map_.copy()*0).astype(np.uint8)
    fine_contour_map = cv2.drawContours(fine_contour_map, contours, 0, (255), 8)
    fine_contour, fine_h = cv2.findContours(fine_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    map_visu = cv2.drawContours(map_visu, fine_contour,0, (0,0,255),1)
    
    # COMPUTE HARD-SPAWN AREA
    hardspawn_contour_map = (map_.copy()*0).astype(np.uint8)
    hardspawn_contour_map = cv2.drawContours(hardspawn_contour_map, fine_contour, 0, (255), 110)
    hardspawn_contour, fine_h = cv2.findContours(hardspawn_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    map_visu = cv2.drawContours(map_visu, hardspawn_contour,1, (235,52,210),40)
    
    # COMPUTE PERFECT NAVIGATION DISTANCE
    distance_contour_map = (map_.copy()*0).astype(np.uint8)
    ditance_contour_map = cv2.drawContours(distance_contour_map, fine_contour, 0, (255), 180)
    distance_contour, fine_h = cv2.findContours(distance_contour_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    map_visu = cv2.drawContours(map_visu, distance_contour,1, (255,204,0),30)
    
    # MAKE SPAWN MAP
    spawn_area = (map_.copy()*0).astype(np.uint8)
    spawn_area = cv2.drawContours(spawn_area, hardspawn_contour ,1, (255), 40)
    spawn_area = cv2.drawContours(spawn_area, distance_contour, 1, (255), 30)
    
    # COMPUTE DISTANCE OF ALL PIXELS TO THE PERFECT NAVIGATION LINE
    map_optimal_nav = np.ones_like(map_,dtype=np.uint8)*255
    map_optimal_nav = cv2.drawContours(map_optimal_nav, distance_contour,1, (0),1)
    map_dist2optimal_nav = cv2.distanceTransform(map_optimal_nav, cv2.DIST_L2, 5)
    
    # MOVE TO CORRECT PROJECTION
    spawn_area = cv2.rotate(spawn_area, cv2.ROTATE_90_COUNTERCLOCKWISE)
    map_dist2optimal_nav = cv2.rotate(map_dist2optimal_nav, cv2.ROTATE_90_COUNTERCLOCKWISE)
    spawn_area = cv2.flip(spawn_area, 0)
    map_dist2optimal_nav = cv2.flip(map_dist2optimal_nav, 0)
    
    # TAKES NAVIGATION LINE AND FITS SMOOTH SPLNE
    x = distance_contour[1][:,0, 0]
    y = distance_contour[1][:,0, 1]
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.arange(0, 1.001, 0.001) 
    out = interpolate.splev(unew, tck)
    sx = out[0]
    sy = out[1]
    error = 1
    t = np.arange(sx.shape[0])
    std = error * np.ones_like(t)
    t2 = np.arange(sx.shape[0]*4)/4
    fx = UnivariateSpline(t, sx, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, sy, k=4, w=1 / np.sqrt(std))
    
    # COMPUTE CURVATURE FROM SPLINE
    sx2 = fx(t2)
    sy2 = fy(t2)
    x1 = fx.derivative(1)(t2)
    x2 = fx.derivative(2)(t2)
    y1 = fy.derivative(1)(t2)
    y2 = fy.derivative(2)(t2)
    curvature = (x1* y2 - y1* x2) / np.power(x1** 2 + y1** 2, 1.5)
    #print(sx2.shape[0]) 
    #print(sy2.shape) 
    # COMPUTE RUNNING CURVATURE
    max_speed = 1.5 #ms
    ep_length = 60 #seconds
    lake_length = 1400. #meters
    window_size = int(0.25*max_speed*ep_length / (lake_length/sx2.shape[0]))
    running_curvature = np.zeros_like(curvature)
    for i in range(sx2.shape[0]):
        if i < sx2.shape[0] - window_size:
            running_curvature[i] = np.mean(np.abs(curvature[i:i+window_size]))
        else:
            running_curvature[i] = np.mean(np.abs((list(curvature[i:curvature.shape[0]])+list(curvature[0:i - curvature.shape[0] + window_size]))))
    
    # DISCTRETIZE THE SHORE LINE
    x_shore = fine_contour[0][:,0,0]
    y_shore = fine_contour[0][:,0,1]
    tck, u = interpolate.splprep([x_shore, y_shore], s=0)
    unew = np.arange(0, 1.0001, 0.0001) 
    out_shore = interpolate.splev(unew, tck)
    sx_shore = out_shore[0]
    sy_shore = out_shore[1]
    
    # APPLY POLYNOMIAL FILTER
    fsx_shore = savgol_filter(sx_shore,401,2)
    fsy_shore = savgol_filter(sy_shore,401,2)
    
    # COMPUTE DISTANCE (TO CHANGE BASED ON CEDRIC'S FEEDBACK)
    diff_shore = (sx_shore - fsx_shore)**2 + (sy_shore-fsy_shore)**2
    
    # WINDOWED STANDARD DEVIATION
    diff_window = np.zeros_like(diff_shore)
    for i in range(sx_shore.shape[0]):
        if (i > 50) and (sx_shore.shape[0] > i+50):
            diff_window[i] = np.std(diff_shore[i-50:i+50])
        elif i < 50: 
            diff_window[i] = np.std(list(diff_shore[0:i+50])+list(diff_shore[-(sx_shore.shape[0]-i+50):]))
        else:
            diff_window[i] = np.std(list(diff_shore[-(i-50):])+list(diff_shore[:sx_shore.shape[0]-i+50]))

    rz = np.arctan2(y1,x1)
    nav_line_pose = np.vstack((sx2, sy2, rz))
    spawn_poses = np.argwhere(spawn_area[:,:] == 255)
    return nav_line_pose, running_curvature, spawn_poses, map_dist2optimal_nav

def euler2quat(yaw):
    ''' CONVERTS THE RZ (or YAW) INTO A WELL FORMED QUATERNION
    rz: the yaw in radiant
    '''
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    qx = 0
    qy = 0
    qz = sy
    qw = cy
    return np.array([qx,qy,qz,qw])

def sample_hard_spawn(p_hardspawn, hard_spawn_poses, hard_spawn_cost, sampled_pose=None):
    ''' THIS FUNCTION SAMPLES A HARD SPAWNING POSITION, IF NO sampled_pose IS PROVIDED,
    THE POSITION IS PICKED AT RANDOM. OTHER_WISE IT WILL BE CLOSE FROM THE REQUESTED
    POSITION
    p_hardspawn: the probabilty of sampling a hard_spawn position given it's distance to the optimal trajectory. A vector of size 101.
    hard_spawn: a list of possible hard_spawn location of size 2: x,y
    hard_spawn_cost: a matrix of cost where the value in each pixel is the distance to the optimal navigation trajectory
    sanpled_pose: a list of 2 values: x and y
    '''
    dmax = np.max(hard_spawn_cost[hard_spawn_poses[:,0],hard_spawn_poses[:,1]])
    dmin = np.min(hard_spawn_cost[hard_spawn_poses[:,0],hard_spawn_poses[:,1]])
    discretization = np.linspace(dmin,dmax,101)
    picked_difficulty = np.random.choice(discretization, replace=True, p=p_hardspawn)
    if sampled_pose is None:
        eps = 10*(dmax - dmin)/101.
        difficulty = hard_spawn_cost[hard_spawn_poses[:,0], hard_spawn_poses[:,1]]
        possibilities = hard_spawn_poses[np.argwhere(np.abs(difficulty - picked_difficulty) < eps)[:,0]]
        idx = np.random.rand(possibilities.shape[0])
        pose = possibilities[idx]
    else:
        possibilities = hard_spawn_poses[np.argwhere(np.sqrt((hard_spawn_poses[:,0] - sampled_pose[0])**2+(hard_spawn_poses[:,1] - sampled_pose[1])**2) < 200)][:,0,:]
        dist_norm = np.sqrt((possibilities[:,0] - sampled_pose[0])**2+(possibilities[:,1] - sampled_pose[1])**2)/200
        difficulty_norm = np.abs(picked_difficulty - hard_spawn_cost[possibilities[:,0], possibilities[:,1]])/(dmax-dmin)
        idx = np.argmin( difficulty_norm + dist_norm)
        pose = possibilities[idx]
    return pose

def sample_hard_spawn_fn(fn, hard_spawn_poses, hard_spawn_cost, sampled_pose=None):
    ''' THIS FUNCTION SAMPLES A HARD SPAWNING POSITION, IF NO sampled_pose IS PROVIDED,
    THE POSITION IS PICKED AT RANDOM. OTHER_WISE IT WILL BE CLOSE FROM THE REQUESTED
    POSITION
    p_hardspawn: the probabilty of sampling a hard_spawn position given it's distance to the optimal trajectory. A vector of size 101.
    hard_spawn: a list of possible hard_spawn location of size 2: x,y
    hard_spawn_cost: a matrix of cost where the value in each pixel is the distance to the optimal navigation trajectory
    sanpled_pose: a list of 2 values: x and y
    '''
    dmax = np.max(hard_spawn_cost[hard_spawn_poses[:,0],hard_spawn_poses[:,1]])
    dmin = np.min(hard_spawn_cost[hard_spawn_poses[:,0],hard_spawn_poses[:,1]])
    normed_cost = (hard_spawn_cost - dmin)/(dmax - dmin)
    picked_difficulty = np.random.choice(discretization, replace=True, p=p_hardspawn)
    if sampled_pose is None:
        eps = 10*(dmax - dmin)/101.
        difficulty = hard_spawn_cost[hard_spawn_poses[:,0], hard_spawn_poses[:,1]]
        possibilities = hard_spawn_poses[np.argwhere(np.abs(difficulty - picked_difficulty) < eps)[:,0]]
        idx = np.random.rand(possibilities.shape[0])
        pose = possibilities[idx]
    else:
        possibilities = hard_spawn_poses[np.argwhere(np.sqrt((hard_spawn_poses[:,0] - sampled_pose[0])**2+(hard_spawn_poses[:,1] - sampled_pose[1])**2) < 200)][:,0,:]
        dist_norm = np.sqrt((possibilities[:,0] - sampled_pose[0])**2+(possibilities[:,1] - sampled_pose[1])**2)/200
        difficulty_norm = np.abs(picked_difficulty - hard_spawn_cost[possibilities[:,0], possibilities[:,1]])/(dmax-dmin)
        idx = np.argmin( difficulty_norm + dist_norm)
        pose = possibilities[idx]
    return pose

def sample_pose_curvature(nav_line, curvature, p_curvature):
    ''' THIS FUNCTION SAMPLES THE BOAT POSITION BASED ON THE TRACK CURVATURE
    nav_line: a numpy array of n elements of size 3: x,y,rz
    curvature: a numpy of m elements of size 3: x,y,curvature
    p_curvature: the probabilty of sampling a curvature of a given value
    '''
    dmax = np.max(curvature)
    dmin = np.min(curvature)
    discretization = np.linspace(dmin,dmax,101)
    target_curvature = np.random.choice(discretization, replace=True, p=p_curvature)
    eps = 10*(dmax - dmin)/101.
    idxs = np.argwhere(np.abs(curvature - target_curvature) < eps)
    if idxs.shape[0] < 30:
        idxs = np.argpartition(np.abs(curvature - target_curvature),15)
        idxs = idxs[:15]
    else:
        idxs = np.squeeze(idxs)
    idx = np.random.choice(idxs)
    pose = nav_line[0:2,idx]
    return pose

def sample_pose_curvature_fn(nav_line, curvature, fn):
    ''' THIS FUNCTION SAMPLES THE BOAT POSITION BASED ON THE TRACK CURVATURE
    nav_line: a numpy array of n elements of size 3: x,y,rz
    curvature: a numpy of m elements of size 3: x,y,curvature
    p_curvature: the probabilty of sampling a curvature of a given value
    '''
    dmax = np.max(curvature)
    dmin = np.min(curvature)
    discretization = np.linspace(dmin,dmax,101)
    target_curvature = np.random.choice(discretization, replace=True, p=p_curvature)
    eps = 10*(dmax - dmin)/101.
    idxs = np.argwhere(np.abs(curvature - target_curvature) < eps)
    if idxs.shape[0] < 30:
        idxs = np.argpartition(np.abs(curvature - target_curvature),15)
        idxs = idxs[:15]
    else:
        idxs = np.squeeze(idxs)
    idx = np.random.choice(idxs)
    pose = nav_line[0:2,idx]
    return pose

def get_heading_from_pose(nav_line, pose):
    ''' THIS FUNCTION GIVES THE BOAT A HEADING GIVEN A POSITION
    pose: a list of 2 values: x and y.
    '''
    idx = np.argmin((nav_line[0] - pose[0])**2 + (nav_line[1] - pose[1])**2)
    quat_head = euler2quat(nav_line[2,idx])
    full_pose = np.concatenate((pose, np.array([0.125]), quat_head),axis=0)
    return full_pose 

def compensate_offset(offset, pose):
    pose[0] = (pose[0] + offset[0])/10
    pose[1] = (pose[1] + offset[1])/10
    return pose

def sample_boat_position(nav_line, offset, curvature=None, chaos=None, p_curvature=None, p_chaos=None, p_hardspawn=None, hard_spawn_poses = None, hard_spawn_cost = None):
    ''' THIS FUNCTION AIMS AT PICKING A SPAWNING POSITION BASED ON SOME PARAMETERS
    nav_line: a numpy array of n elements of size 3: x,y,rz
    nav_area: a list of possible spawn location
    curvature: a numpy of m elements of size 3: x,y,curvature
    offset: a list of values: the x and y offset
    chaos: a numpy array of k elements of size 3: x,y,chaos
    p_curvature: the probabilty of sampling a curvature of a given value
    p_hardspawn: the probabilty of sampling a hard_spawn position given it's distance to the optimal trajectory. A vector of size 101.
    hard_spawn: a list of possible hard_spawn location of size 2: x,y
    hard_spawn_cost: a matrix of cost where the value in each pixel is the distance to the optimal navigation trajectory
    '''
    sampled_pose = None
    # Curvature Sampling
    cs_c1 = curvature is not None
    cs_c2 = p_curvature is not None
    if (cs_c1 + cs_c2) == 2:
        sampled_pose = sample_pose_curvature(nav_line, curvature, p_curvature)
    elif (cs_c1 + cs_c2) == 1:
        raise Exception('Please provide all the variables needed to perform curvature based spawning')
        
    # Hard Spawn Sampling
    hs_c1 = p_hardspawn is not None
    hs_c2 = hard_spawn_poses is not None
    hs_c3 = hard_spawn_cost is not None
    if (hs_c1 + hs_c2 + hs_c3) == 3:
        pose = sample_hard_spawn(p_hardspawn, hard_spawn_poses, hard_spawn_cost, sampled_pose)
    elif (hs_c1 + hs_c2 + hs_c3) >= 1:
        raise Exception('Please provide all the variables needed to perform random hard spawning')
    
    full_pose = get_heading_from_pose(nav_line, pose)
    full_pose_true_coords = compensate_offset(offset, full_pose)
    return full_pose_true_coords
