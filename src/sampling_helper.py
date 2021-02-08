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
    running_curvature_pose = np.vstack((sx2, sy2, running_curvature))
    spawn_poses = np.argwhere(spawn_area[:,:] == 255)

    # sample_pose_curvature dictionary
    print("Creating dictionary for sample_pose_curvature..")
    dict_curv_idx = {}
    curvature_array = running_curvature_pose[2, :]
    for idx in range(len(curvature_array)):
        key = curvature_array[idx]
        if key in dict_curv_idx.keys():
            dict_curv_idx[key].append(idx)
        else:
            dict_curv_idx[key] = [idx]
    print("Done")
    # sample_hard_spawn dictionary
    print("Creating dictionary for sample_hard_spawn..")
    dict_diff_idx = {}
    print("Done")


    return (nav_line_pose, running_curvature_pose, spawn_poses, map_dist2optimal_nav, dict_curv_idx, dict_diff_idx)

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

def sample_hard_spawn_fn(p_hardspawn, hard_spawn_poses, hard_spawn_cost, sampled_pose=None, dict_diff_idx=None,
                         pdf="gaussian", difficulty=0, constant=0.1, do_weight=False, distance_threshold=200):
    ''' THIS FUNCTION SAMPLES A HARD SPAWNING POSITION, IF NO sampled_pose IS PROVIDED,
    THE POSITION IS PICKED AT RANDOM. OTHER_WISE IT WILL BE CLOSE FROM THE REQUESTED
    POSITION
    p_hardspawn: the probabilty of sampling a hard_spawn position given it's distance to the optimal trajectory. A vector of size 101.
    hard_spawn: a list of possible hard_spawn location of size 2: x,y
    hard_spawn_cost: a matrix of cost where the value in each pixel is the distance to the optimal navigation trajectory
    sanpled_pose: a list of 2 values: x and y
    '''
    if sampled_pose is not None:
        possibilities = hard_spawn_poses[np.argwhere(np.sqrt((hard_spawn_poses[:,0] - sampled_pose[0])**2+ \
                                                             (hard_spawn_poses[:,1] - sampled_pose[1])**2) < distance_threshold)][:,0,:]

        hard_spawn_array = hard_spawn_cost[possibilities[:,0], possibilities[:,1]] 

    else:
        hard_spawn_array = hard_spawn_cost[hard_spawn_poses[:,0],hard_spawn_poses[:,1]]

    length_array = len(hard_spawn_array)

    for idx in range(len(hard_spawn_array)):
        key = hard_spawn_array[idx]
        if key in dict_diff_idx.keys():
            dict_diff_idx[key].append(idx)
        else:
            dict_diff_idx[key] = [idx]


    dmax = np.max(hard_spawn_array)
    dmin = np.min(hard_spawn_array)

    mu = dmin + difficulty*(dmax-dmin)
    sigma = 0.1*(dmax-dmin) # 10% of the curvature range

    # Determine the importance of each curvature value
    if pdf == "gaussian": 
        prob_difficulty = gaussian(hard_spawn_array, mu, sigma)

    elif pdf == "constant-gaussian":
        prob_difficulty = [constant_gaussian(x, mu, sigma, constant=constant) for x in hard_spawn_array]

    elif pdf == "2sigma-gaussian":
        sigma_left = 4*sigma
        sigma_right = sigma
        prob_difficulty = [twosigma_gaussian(x, mu, sig_l=sigma_left, sig_r=sigma_right) for x in hard_spawn_array]

    else:
        raise Exception('Please provide a supported probability density function.')


    # Weight the probabilities
    if do_weight:
        weight_matrix = create_weight_matrix(hard_spawn_array)
        prob_difficulty = np.multiply(prob_difficulty, weight_matrix)

    # Sum of the array equals to 1
    prob_sum = np.sum(prob_difficulty)
    prob_difficulty = prob_difficulty/prob_sum

    target_diff = np.random.choice(hard_spawn_array, p=prob_difficulty)
    target_idx = np.random.choice(dict_diff_idx[target_diff])

    print("difficulty", target_idx, target_diff)

    if sampled_pose is not None:
        pose = possibilities[target_idx]

    else:
        pose = hard_spawn_poses[target_idx]

    print("pose", pose)

    return pose



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
        print("possibilities", possibilities)
        dist_norm = np.sqrt((possibilities[:,0] - sampled_pose[0])**2+(possibilities[:,1] - sampled_pose[1])**2)/200
        print("dist_norm", dist_norm)
        difficulty_norm = np.abs(picked_difficulty - hard_spawn_cost[possibilities[:,0], possibilities[:,1]])/(dmax-dmin)
        print("difficulty_norm", difficulty_norm)
        idx = np.argmin( difficulty_norm + dist_norm)
        pose = possibilities[idx]
    return pose


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def constant_gaussian(x, mu, sig, constant):
    if x < mu-sig:
        return constant
    else:
        return gaussian(x, mu, sig)

def twosigma_gaussian(x, mu, sig_l, sig_r):
    if x < mu:
        return gaussian(x, mu, sig_l)
    else:
        return gaussian(x, mu, sig_r)



def create_weight_matrix(current_array):
    '''
    THIS FUNCTION RETURNS THE WEIGHT MATRIX OF THE CURVATURE MATRIX, IN ORDER TO HAVE A UNIFORM DISTRIBUTION OF THESE SAMPLES
    curvature: a numpy of m elements of size 3: x,y,curvature
    '''
    length_array = len(current_array)

    # Weight the samples
    hist_array = np.histogram(current_array, bins=101)
    sum_hist = np.array(hist_array[0])
    values_hist = np.array(hist_array[1])

    weight_matrix = np.zeros(length_array)
    for idx in range(length_array):
        current_curv = current_array[idx]
        idx_hist = np.where((values_hist-current_curv >= 0) == True)[0][0] - 1
        weight_matrix[idx] = 1/float(sum_hist[idx_hist])

    return weight_matrix


def sample_pose_curvature(nav_line, curvature, difficulty=0, dict_curv_idx=None, pdf="gaussian", constant=0.1, do_weight=False):
    ''' 
    THIS FUNCTION SAMPLES THE BOAT POSITION BASED ON THE TRACK CURVATURE
    nav_line: a numpy array of n elements of size 3: x,y,rz
    curvature: a numpy of m elements of size 3: x,y,curvature
    difficulty: between 0 and 1, it moves the mean of the Gaussian distribution
    pdf: "gaussian", "constant-gaussian", "2sigma-gaussian"
    constant: value of the constant for the constant-gaussian pdf
    '''
    curvature_array = curvature[2, :]
    length_array = len(curvature_array)
    

    cmax = np.max(curvature_array)
    cmin = np.min(curvature_array)

    mu = cmin + difficulty*(cmax-cmin)
    sigma = 0.1*(cmax-cmin) # 10% of the curvature range

    # Determine the importance of each curvature value
    if pdf == "gaussian": 
        prob_curvature = gaussian(curvature_array, mu, sigma)

    elif pdf == "constant-gaussian":
        prob_curvature = [constant_gaussian(x, mu, sigma, constant=constant) for x in curvature_array]

    elif pdf == "2sigma-gaussian":
        sigma_left = 4*sigma
        sigma_right = sigma
        prob_curvature = [twosigma_gaussian(x, mu, sig_l=sigma_left, sig_r=sigma_right) for x in curvature_array]

    else:
        raise Exception('Please provide a supported probability density function.')
    
    # Weight the probabilities
    if do_weight:
        weight_matrix = create_weight_matrix(curvature_array)
        # print("Weight the probabilities")
        prob_curvature = np.multiply(prob_curvature, weight_matrix)

    # Sum of the array equals to 1
    prob_sum = np.sum(prob_curvature)
    prob_curvature = prob_curvature/prob_sum

    target_curv = np.random.choice(curvature_array, p=prob_curvature)
    target_idx = np.random.choice(dict_curv_idx[target_curv])

    print("curvature:", target_idx, target_curv)
    pose = nav_line[0:2, target_idx]
    print("sample_pose", pose)

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


def ema_function(previous_value, new_value, alpha):
    result = new_value*alpha + previous_value*(1-alpha)

    return result


def compute_difficulty_fixed_order(current_step, max_step):
    '''
    THIS FUNCTION RETURNS THE CURRENT DIFFICULTY BASED ON THE CURRENT STEP AND THE (ESTIMATED) MAXIMUM NUMBER OF STEPS.
    current_step: number of steps since the beginning
    max_step: estimated maximum number of steps
    '''
    difficulty = np.clip(current_step/max_step, 0, 1)

    return difficulty


def compute_difficulty_adaptive_reward(estimated_reward, min_reward, max_reward):
    '''
    THIS FUNCTION RETURNS THE CURRENT DIFFICULTY BASED ON THE CURRENT ESTIMATED REWARD AND THE RANGE OF REWARD.
    estimated_reward: estimated reward using EMA
    min_reward: minimum reward for an episode
    max_reward: maximum reward for an episode
    '''
    difficulty = np.clip((estimated_reward - min_reward)/(max_reward - min_reward), 0, 1)

    return difficulty


def compute_adaptive_difficulty(estimated_reward, previous_reward, previous_reward_ema, threshold_reward, 
                                list_states, sigma_tight, sigma_large):
    '''
    THIS FUNCTION RETURNS THE CURRENT DIFFICULTY BASED ON THE CURRENT ESTIMATED REWARD, IN AN ADAPTIVE WAY.
    estimated_reward: estimated reward using EMA
    previous_reward: previous estimated reward
    threshold_reward: threshold for a significant change
    list_states: (difficulty mean, difficulty influence)
                  difficulty mean: difficulty given to an environment parameter
                  difficulty influence: positive, neutral or negative influence on the global progression
    sigma_tight: sigma for twosigma_gaussian
    sigma_large: sigma for twosigma_gaussian (for positive or negative influence direction)
    '''

    # Update influence based on the global progression
    list_diff_mean = [x[0] for x in list_states]

    # Global progression slowing down -> decrease highest difficulty influence
    if estimated_reward < previous_reward - threshold_reward or estimated_reward < previous_reward_ema + threshold_reward:
        idx_max_diff = np.argmax(list_diff_mean)
        list_states[idx_max_diff][1] = np.clip(list_states[idx_max_diff][1] - 1, -1, 1).astype(int)

        # Update previous_reward
        previous_reward = estimated_reward

    # Global progression increasing -> increase lowest difficulty influence
    elif estimated_reward >= previous_reward + threshold_reward:
        idx_min_diff = np.argmin(list_diff_mean)
        list_states[idx_min_diff][1] = np.clip(list_states[idx_min_diff][1] + 1, -1, 1).astype(int)

        # Update previous_reward
        previous_reward = estimated_reward


    # Update new difficulties
    points_array = np.linspace(0, 1, 101)
    for idx in range(len(list_states)):
        diff_mean = list_states[idx][0]
        diff_influence = list_states[idx][1]

        # Difficulty increasing progression
        if diff_influence == 1:
            sigma_left = sigma_tight
            sigma_right = sigma_large

        # Difficulty not influencing progression
        elif diff_influence == 0:
            sigma_left = sigma_tight
            sigma_right = sigma_tight

        # Difficulty slowing down progression
        elif diff_influence == -1:
            sigma_left = sigma_large
            sigma_right = sigma_tight

        # Compute new probability sampling for difficulty
        prob_sampling = [twosigma_gaussian(x, diff_mean, sig_l=sigma_left, sig_r=sigma_right) for x in points_array]
        prob_sum = np.sum(prob_sampling)
        prob_sampling = prob_sampling/prob_sum

        new_diff_mean = np.random.choice(points_array, p=prob_sampling)

        # Update new difficulty mean
        list_states[idx][0] = new_diff_mean

    return (list_states, previous_reward)


def sample_boat_position(nav_line, offset, curvature=None, chaos=None, p_curvature=None, p_chaos=None, 
                         p_hardspawn=None, hard_spawn_poses = None, hard_spawn_cost = None, 
                         dict_curv_idx=None, dict_diff_idx=None, 
                         diff_function=None, list_difficulties=None):
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
        print("nav_line shape", np.shape(nav_line))
        print("curvature shape", np.shape(curvature))
        sampled_pose = sample_pose_curvature(nav_line, curvature, difficulty=list_difficulties[0], dict_curv_idx=dict_curv_idx,
                                             pdf="gaussian", constant=0.1, do_weight=True)
    elif (cs_c1 + cs_c2) == 1:
        raise Exception('Please provide all the variables needed to perform curvature based spawning')

    # Hard Spawn Sampling
    hs_c1 = p_hardspawn is not None
    hs_c2 = hard_spawn_poses is not None
    hs_c3 = hard_spawn_cost is not None
    if (hs_c1 + hs_c2 + hs_c3) == 3:
        #sampled_pose = np.array([1766.13898662, 1811.14849046])
        pose = sample_hard_spawn_fn(p_hardspawn, hard_spawn_poses, hard_spawn_cost, dict_diff_idx=dict_diff_idx,
                                    pdf="gaussian", difficulty=list_difficulties[1], sampled_pose=sampled_pose, do_weight=True)
        #pose = sample_hard_spawn(p_hardspawn, hard_spawn_poses, hard_spawn_cost, sampled_pose=sampled_pose)
    elif (hs_c1 + hs_c2 + hs_c3) >= 1:
        raise Exception('Please provide all the variables needed to perform random hard spawning')
    
    full_pose = get_heading_from_pose(nav_line, pose)
    full_pose_true_coords = compensate_offset(offset, full_pose)
    return full_pose_true_coords
