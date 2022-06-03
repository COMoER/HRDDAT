import sys
from pathlib import Path
base_dir = str(Path(__file__).resolve().parent)
sys.path.append(base_dir)
from utils.pure_env_wrapper import wrap_pytorch_task
import numpy as np
import os
# from scipy.ndimage import binary_dilation
from heapq import heappush, heappop
import cv2

S = 200
R_S = 700
reso = R_S / S
robot_radius = 18.75  # [m] 机器人尺寸

W = int(np.round(robot_radius / reso))

KP = 0.001  # 引力势场增益
ETA = 10000.0  # 斥力势场增益

Q = 21  # 障碍物作用范围 pixel
Dt = 100  # 目标二次区范围 pixel

DILATION = 12
ERROSION = 8

repulsive_fn = lambda dq: 0.5 * ETA * (1.0 / np.clip(dq,robot_radius,Q*reso) - 1.0 / (Q * reso)) ** 2
def attractive_fn(ds):
    d1 = np.clip(ds,-np.inf,Dt*reso)
    d2 = np.max(np.stack([2*ds-Dt*reso,ds],-1),-1)
    return 0.5 * KP * d1 * d2



def get_windows_stride(x, window_shape):
    """
    x:[H,W,C]
    """
    window_shape = (*window_shape, x.shape[2])
    new_shape = (x.shape[0] + window_shape[0] // 2 * 2, x.shape[1] + window_shape[1] // 2 * 2, x.shape[2])
    new_x = np.zeros(new_shape)
    new_x[window_shape[0] // 2:-(window_shape[0] // 2), window_shape[1] // 2:-(window_shape[1] // 2)] = x
    x = new_x

    axis = tuple(range(x.ndim))

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return np.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)

class AStar:
    def __init__(self,
                 obs_map,
                 start,
                 heuristic_cost = 'euclidean',
                 neighbor=8,
                 dilation=5):
        
        self.original_grid_map = np.squeeze(obs_map,-1)
        self.start = (int(start[0]),int(start[1]))
    
        self.obstruction_dilation(dilation)
        self.height,self.width = self.grid_map.shape
        self.reset()
        
        if heuristic_cost == 'euclidean':
            self.heuristic_cost_estimate = self.heuristic_euclidean
        elif heuristic_cost == 'manhattan':
            self.heuristic_cost_estimate = self.heuristic_manhattan
        else:
            raise NotImplementedError
        
        if neighbor == 8:
            self.neighbor_bias = np.array([[-1,-1],[-1,0],[-1,1],
                                           [0,-1],        [0,1],
                                           [1,-1], [1,0], [1,1]])
        elif neighbor == 4:
            self.neighbor_bias = np.array([[-1,0],[0,-1],[0,1],[1,0]])
        else:
            raise NotImplementedError
            
        goals_index = np.where(self.grid_map == 2)
        if not start:
            self.start = np.array([-1,-1])
        self.goal = np.array([np.mean(goals_index[0]),np.mean(goals_index[1])])
        
    def reset(self):
        self.G = np.ones_like(self.grid_map,dtype=float)#*self.infinite
        self.F = np.ones_like(self.grid_map,dtype=float)#*self.infinite
        self.parent = -np.ones((self.height,self.width,2),dtype=int)
        self.open = np.zeros_like(self.grid_map) != 0
        # self.close = np.zeros_like(self.grid_map) != 0
        
    def obstruction_dilation(self,dilation=DILATION,start_square = ERROSION):
        grid_map = self.original_grid_map.copy()
        goal_index = np.where(grid_map == 2)
        grid_map[goal_index] = 0
        
        dilate_kernel = np.ones((DILATION, DILATION), dtype=np.uint8)
        grid_map = grid_map.astype(np.uint8)
        grid_map = cv2.dilate(grid_map,dilate_kernel).astype(int)
        
        grid_map[self.start[0]-start_square:self.start[0]+start_square+1,
                 self.start[1]-start_square:self.start[1]+start_square+1] = 0
        grid_map[goal_index] = 2
        self.grid_map = grid_map
    
    def is_goal_reached(self,current):
        if type(current) is tuple:
            return self.grid_map[current] == 2
        if type(current) is np.ndarray:
            return self.grid_map[(int(current[0]),int(current[1]))] == 2
    
    def heuristic_euclidean(self,p1,p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def heuristic_manhattan(self,p1,p2):
        return np.abs(p1[0]-p2[0]) + np.abs(p1[1]-p2[1])
    
    def reconstruct_path(self, current):
        path = [[current[0],current[1]]]
        while not (self.parent[(current[0],current[1])] == np.array([-1,-1])).any():
            current = self.parent[(current[0],current[1])]
            path.append([current[0],current[1]])
            if current[0] == self.start[0] and current[1] == self.start[1]:
                break
            # print(self.parent[(current[0],current[1])])
        path.reverse()
        self.trace = np.array(path)
        return np.array(path)
    
    def step_distance(self,bias):
        return np.sqrt(np.sum(np.square(bias),-1))
    
    def valid_index(self,index):
        if index[0] < 0 or index[0] >= self.height:
            return False
        elif index[1] < 0 or index[1] >= self.width:
            return False
        elif self.grid_map[index] == 1:
            return False
        return True
        
    def calc_astar_trace(self,start=None):        
        if self.is_goal_reached(start):
            return [start]
        # searchNodes = AStar.SearchNodeDict()
        # startNode = searchNodes[start] = AStar.SearchNode(
        #     start, gscore=.0, fscore=self.heuristic_cost_estimate(start, self.goal))
        openSet = []
        if start:
            self.start = start
        start_index = (int(self.start[0]),int(self.start[1]))
        heappush(openSet, (self.F[start_index],
                           start_index[0],
                           start_index[1])
                 )
        
        while openSet:
            current_F, current_x,current_y = heappop(openSet)
            
            current_index = (int(current_x),int(current_y))
            if self.is_goal_reached(current_index):
                return self.reconstruct_path(current_index)

        
        ## I don't know why this is not working, so I change it to my own implementation.
        ## HOPE NO BUG IN THE WORLD TOMORROW.
        #     self.notopen[current_index] = True
        #     self.close[current_index] = True
            
            for bias in self.neighbor_bias:
                neighbor = current_index + bias 
                neighbor_index = (int(neighbor[0]),int(neighbor[1]))
                if not self.valid_index(neighbor_index):
                    continue
                
                # Determine whether a diagonal line can pass.
                if sum(bias) == 2 and self.grid_map[(current_index[0],neighbor_index[1])] == 1 and self.grid_map[(neighbor_index[0],current_index[1])] == 1:
                    continue
                
                # Calculate G, H, F value.
                addG = self.step_distance(bias)
                G = self.G[current_index] + addG
                H = self.heuristic_cost_estimate(neighbor, self.goal)
                F = G + H
            
                if self.open[neighbor_index]:
                    if G < self.G[neighbor_index]:
                        self.G[neighbor_index] = G
                        self.F[neighbor_index] = F
                        self.parent[neighbor_index] = current_index
                else:
                    self.open[neighbor_index] = True
                    self.G[neighbor_index] = G
                    self.F[neighbor_index] = F
                    self.parent[neighbor_index] = current_index
                    heappush(openSet, (F,
                                       neighbor_index[0],
                                       neighbor_index[1])
                             )
                    self.open[neighbor_index] = True
        return []
        #         if self.close[neighbor_index]:
        #             continue
                
        #         tentative_gscore = self.G[current_index] + \
        #             self.step_distance(bias)
        #         if tentative_gscore < self.G[neighbor_index]:
        #             continue
        #         else:
        #             self.parent[neighbor_index] = current
        #             self.G[neighbor_index] = tentative_gscore
        #             self.F[neighbor_index] = tentative_gscore + \
        #                 self.heuristic_cost_estimate(neighbor, self.goal)
                        
        #         if self.notopen[neighbor_index]:
        #             self.notopen[neighbor_index] = False
        #             heappush(openSet, (-self.F[neighbor_index],neighbor))
        #         else:
        #             # re-add the node in order to re-sort the heap
        #             openSet.remove(neighbor)
        #             heappush(openSet, (-self.F[neighbor_index],neighbor))
                    
        # return None
    
    def examine_trace(self,trace,savefig_name = None,origin_map=True):
        # 0:空白->255白
        # 1：障碍->0黑
        if origin_map:
            read_map = self.original_grid_map
        else:
            read_map = self.grid_map
        traced_map = read_map.copy().astype(np.float32)
        traced_map = cv2.cvtColor(traced_map, cv2.COLOR_GRAY2RGB)
        
        traced_map[read_map == 0] = [255,255,255]
        traced_map[read_map == 1] = [0,0,0]
        traced_map[read_map == 2] = [0,0,255]
        
        if trace.size:
            for trace_point in trace:
                traced_map[(trace_point[0],trace_point[1])] = [255,0,255]
        
        if savefig_name is not None:
            cv2.imwrite(savefig_name,traced_map.astype(np.uint8))
            
    def calc_costmap(self,whole_map,trace_dir = None):
        # 对700*700map,距每个点最近的路径点编号为cost
        costmap = np.zeros_like(whole_map,dtype=int)
        
        if trace_dir:
            trace = np.load(trace_dir)
        else:
            trace = self.trace
            
        trace_square_sum = np.sum(trace**2,axis = 1)
        for h in range(R_S):
            for w in range(R_S):
                pos_scaled = np.array([h/reso,w/reso])
                trace_pos_multi_sum = np.sum(trace*pos_scaled,axis = 1)
                pos_square_sum = np.sum(pos_scaled**2)
                cost_dist = trace_square_sum -2*trace_pos_multi_sum + pos_square_sum
                costmap[h][w] = np.argmin(cost_dist)
                
        np.save(f"costmap/astar/S{S}/map{i}_cost{ctrl_agent_index}.npy",costmap)
        
        traced_costmap = costmap.copy().astype(np.float32)*255.0/trace.shape[0]
        traced_costmap = cv2.cvtColor(traced_costmap, cv2.COLOR_GRAY2RGB)
        traced_costmap[whole_map==1] = [0,0,0]
        traced_costmap[whole_map==2] = [0,0,255]
        cv2.imwrite(f"costmap/astar/S{S}/map{i}_cost{ctrl_agent_index}.png",traced_costmap)
        
        
# def calc_potential_field(obs_map):
#     windows_dis = np.empty((Q,Q))
#     for i in range(Q):
#         for j in range(Q):
#             i_ = i - Q//2
#             j_ = j - Q//2
#             windows_dis[i, j] = np.sqrt((i_ * reso) ** 2 + (j_ * reso) ** 2)
#     windows_dis = windows_dis.reshape((1, 1, Q,Q)).repeat(S, axis=0).repeat(S, axis=1)
#     windows = get_windows_stride(obs_map, (Q,Q)).reshape((S, S, Q,Q))
#     e = repulsive_fn(windows_dis)
#     e = np.sum((e*(windows==1)).reshape(S,S,-1),-1)

#     grid = np.stack(np.meshgrid(np.arange(S),np.arange(S)),-1)
#     target_pos = grid[obs_map.reshape(S,S)==2,:].mean(0)

#     dis = ((grid - target_pos.reshape(1,1,2))*reso)
#     ds = np.sqrt(np.sum(dis**2,-1))

#     attr = attractive_fn(ds)
#     pmap = e+attr
#     return pmap

# def examine_potential_field(costmap,mapid):
#     local_minimum = ((costmap <= np.roll(costmap,  1, 0)) &
#             (costmap <= np.roll(costmap, -1, 0)) &
#             (costmap <= np.roll(costmap,  1, 1)) &
#             (costmap <= np.roll(costmap, -1, 1)))
#     annotated_costmap = costmap.astype(np.float32)
#     annotated_costmap = cv2.cvtColor(annotated_costmap, cv2.COLOR_GRAY2RGB)
#     annotated_costmap[local_minimum] = [0,0,255]
    
#     window_name = f"map{mapid}"
    
#     # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow(window_name, 700, 700)
#     # cv2.imshow(window_name,annotated_costmap.astype(np.uint8))
#     # cv2.waitKey(0)
#     cv2.imwrite(f"./costmap/S{S}/costmap{i}.png",annotated_costmap.astype(np.uint8))



if __name__ == '__main__':
    dst_dir = os.path.join(base_dir,"costmap","astar",f"S{S}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    save_maps = True
    examine_maps = True
    
    ctrl_agent_index = 0
    
    for i in range(1, 12):
        print(f"Compute map {i} by astar")
        game = wrap_pytorch_task(f"olympics-running:{i}")
        m = game.env_core.get_map_grid([S, S])

        pos = game.env_core.get_agent_position()

        pos_index = [pos[ctrl_agent_index][1]//reso,pos[ctrl_agent_index][0]//reso]
        
        # costmap = calc_potential_field(m)
        astar_algo = AStar(m,start = pos_index)
        trace = astar_algo.calc_astar_trace()

        
        if save_maps:
            np.save(f"costmap/astar/S{S}/map{i}_trace{ctrl_agent_index}.npy",trace)
        if examine_maps:
            # examine_potential_field(costmap,i)
            astar_algo.examine_trace(trace,
                                     savefig_name = f"costmap/astar/S{S}/map{i}_trace{ctrl_agent_index}.png",
                                     origin_map=(ctrl_agent_index == 0))
            full_m = (game.env_core.get_map_grid([R_S,R_S])).squeeze(-1)
            astar_algo.calc_costmap(full_m)

