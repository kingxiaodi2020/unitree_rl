
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym() # 获取gym对象

        self.sim_params = sim_params # 保存仿真参数
        self.physics_engine = physics_engine # 设置物理引擎
        self.sim_device = sim_device # 使用gpu还是cpu
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device) # 解析设备字符串，cuda:0,cpu
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:  # 设备选择
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id # 渲染设备选择
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs # 并行运行的多少个仿真环境
        self.num_obs = cfg.env.num_observations # 每个环境中的观察空间的维度
        self.num_privileged_obs = cfg.env.num_privileged_obs # 特权观察空间的维度，用于提供额外的信息给代理，如内部状态、传感器数据等
        self.num_actions = cfg.env.num_actions # 每个环境中的动作空间的维度，智能体执行的动作数量

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False) #  禁用了 PyTorch 的 JIT（即时编译）优化的分析模式
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers # 构建张量初始化了若干个缓冲区 ，用于存储每个环境的状态、奖励、重置标志、时间步等数据
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim() # 创建仿真环境 # 子类实现
        self.gym.prepare_sim(self.sim) # 准备仿真，完成所有的初始化步骤

        # todo: read from config
        self.enable_viewer_sync = True # 启用图形渲染与仿真同步
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False: # 创建一个视图器来显示仿真画面
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties()) # 创建一个新的视图器
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT") # 退出仿真
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync") # 切换视图器同步

    def get_observations(self): # 返回当前所有环境的观测数据
        return self.obs_buf
    
    def get_privileged_observations(self): # 返回当前所有环境的特权观测数据
        return self.privileged_obs_buf

    def reset_idx(self, env_ids): # 抽象方法 重置某些特定的环境或机器人
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots""" # 重置所有的机器人
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions): # 执行一个仿真步骤，子类实现
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer: # 是否创建了图形界面
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer): # 是否关闭了图形界面窗口，关闭就退出程序
                sys.exit()

            # check for keyboard events # 查询图形界面中的所有用户输入事件
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0: # 按下退出键
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:  # 按下同步切换键,切换同步模式
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu': # 如果仿真设备不是 CPU，获取仿真结果。这通常是确保仿真计算与图形渲染一致
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim) # 执行图形的更新步骤
                self.gym.draw_viewer(self.viewer, self.sim, True) # 绘制最新的仿真场景
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim) # 同步图形的帧时间与仿真时间
            else:
                self.gym.poll_viewer_events(self.viewer) # 轮询图形界面的事件（例如用户输入），但不更新图形。