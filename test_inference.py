from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient
from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta

# 1. 配置并初始化客户端 (注意变量名的修正)
cfg = AgentCfg(
    server_host='localhost',
    server_port=8087,
    model_name='internvla_n1',
    ckpt_path='',
    model_settings={
        'policy_name': "InternVLAN1_Policy",
        'state_encoder': None,
        'env_num': 1,
        'sim_num': 1,
        'model_path': "checkpoints/InternVLA-N1", 
        'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
        'width': 640,
        'height': 480,
        'hfov': 79,
        'resize_w': 384,
        'resize_h': 384,
        'max_new_tokens': 1024,
        'num_frames': 32,
        'num_history': 8,
        'num_future_steps': 4,
        'device': 'cuda:0',
        'predict_step_nums': 32,
        'continuous_traj': True,
        'vis_debug': False
    }
)

# 实例化客户端与服务端建立连接
agent = AgentClient(cfg)

# 2. 加载官方预置的测试观测数据（一张从 RealSense 深度相机录制的帧）
rs_meta_path = './scripts/iros_challenge/onsite_competition/captures/rs_meta.json'
fake_obs = load_obs_from_meta(rs_meta_path)

# 强行注入一条自然语言导航指令
fake_obs['instruction'] = 'go to the red car'
print(f"输入图像尺寸: RGB {fake_obs['rgb'].shape}, Depth {fake_obs['depth'].shape}")

# 3. 发起推理请求：将观测数据发给服务端，获取动作
result = agent.step([fake_obs])
action = result[0]['action'][0]

print(f"服务端返回的动作指令 (Action taken): {action}")
