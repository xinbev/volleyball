import os
import sys
import os.path as osp
import torch
import wandb
import numpy as np

from isaacgym import gymapi, gymutil
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf
import hydra

from phc.utils.config import set_np_formatting, set_seed, SIM_TIMESTEP
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver

from learning import (
    im_amp, im_amp_players, amp_agent, amp_self_play_agent,
    amp_players, amp_self_play_players, amp_models,
    amp_network_builder, amp_network_mcp_builder,
    amp_network_pnn_builder, amp_network_z_builder
)

from env.tasks import humanoid_amp_task

# Optional auto-resume support
try:
    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
except ModuleNotFoundError:
    AutoResume = None


# ------------------------------
# Utility: Simulation Parameters
# ------------------------------
def parse_sim_params(cfg):
    """
        根据 Hydra 配置构造 Isaac Gym 所需的 SimParams。
        支持 PHYSX 与 Flex 引擎。
        Returns:
            sim_params (gymapi.SimParams): 仿真参数配置
        """
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = cfg.sim.slices

    if cfg.sim.use_flex:
        sim_params.use_flex.shape_collision_margin = 0.01
        sim_params.use_flex.num_outer_iterations = 4
        sim_params.use_flex.num_inner_iterations = 10
    else:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = max(cfg.sim.physx.num_threads, 1)
        sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
        sim_params.physx.num_subscenes = cfg.sim.subscenes
        sim_params.physx.max_gpu_contact_pairs = (
            4 * 1024 * 1024 if flags.test and not flags.im_eval else 16 * 1024 * 1024
        )

    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]

    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    return sim_params


# ------------------------------
# Utility: Env Creator
# ------------------------------
def create_rlgpu_env(**kwargs):
    global cfg_train, cfg

    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd
        rank = hvd.rank()
        cfg_train['params']['seed'] += rank
        cfg['rank'] = rank
        cfg['rl_device'] = f'cuda:{rank}'
        kwargs.update({"device_id": rank, "rl_device": f"cuda:{rank}"})

    sim_params = parse_sim_params(cfg)
    args = EasyDict({
        "task": cfg.env.task,
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
        "headless": cfg.headless,
        "device": cfg.device,
    })

    _, env = parse_task(args, cfg, cfg_train, sim_params)
    return env


# ------------------------------
# RL-Games Observer
# ------------------------------
class RLGPUAlgoObserver(AlgoObserver):
    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, algo.games_to_track).to(algo.ppo_device)
        self.writer = algo.writer

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            key = 'successes' if flags.has_eval else 'consecutive_successes'
            if key in infos:
                data = infos[key].clone()
                if key == 'successes':
                    data = data[done_indices]
                self.consecutive_successes.update(data.to(self.algo.ppo_device))

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_val = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_val, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_val, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_val, total_time)


# ------------------------------
# RL-Games Env Wrapper
# ------------------------------
class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = self.env.num_states > 0
        self.full_state = {"obs": self.reset()}
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.full_state["obs"] = obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, rew, done, info
        return obs, rew, done, info

    def reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        self.full_state["obs"] = obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        return obs

    def get_env_info(self):
        info = {
            'action_space': self.env.action_space,
            'observation_space': self.env.observation_space,
            'amp_observation_space': self.env.amp_observation_space,
            'enc_amp_observation_space': self.env.enc_amp_observation_space,
            'task_obs_size': self.env.task.get_task_obs_size() if isinstance(self.env.task,
                                                                             humanoid_amp_task.HumanoidAMPTask) else 0
        }
        if self.use_global_obs:
            info['state_space'] = self.env.state_space
        return info


# Register environment
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {'env_creator': create_rlgpu_env, 'vecenv_type': 'RLGPU'})


# ------------------------------
# Build RL-Games Runner
# ------------------------------
def build_alg_runner(observer):
    """
    构建并注册 AMP / AMP Self-Play / IMAMP 等玩家、网络、算法工厂
    Args:
        algo_observer: 训练过程中自定义观察器
    Returns:
        Runner 实例
    """
    runner = Runner(observer)
    runner.player_factory.register_builder('amp_discrete', lambda **kw: amp_players.AMPPlayerDiscrete(**kw))
    runner.player_factory.register_builder('amp', lambda **kw: amp_players.AMPPlayerContinuous(**kw))
    runner.player_factory.register_builder('amp_self_play',
                                           lambda **kw: amp_self_play_players.AMPSelfPlayPlayerContinuous(**kw))
    runner.player_factory.register_builder('im_amp', lambda **kw: im_amp_players.IMAMPPlayerContinuous(**kw))

    runner.algo_factory.register_builder('amp', lambda **kw: amp_agent.AMPAgent(**kw))
    runner.algo_factory.register_builder('amp_self_play', lambda **kw: amp_self_play_agent.AMPSelfPlayAgent(**kw))
    runner.algo_factory.register_builder('im_amp', lambda **kw: im_amp.IMAmpAgent(**kw))

    runner.model_builder.model_factory.register_builder('amp', lambda net, **kw: amp_models.ModelAMPContinuous(net))
    runner.model_builder.network_factory.register_builder('amp', lambda **kw: amp_network_builder.AMPBuilder())
    runner.model_builder.network_factory.register_builder('amp_mcp',
                                                          lambda **kw: amp_network_mcp_builder.AMPMCPBuilder())
    runner.model_builder.network_factory.register_builder('amp_pnn',
                                                          lambda **kw: amp_network_pnn_builder.AMPPNNBuilder())
    runner.model_builder.network_factory.register_builder('amp_z', lambda **kw: amp_network_z_builder.AMPZBuilder())
    return runner


# ------------------------------
# Hydra Entrypoint
# ------------------------------
@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    global cfg, cfg_train

    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()

    # Setup flags
    for k in vars(flags):
        if hasattr(cfg, k):
            setattr(flags, k, cfg[k])

    if cfg.server_mode:
        flags.follow = cfg.follow = True
        flags.fixed = cfg.fixed = True
        flags.no_collision_check = True
        flags.show_traj = True
        cfg.env.episode_length = 99999999999999

    if cfg.real_traj:
        cfg.env.episode_length = 99999999999999
        flags.real_traj = True

    # Auto-resume
    if AutoResume:
        details = AutoResume.get_resume_details()
        if details:
            cfg.epoch = int(details['resume_epoch'])
            cfg.resume_str = details['wandb_id']
            print(f"[Auto Resume] {details['resume_path']} epoch {cfg.epoch} wandb: {cfg.resume_str}")

    cfg.train = not cfg.test
    cfg_train = cfg.learning

    # WandB setup
    if not cfg.no_log and not cfg.test and not cfg.debug:
        wandb.init(entity=cfg.wandb_entity, project=cfg.get("project_name", "egoquest"),
                   resume='allow', id=cfg.resume_str, notes=cfg.get("notes", "no notes"))
        wandb.config.update(cfg, allow_val_change=True)
        wandb.run.name = cfg.exp_name
        wandb.run.save()

    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    # Setup paths
    cfg_train['params']['config']['network_path'] = cfg.output_path
    cfg_train['params']['config']['train_dir'] = cfg.output_path
    cfg_train['params']['config']['num_actors'] = cfg.env.num_envs
    os.makedirs(cfg.output_path, exist_ok=True)

    # Resume checkpoint
    model_name = cfg_train['params']['config']['name']
    if cfg.epoch > 0:
        ckpt_path = osp.join(cfg.output_path, f"{model_name}_{cfg.epoch:08d}.pth")
        cfg_train['params']['load_checkpoint'] = True
        cfg_train['params']['load_path'] = ckpt_path
        print(f"Resuming from {ckpt_path}")
    elif cfg.epoch == -1:
        ckpt_path = osp.join(cfg.output_path, f"{model_name}.pth")
        if osp.exists(ckpt_path):
            cfg_train['params']['load_checkpoint'] = True
            cfg_train['params']['load_path'] = ckpt_path
            print(f"Resuming from {ckpt_path}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")

    # Build runner and start training
    observer = RLGPUAlgoObserver()
    runner = build_alg_runner(observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(cfg)


if __name__ == '__main__':
    main()
