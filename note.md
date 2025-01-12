## PushT environment
1. observation space contains 5 parameters: **agent position**, **block position**, **block angle**, from **[0,0,0,0,0]** to **[ws,ws,ws,ws,pi*2]** (ws=512 is the size of the PyGame window)

2. action space contains 2 parameters: **agent goal position**, from **[0,0]** to **[ws,ws]**

3. step functiuon: **PD control**, according to the latest action, update agent's velocity by PD control (or P control) and then step physics in dt (1 / sim_hz)

4. The origin of the coordinate is located in the upper left corner of the window of pusht pygame.

5. sim_hz=100 means simulation frequency is 100hz and control frequency is 10 hz, n_steps = self.sim_hz // self.control_hz, thus in n_step range, agent execute action per 0.01s.

6. reward is generated by np.clip(coverage / self.success_threshold, 0, 1) and once converage surpasses the success threshold, the process ends.

## PushT keypoints environment
1. observation space: 40 dimension 

## Vectorized Environments
向量化环境Vectorized Environments，是运行多个（独立）子环境的环境，可以按顺序运行，也可以使用多处理并行运行。矢量化环境将一批action作为输入，并返回一批observation。

gym.vector.SyncVectorEnv： 其中的子环境按顺序执行。  
gym.vector.AsyncVectorEnv： 其中的子环境使用多进程并行执行。这将为每个子环境创建一个进程。

## DDPMScheduler
1. function **step**:  Sets the discrete timesteps used for the diffusion chain (to be run before inference).  
parameter:  
**num_inference_steps** (int)-The number of diffusion steps used when generating samples with a pre-trained model. If used, timesteps must be None.  
d**evice** (str or torch.device, optional)  
**timesteps** (List[int], optional) 

2. function **set_timesteps**: Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion process from the learned model outputs (most often the predicted noise).  
parameter:  
**model_output** (torch.FloatTensor) — The direct output from learned diffusion model.  
**timestep** (float) — The current discrete timestep in the diffusion chain.  
**sample** (torch.FloatTensor) — A current instance of a sample created by the diffusion process.  
**generator** (torch.Generator, optional) — A random number generator.  
**return_dict** (bool, optional, defaults to True) — Whether or not to return a DDPMSchedulerOutput or tuple.  
Returns:  
DDPMSchedulerOutput or tuple

## Diffusion U-Net low dim policy
### conditional sample
对应diffusion model中的sampling过程  
![sampling](https://miro.medium.com/v2/resize:fit:1400/1*r9kwLdZtngT5mgFPn3WXRQ.png)  
parameter:
condition_data, condition_mask,  
local_cond, global_cond
generator, **kwargs # keyword arguments to scheduler.step

在conditional sample过程中，首先通过`trajectory = torch.randn(size, dtype,device, generator)`  
随机采样生成全是噪声的trajectory,
对应sampling第一步，再在逆向去噪循环中，首先通过  
`trajectory[condition_mask] = condition_data[condition_mask]`  
对整个采样过程做condition，然后通过Conditional1DUnet模型预测得到model_output  
`model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)`  
对应第四步中的epsilon(xt,t), 利用DDPMScheduler的step函数完成推理对应整个第四步，循环结束最后再执行一遍  
`trajectory[condition_mask] = condition_data[condition_mask]`确保整个过程都有condition.

### predict action



