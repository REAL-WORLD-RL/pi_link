#!/bin/bash
# 使用 remote policy 与环境交互并录制视频的示例脚本

# 配置参数
POLICY_URI="ws://127.0.0.1:8000"  # Policy server 地址
ENV_URI="ws://127.0.0.1:9010"     # Environment server 地址
TOTAL_EPISODES=10                  # 运行的总 episode 数
NUM_ENVS=1                         # 并行环境数
VIDEO_FOLDER="./demo_videos"       # 视频保存文件夹
FPS=30                             # 视频帧率

# 注意：对于 ManiSkill 环境，代码会自动拼接 base_camera 和 hand_camera 的图像

# 运行脚本
python pi_link/demo_gym_loop.py \
    --policy_uri "${POLICY_URI}" \
    --env_uri "${ENV_URI}" \
    --total_epoches ${TOTAL_EPISODES} \
    --total_envs ${NUM_ENVS} \
    --use_policy \
    --record_video \
    --video_folder "${VIDEO_FOLDER}" \
    --video_prefix "demo" \
    --fps ${FPS}

echo "完成！视频已保存到 ${VIDEO_FOLDER}"
