# Examples:

#   bash scripts/train_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/train_policy.sh dp_224x224_r3m gr1_dex-image 0913_example

dataset_path=/media/Image_Lab/embod_data/MarsMind_data


DEBUG=False
wandb_mode=offline


alg_name=idp3_arm
task_name=marsmind-arm
config_name=${alg_name}
addition_info=marsmind_arm_grasp_only-loss_p0-depth-0-8m_p1-depth-0-5m_p0-rgb_p1-rgb
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="/media/Image_Lab/hsb/idp3/${exp_name}_seed${seed}"

gpu_id=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    save_ckpt=False
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi


cd /home/hsb/prj/MMDP/Improved-3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.data_path=$dataset_path \
                            task.dataset.task_name=grasp




                                