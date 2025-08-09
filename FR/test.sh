# Example 1: HDR video quality assessment
CUDA_VISIBLE_DEVICES=0 python VQA_FR.py \
    --distorted 10_Fountain_HDR10_960x540_1500k.mp4 \
    --reference 10_Fountain_HDR10_3840x2160_50000k.mp4 \
    --model_path ckpts/FR_HDR_VQA.pth \
    --profile_path ckpts/FR_HDR_VQA.npy

# Example 2: Another HDR video
# CUDA_VISIBLE_DEVICES=0 python VQA_FR.py \
#     --distorted 14_Knitting_Close_HDR10_960x540_1500k.mp4 \
#     --reference 14_Knitting_Close_HDR10_3840x2160_50000k.mp4 \
#     --model_path ckpts/FR_HDR_VQA.pth \
#     --profile_path ckpts/FR_HDR_VQA.npy

# Example 3: SDR video quality assessment
# CUDA_VISIBLE_DEVICES=0 python VQA_FR.py \
#     --distorted 1_Basketball_Afternoon_SDR_960x540_800k.mp4 \
#     --reference 1_Basketball_Afternoon_SDR_3840x2160_50000k.mp4 \
#     --model_path ckpts/FR_HDR_VQA.pth \
#     --profile_path ckpts/FR_HDR_VQA.npy