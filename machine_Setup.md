download paligemma tokenizer

mkdir -p /home/ubuntu/workspace/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/model_bundle/assets

curl -L \
  -o /home/ubuntu/workspace/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/model_bundle/assets/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

hf download facebook/dinov3-vits16plus-pretrain-lvd1689m \
  --local-dir /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/model_bundle/weights/dino/dinov3-vits16plus-pretrain-lvd1689m

hf download huggingaccounttest/RO-DUME-COMP-CHUNK-17500-100 --local-dir /home/ubuntu/RO-Inference-Comp-Action-Chunk-Planner-Raw-Phase-2/model_bundle/weights/openpi/pi05_origami_comp_action_chunk_phase2


nvidia-smi

docker --version

which nvidia-ctk || echo "NVIDIA Container Toolkit NOT installed"

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker info | grep -i runtime

sudo docker run --rm --gpus all ubuntu nvidia-smi
