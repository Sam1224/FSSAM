gpu=$1  # 4
port=$2  # 1234
dataset=$3  # pascal/coco
exp_name=split$4  # 0/1/2/3
shot=$5  # 1/5
arch=$6  # FSSAM
net=$7  # small

if [ $shot -eq 1 ]; then
  postfix=batch
elif [ $gpu -eq 5 ]; then
  postfix=5s_batch
else
  echo "Only 1 and 5 shot are supported"
  exit 1
fi

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${net}/${dataset}_${exp_name}_${net}_${postfix}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")

echo ${arch}
echo ${config}

if [ $shot -eq 1 ]; then
  mkdir -p priors/${dataset}/${net}/${ver_dino}/${exp_name}/1shot
elif [ $shot -eq 5 ]; then
  mkdir -p priors/${dataset}/${net}/${ver_dino}/${exp_name}/5shot
else
  echo "Only 1 and 5 shot are supported"
  exit 1
fi

python3 -m torch.distributed.launch --nproc_per_node=${gpu} --master_port=${port} train.py \
        --config=${config} \
        --arch=${arch} \
        --num_refine=3 \
        --ver_refine=v1 \
        --ver_dino=dinov2_vitb14 \
        2>&1 | tee ${result_dir}/train-$now.log
