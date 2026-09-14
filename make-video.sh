#!/bin/bash

#SBATCH --nodes=60
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --time=00:15:00
#SBATCH --mem=50000
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --output=log/video-%j.out
#SBATCH --error=log/video-%j.err

OUTPUT_DIR=video

start=0
end=360
step=1


start_time=$(date +%s.%N)

set -o pipefail
source ../load_env.sh
cargo build --release


frame=0
angle=$start
N=$(echo "scale=0; (($end - $start) / $step)" | bc)

export OUTPUT_DIR start step N

echo "Rendering $N frames across $SLURM_NNODES nodes..."
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    frame=$SLURM_PROCID
    while (( frame < N )); do
        angle=$(echo "$start + $frame * $step" | bc)
        echo "[node $SLURM_PROCID] frame $frame/$N (angle=$angle)"
        ./target/release/render --output $OUTPUT_DIR/frames/frame$frame.png --dragon-angle $angle
        frame=$(( frame + SLURM_NNODES ))
    done
'

echo "Combining frames into video..."

/leonardo/pub/userexternal/mpoppe00/ffmpeg-git-20240629-amd64-static/ffmpeg -y -framerate 30 -i "$OUTPUT_DIR/frames/frame%d.png" \
       -c:v libx264 -pix_fmt yuv420p -crf 18 -frames:v $N \
       "$OUTPUT_DIR/output.mp4"

echo "Video written to $OUTPUT_DIR/output.mp4"

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc -l)

echo "----------------------------------------"
printf "Total time: %.2f seconds\n" "$elapsed"