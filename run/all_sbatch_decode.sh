chunks=(0 1 2 3)
for chunk in "${chunks[@]}"; do
    sbatch run/sbatch_decode.sh $chunk
done
chunks=(4 5 6 7)
for chunk in "${chunks[@]}"; do
    sbatch run/sbatch_decode2.sh $chunk
done