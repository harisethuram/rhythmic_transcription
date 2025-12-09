lambdas=(0.1 0.3 0.5 0.7 0.9 1.0)
beam_widths=(5 10 15)
sigmas=(0.05 0.1 0.2 0.3 0.4 0.5)

for beam_width in "${beam_widths[@]}"; do
    for lambda in "${lambdas[@]}"; do
        for sigma in "${sigmas[@]}"; do
            echo "Submitting job with beam_width: $beam_width, lambda: $lambda, sigma: $sigma"
            sbatch run/sbatch/gaussian_decode.sh "$beam_width" "$lambda" "$sigma"
        done
    done
done