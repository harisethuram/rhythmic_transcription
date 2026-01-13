# lambdas=(0.1 0.3 0.5 0.7 1.0)
# beam_widths=(10)
# sigmas=(0.05 0.2 0.5)

lambdas=(1.0)
beam_widths=(10)
sigmas=(0.05)
counter=0
# if counter < 3 then use l40s else use l40

for beam_width in "${beam_widths[@]}"; do
    for lambda in "${lambdas[@]}"; do
        for sigma in "${sigmas[@]}"; do
            echo "Submitting job with beam_width: $beam_width, lambda: $lambda, sigma: $sigma"
            # sbatch run/sbatch/gaussian_decode.sh "$beam_width" "$lambda" "$sigma"
            if [ $counter -lt 4 ]; then
                sbatch run/sbatch/gaussian_decode_l40s.sh "$beam_width" "$lambda" "$sigma"
            else
                sbatch run/sbatch/gaussian_decode_l40.sh "$beam_width" "$lambda" "$sigma"
            fi
            ((counter++)) 
        done
    done
done

# sbatch run/sbatch/gaussian_decode_l40.sh 10 0.5 0.5