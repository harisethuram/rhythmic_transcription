performance_path=$1
score_path=$2
score_part_number=$3
output_dir=$4

echo "performance_path: $performance_path"
echo "score_path: $score_path"
echo "score_part_number: $score_part_number"
echo "output_dir: $output_dir"

python piece_wise_linear_fit.py \
    --performance_path $performance_path \
    --eval \
    --score_path $score_path \
    --score_part_number $score_part_number \
    --loss_fn "sin_loss" \
    --lbda 10 \
    --gamma 0.01 \
    --output_dir $output_dir \
    --dtw "hybrid" \
    # --test_len 10 \
    
# "data/URMP/08_Spring_fl_vn/Notes_1_fl_08_Spring.txt" \

# "data/URMP/08_Spring_fl_vn/Sco_08_Spring_fl_vn.mid" \