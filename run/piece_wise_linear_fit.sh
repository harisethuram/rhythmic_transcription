python piece_wise_linear_fit.py \
    --performance_path "data/URMP/08_Spring_fl_vn/Notes_1_fl_08_Spring.txt" \
    --score_path "data/URMP/08_Spring_fl_vn/Sco_08_Spring_fl_vn.mid" \
    --score_part_number 0 \
    --loss_fn "sin_loss" \
    --lbda 0.04 \
    --gamma 0.01 \
    --test_len 23
