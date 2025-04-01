modes="1/4,1/3,1/2,3/4,1"
spreads="0.05,0.05,0.05,0.05,0.05"
output_dir="models/tmp/test_beta"

python fit_beta_channel_model.py \
    --modes $modes \
    --spreads $spreads \
    --output_dir $output_dir