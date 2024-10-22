# python utility_scripts/kern_processor.py \
#     --input_dir "data/bach-370-chorales/kern" \
#     --output_dir "processed_data/bach-370-chorales" \
    # --test_limit 10 \
    # --want_barlines \
    # 

python utility_scripts/kern_processor.py \
    --input_dir "data/art-of-the-fugue/kern,data/bach-musical-offering/kern,data/bach-wtc-fugues/kern,data/inventions/kern" \
    --output_dir "processed_data/bach_fugues" \
    --no_expressions \