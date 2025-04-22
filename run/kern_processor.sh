# python kern_processor.py \
#     --input_dir "data/bach-370-chorales/kern" \
#     --output_dir "processed_data/bach-370-chorales" \
    # --test_limit 10 \
    # --want_barlines \
    # 
# python kern_processor.py \
#     --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern" \
#     --output_dir "test/processed_data/beethoven" \
#     --want_barlines \
#     # --no_expressions \
#     # --debug 

# python kern_processor.py \
#     --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern" \
#     --output_dir "test/processed_data/beethoven2" \
    # --want_barlines \
    # --no_expressions \
    # --debug 

python kern_processor.py \
    --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/vivaldi/vivaldi/op01/kern,data/unaligned/vivaldi/vivaldi/op02/kern,data/unaligned/beethoven/beethoven-string-quartet s/kern,data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern,data/unaligned/bach/chorales/bach-370-chorales/kern,data/unaligned/vocals_only/vocals_only" \
    --output_dir "processed_data/all/no_barlines" \
#     --want_barlines \
    # --no_expressions \
    # --debug 

python kern_processor.py \
    --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/vivaldi/vivaldi/op01/kern,data/unaligned/vivaldi/vivaldi/op02/kern,data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern,data/unaligned/bach/chorales/bach-370-chorales/kern,data/unaligned/vocals_only/vocals_only" \
    --output_dir "processed_data/all/barlines" \
    --want_barlines 

# python kern_processor.py\
#     --input_dir "data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern" \
#     --output_dir "processed_data/test/"

# python kern_processor.py\
#     --input_dir "data/unaligned/bach/fugues/bach-musical-offering/kern" \
#     --output_dir "processed_data/test_barlines/yes" \
#     --want_barlines 

# python kern_processor.py\
#     --input_dir "data/unaligned/bach/fugues/bach-musical-offering/kern" \
#     --output_dir "processed_data/test_barlines/no" \
    # --want_barlines 

# python kern_processor.py \
#     --input_dir "data/unaligned/vocals_only/vocals_only" \
#     --output_dir test/processed_data2/vocals_only \