# python kern_processor.py \
#     --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/vivaldi/vivaldi/op01/kern,data/unaligned/vivaldi/vivaldi/op02/kern,data/unaligned/beethoven/beethoven-string-quartet s/kern,data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern,data/unaligned/bach/chorales/bach-370-chorales/kern,data/unaligned/vocals_only/vocals_only" \
#     --output_dir "processed_data/all/no_barlines" \

# python kern_processor.py \
#     --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/vivaldi/vivaldi/op01/kern,data/unaligned/vivaldi/vivaldi/op02/kern,data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern,data/unaligned/bach/chorales/bach-370-chorales/kern,data/unaligned/vocals_only/vocals_only" \
#     --output_dir "processed_data/all/barlines" \
#     --want_barlines 

echo "no barlines"
python kern_processor.py \
    --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/vivaldi/vivaldi/op01/kern,data/unaligned/vivaldi/vivaldi/op02/kern,data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern,data/unaligned/bach/chorales/bach-370-chorales/kern,data/unaligned/vocals_only/vocals_only" \
    --output_dir "processed_data/elucidation_experiment/no_barlines" \
    --want_elucidation \

echo "barlines"
python kern_processor.py \
    --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/vivaldi/vivaldi/op01/kern,data/unaligned/vivaldi/vivaldi/op02/kern,data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern,data/unaligned/bach/chorales/bach-370-chorales/kern,data/unaligned/vocals_only/vocals_only" \
    --output_dir "processed_data/elucidation_experiment/barlines" \
    --want_barlines \
    --want_elucidation

echo "barlines and measure lengths"
python kern_processor.py \
    --input_dir "data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/vivaldi/vivaldi/op01/kern,data/unaligned/vivaldi/vivaldi/op02/kern,data/unaligned/beethoven/beethoven-string-quartets/kern,data/unaligned/bach/fugues/art-of-the-fugue/kern,data/unaligned/bach/fugues/bach-musical-offering/kern,data/unaligned/bach/fugues/bach-wtc-fugues/kern,data/unaligned/bach/fugues/inventions/kern,data/unaligned/bach/chorales/bach-370-chorales/kern,data/unaligned/vocals_only/vocals_only" \
    --output_dir "processed_data/elucidation_experiment/barlines_and_measure_lengths" \
    --want_barlines \
    --want_measure_lengths \
    --want_elucidation 


