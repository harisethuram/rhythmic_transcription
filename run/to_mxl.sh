input_path=$1
note_info_path=$2
output_dir=$3

python utility_scripts/note_tuple_list_to_music_xml.py \
    --input_path $input_path \
    --note_info_path $note_info_path \
    --output_dir $output_dir