# test out the trained barline s2s model

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import open_processed_data_dir
from src.preprocess.collate import preprocess_parallel_data
from src.const_tokens import *
from src.model.BarlineS2SLSTM import BarlineS2SLSTM
from src.model.BarlineS2STransformer import BarlineS2STransformer

if __name__ == "__main__":
    
    # model_path = "output/test_barline_s2s_expand/model.pth"
    model_path = "models/barline_s2s_hyperparam/lr_0.001/batch_size_64/hidden_dim_256/num_layers_1/model.pth"
    data_dir = "processed_data/all/parallel_barlines"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model = torch.load(model_path).to(DEVICE)
    
    # Load the data
    _, val_loader, token_to_id, id_to_token = preprocess_parallel_data(data_dir, batch_size=1, device=DEVICE)
    print("decoding...")
    with torch.no_grad():
        for _, input_data, target_data, src_key_padding_mask, _ in val_loader:
            # pass the input data through the model
            # print(token_to_id)
            curr_output = torch.Tensor([token_to_id[START_OF_SEQUENCE_TOKEN]]).to(DEVICE).long().unsqueeze(0)
            # print(curr_output)
            
            for i in tqdm(range(input_data.shape[-1] * 2)):
                # print(input_data.shape, curr_output.shape, src_key_padding_mask.shape)
                # print(curr_output)
                if isinstance(model, BarlineS2STransformer):
                    logits = model(
                        src=input_data,
                        tgt=curr_output,
                        src_key_padding_mask=src_key_padding_mask,
                    )
                else:
                    logits, _, _ = model(
                        src=input_data,
                        tgt=curr_output,
                    )
                
                # get the last token of the output
                # print(logits)
                last_token = logits[..., -1, :].argmax(dim=-1)
                # print(logits[..., -1, :])
                # input()
                
                # append the last token to the output
                curr_output = torch.cat((curr_output, last_token.unsqueeze(0)), dim=1)
                if last_token.item() == token_to_id[END_OF_SEQUENCE_TOKEN]:
                    break
                
            # print the output
            print("end:")
            print(curr_output[:, :100])
            print(curr_output.shape)
            # print the target data
            print(target_data[:, :100])
            print(target_data.shape)
            # print the input data
            print(input_data[:, :100])
            print(input_data.shape)
            
            input()
    
    