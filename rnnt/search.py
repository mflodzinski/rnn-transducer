import torch
import torch.nn.functional as F
import utils
import os

def main():
    CONFIG_PATH = "config/config.yaml"
    OUTPUT_FILE = "timit/transcriptions_val.txt"  

    config = utils.load_config(CONFIG_PATH)
    logger = utils.setup_logger(config)
    device = utils.setup_device(logger)
    
    train_data, test_data, val_data, tokenizer = utils.prepare_data_loaders(config)
    
    model = utils.initialize_model(config, tokenizer.vocab_size, device)
    model.eval()

    with open(OUTPUT_FILE, 'w') as f:
        for _, (inputs, inputs_length, targets, targets_length) in enumerate(val_data):
            inputs, inputs_length, targets, targets_length = (
            inputs.to(device),
            inputs_length.to(device),
            targets.to(device),
            targets_length.to(device),
        )
            decoded_sequences = model.recognize(inputs, inputs_length)
            decoded_sequences = tokenizer.ids2tokens(decoded_sequences)
            target_sequences = tokenizer.ids2tokens(targets.tolist())
            
            for decoded, target in zip(decoded_sequences, target_sequences):
                decoded_sentence = "".join(decoded)
                target_sentence = "".join(target)
                f.write(f"{decoded_sentence}\n{target_sentence}\n")

if __name__ == "__main__":
    main()
