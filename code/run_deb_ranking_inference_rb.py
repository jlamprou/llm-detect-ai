import sys

sys.path.insert(0, '/kaggle/input/omegaconf')
sys.path.insert(0, '/kaggle/input/utils-ai-v10')

import argparse
import os
import gc

import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import PeftModel

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from r_ranking.ai_dataset import AiDataset
from r_ranking.ai_loader import AiCollator, show_batch
from r_ranking.ai_model import AiModel

char_to_remove = ['{', '£', '\x97', '¹', 'å', '\\', '\x85', '<', '\x99', \
                  'é', ']', '+', 'Ö', '\xa0', '>', '|', '\x80', '~', '©', \
                  '/', '\x93', '$', 'Ó', '²', '^', ';', '`', 'á', '*', '(', \
                  '¶', '®', '[', '\x94', '\x91', '#', '-', 'ó', ')', '}', '=']

def preprocess_text(text):
    text = text.encode("ascii", "ignore").decode('ascii')        
    text = text.strip()
    text = text.strip("\"")

    for c in char_to_remove:
        text = text.replace(c, "")

    if text[-1]!=".":
        text = text.split(".")
        text = ".".join(text[:-1])
        text += "."
    return text


def run_inference(accelerator, model, infer_dl, example_ids):
    model.eval()
    all_predictions = []

    progress_bar = tqdm(range(len(infer_dl)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(infer_dl):
        with torch.no_grad():
            logits, _ = model(**batch)

        logits = logits.reshape(-1)
        predictions = torch.sigmoid(logits)
        predictions = accelerator.gather_for_metrics(predictions)
        predictions = predictions.cpu().numpy().tolist()

        all_predictions.extend(predictions)

        progress_bar.update(1)
    progress_bar.close()

    result_df = pd.DataFrame()
    result_df["tweet_id"] = example_ids
    result_df["generated"] = all_predictions

    return result_df

def main(cfg, save_dir, model_id, csv_path):
    
    # create accelerator
    accelerator = Accelerator()
    
    # read test data
    test_df = pd.read_csv(csv_path, sep=',')
    test_df['text'] = test_df['text'].apply(preprocess_text)
    accelerator.print(f'Test csv shape: {test_df.shape}')
    test_df['generated'] = 1 # TODO: NEEDED NOW, FIx it 
    
    with accelerator.main_process_first():
        dataset_creator = AiDataset(cfg)
        infer_ds = dataset_creator.get_dataset(test_df)
    
    tokenizer = dataset_creator.tokenizer
    
    infer_ds = infer_ds.sort("input_length")
    infer_ds.set_format(
        type=None,
        columns=[
            'tweet_id',
            'input_ids',
            'attention_mask',
            'generated'
        ]
    )
    
    infer_ids = infer_ds["tweet_id"]  # .tolist()
    
    #--
    data_collator = AiCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64
    )

    infer_dl = DataLoader(
        infer_ds,
        batch_size=cfg.predict_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    accelerator.print("~~"*40)
    accelerator.wait_for_everyone()
    
    
    #----------
    for b in infer_dl:
        break
    show_batch(b, tokenizer, task='infer', print_fn=accelerator.print)
    accelerator.print("~~"*40)
    #----------
    # model -----------------------------------------------------------------------------#
    model = AiModel(cfg, accelerator.device)

    checkpoint_path = cfg.predict_params.checkpoint_path
    accelerator.print("=="*50)
    accelerator.print(f"loading model from checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    del ckpt
    gc.collect()
    print("loaded!")
    accelerator.print("### Loaded Model Weights ###")
    
    model, infer_dl = accelerator.prepare(model, infer_dl)
    
    # run inference ---
    sub_df = run_inference(accelerator, model, infer_dl, infer_ids)
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        save_path = os.path.join(save_dir, f"{model_id}.parquet")
        sub_df.to_parquet(save_path)
        accelerator.print("done!")
        accelerator.print("~~"*40)
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--model_id', type=str, required=True)
    ap.add_argument('--csv_path', type=str, required=True)

    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)

    os.makedirs(args.save_dir, exist_ok=True)

    # execution
    main(
        cfg,
        save_dir=args.save_dir,
        model_id=args.model_id,
        csv_path=args.csv_path
    )
