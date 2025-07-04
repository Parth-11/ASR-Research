import json
import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset
import evaluate
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# Install required packages
# !pip install datasets transformers torchaudio librosa evaluate peft

# Configuration
DATA_JSON_PATH = "fleurs_punjabi/data.json"  # Path to your data.json file
AUDIO_DIR = "fleurs_punjabi/"  # Directory containing audio files
OUTPUT_DIR = "./wav2vec2-xlsr-punjabi-lora"
BASE_MODEL = "facebook/wav2vec2-large-xlsr-53"

# LoRA Configuration - Fixed for Wav2Vec2
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "lm_head",  # CTC head
        "wav2vec2.encoder.layers.0.attention.q_proj",
        "wav2vec2.encoder.layers.0.attention.k_proj", 
        "wav2vec2.encoder.layers.0.attention.v_proj",
        "wav2vec2.encoder.layers.0.attention.out_proj",
        "wav2vec2.encoder.layers.1.attention.q_proj",
        "wav2vec2.encoder.layers.1.attention.k_proj", 
        "wav2vec2.encoder.layers.1.attention.v_proj",
        "wav2vec2.encoder.layers.1.attention.out_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

def load_fluers_dataset(data_json_path: str, audio_dir: str, test_split: float = 0.2, max_samples: int = None):
    """Load and split the Fluers Punjabi dataset"""
    
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare data for Dataset creation
    dataset_dict = {
        'path': [],
        'sentence': [],
        'duration': []
    }
    
    valid_samples = 0
    for item in data:
        audio_path = os.path.join(audio_dir, item['audioFilename'])
        if os.path.exists(audio_path):
            # Filter out very short or very long audio files
            duration = item.get('duration', 0)
            if 0.5 <= duration <= 20.0:  # Between 0.5 and 20 seconds
                dataset_dict['path'].append(audio_path)
                dataset_dict['sentence'].append(item['text'])
                dataset_dict['duration'].append(duration)
                valid_samples += 1
                
                if max_samples and valid_samples >= max_samples:
                    break
    
    if not dataset_dict['path']:
        raise ValueError("No valid audio files found!")
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and test
    dataset = dataset.train_test_split(test_size=test_split, seed=42)
    
    print(f"Dataset loaded: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")
    return dataset['train'], dataset['test']

def show_random_elements(dataset, num_examples=5):
    """Display random elements from the dataset"""
    picks = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    sample_data = dataset[picks]
    
    df = pd.DataFrame({
        'sentence': sample_data['sentence'],
        'duration': sample_data['duration']
    })
    print("Sample data:")
    print(df.to_string(index=False))

def remove_special_characters(batch):
    """Remove special characters and normalize text"""
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\"\%\'\"\�\(\)\[\]]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).strip()
    if not batch["sentence"]:  # Handle empty strings
        batch["sentence"] = " "
    return batch

def extract_all_chars(batch):
    """Extract all unique characters from the dataset"""
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def create_vocab_and_tokenizer(train_dataset, test_dataset):
    """Create vocabulary and tokenizer from datasets"""
    
    # Extract vocabulary
    vocab_train = train_dataset.map(
        extract_all_chars, 
        batched=True, 
        batch_size=-1, 
        keep_in_memory=True, 
        remove_columns=train_dataset.column_names
    )
    vocab_test = test_dataset.map(
        extract_all_chars, 
        batched=True, 
        batch_size=-1, 
        keep_in_memory=True, 
        remove_columns=test_dataset.column_names
    )
    
    # Create vocabulary dictionary
    vocab_list = list(set(vocab_train[0]["vocab"]) | set(vocab_test[0]["vocab"]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    # Replace space with word delimiter
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    
    # Add special tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    print(f"Vocabulary size: {len(vocab_dict)}")
    print(f"Sample vocab: {list(vocab_dict.keys())[:20]}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save vocabulary
    vocab_path = os.path.join(OUTPUT_DIR, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=2)
    
    # Create tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|"
    )
    
    return tokenizer, vocab_dict

def speech_file_to_array_fn(batch):
    """Convert speech file to array with error handling"""
    try:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        # Convert to mono if stereo
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        
        batch["speech"] = speech_array[0].numpy()
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["sentence"]
        return batch
    except Exception as e:
        print(f"Error loading audio file {batch['path']}: {e}")
        # Return empty array for failed loads
        batch["speech"] = np.array([])
        batch["sampling_rate"] = 16000
        batch["target_text"] = batch["sentence"]
        return batch

def resample_audio(batch):
    """Resample audio to 16kHz with error handling"""
    try:
        if len(batch["speech"]) == 0:
            return batch
            
        if batch["sampling_rate"] != 16000:
            batch["speech"] = librosa.resample(
                batch["speech"], 
                orig_sr=batch["sampling_rate"], 
                target_sr=16000
            )
            batch["sampling_rate"] = 16000
        
        # Ensure minimum length
        if len(batch["speech"]) < 1000:  # Less than ~0.06 seconds
            batch["speech"] = np.pad(batch["speech"], (0, 1000 - len(batch["speech"])))
            
        return batch
    except Exception as e:
        print(f"Error resampling audio: {e}")
        # Return padded array for failed resampling
        batch["speech"] = np.zeros(1000)
        batch["sampling_rate"] = 16000
        return batch

def prepare_dataset(batch, processor):
    """Prepare dataset for training with error handling"""
    try:
        # Filter out empty audio
        valid_indices = [i for i, speech in enumerate(batch["speech"]) if len(speech) > 0]
        
        if not valid_indices:
            return {
                "input_values": [],
                "labels": []
            }
        
        # Get valid samples
        valid_speech = [batch["speech"][i] for i in valid_indices]
        valid_text = [batch["target_text"][i] for i in valid_indices]
        valid_sr = [batch["sampling_rate"][i] for i in valid_indices]
        
        # Check sampling rate consistency
        if len(set(valid_sr)) > 1:
            print(f"Warning: Inconsistent sampling rates: {set(valid_sr)}")
        
        # Process audio
        input_values = processor(
            valid_speech, 
            sampling_rate=valid_sr[0] if valid_sr else 16000
        ).input_values
        
        # Process text
        with processor.as_target_processor():
            labels = processor(valid_text).input_ids
        
        return {
            "input_values": input_values,
            "labels": labels
        }
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return {
            "input_values": [],
            "labels": []
        }

@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for CTC with padding"""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Filter out empty features
        valid_features = [f for f in features if len(f["input_values"]) > 0 and len(f["labels"]) > 0]
        
        if not valid_features:
            # Return empty batch
            return {
                "input_values": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long)
            }
        
        input_features = [{"input_values": feature["input_values"]} for feature in valid_features]
        label_features = [{"input_ids": feature["labels"]} for feature in valid_features]

        try:
            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

            return batch
        except Exception as e:
            print(f"Error in data collator: {e}")
            # Return minimal batch
            return {
                "input_values": torch.zeros(1, 1000),
                "labels": torch.full((1, 1), -100, dtype=torch.long)
            }

def compute_metrics(pred, processor, wer_metric):
    """Compute WER metric"""
    try:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"wer": 1.0}

def main():
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load dataset with limited samples for testing
    print("Loading Fluers Punjabi dataset...")
    train_dataset, test_dataset = load_fluers_dataset(DATA_JSON_PATH, AUDIO_DIR, max_samples=1000)
    
    # Show sample data
    show_random_elements(train_dataset)
    
    # Clean text
    print("Cleaning text data...")
    train_dataset = train_dataset.map(remove_special_characters)
    test_dataset = test_dataset.map(remove_special_characters)
    
    # Create vocabulary and tokenizer
    print("Creating vocabulary and tokenizer...")
    tokenizer, vocab_dict = create_vocab_and_tokenizer(train_dataset, test_dataset)
    
    # Create feature extractor and processor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=True
    )
    
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )
    
    # Process audio files
    print("Processing audio files...")
    train_dataset = train_dataset.map(
        speech_file_to_array_fn, 
        remove_columns=train_dataset.column_names,
        num_proc=1  # Reduce to 1 to avoid multiprocessing issues
    )
    test_dataset = test_dataset.map(
        speech_file_to_array_fn, 
        remove_columns=test_dataset.column_names,
        num_proc=1
    )
    
    # Resample audio
    print("Resampling audio to 16kHz...")
    train_dataset = train_dataset.map(resample_audio, num_proc=1)
    test_dataset = test_dataset.map(resample_audio, num_proc=1)
    
    # Filter out empty audio
    print("Filtering valid audio samples...")
    train_dataset = train_dataset.filter(lambda x: len(x["speech"]) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x["speech"]) > 0)
    
    print(f"After filtering: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    # Prepare dataset
    print("Preparing dataset for training...")
    def prepare_dataset_with_processor(batch):
        return prepare_dataset(batch, processor)
    
    train_dataset = train_dataset.map(
        prepare_dataset_with_processor,
        remove_columns=train_dataset.column_names,
        batch_size=4,  # Reduced batch size
        num_proc=1,
        batched=True
    )
    test_dataset = test_dataset.map(
        prepare_dataset_with_processor,
        remove_columns=test_dataset.column_names,
        batch_size=4,
        num_proc=1,
        batched=True
    )
    
    # Filter out empty processed samples
    train_dataset = train_dataset.filter(lambda x: len(x["input_values"]) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x["input_values"]) > 0)
    
    print(f"After processing: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    # Create data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    # Load base model
    print("Loading base model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        torch_dtype=torch.float32  # Ensure consistent dtype
    )
    
    # Freeze feature extractor
    model.freeze_feature_extractor()
    
    # Apply LoRA
    print("Applying LoRA...")
    try:
        model = get_peft_model(model, LORA_CONFIG)
        print("LoRA applied successfully!")
        model.print_trainable_parameters()
    except Exception as e:
        print(f"LoRA application failed: {e}")
        print("Continuing with full fine-tuning...")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=2,  # Reduced batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        eval_strategy="steps",
        num_train_epochs=10,  # Reduced for testing
        fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA available
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=3e-4,
        warmup_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=None,
        dataloader_num_workers=0,  # Disable multiprocessing for data loading
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, processor, wer_metric),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        processor.save_pretrained(OUTPUT_DIR)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = trainer.evaluate()
        print(f"Test WER: {test_results['eval_wer']:.4f}")
        
        # Test inference on a sample
        print("\nTesting inference on a sample...")
        test_sample_inference(model, processor, test_dataset, device)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

def test_sample_inference(model, processor, test_dataset, device):
    """Test inference on a random sample"""
    try:
        model.eval()
        
        # Get a random sample
        sample_idx = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[sample_idx]
        
        # Prepare input
        input_dict = processor(
            sample["input_values"], 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        
        # Get prediction
        with torch.no_grad():
            logits = model(**input_dict).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
        
        # Decode prediction
        prediction = processor.decode(pred_ids)
        reference = processor.decode(sample["labels"], group_tokens=False)
        
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")
        
    except Exception as e:
        print(f"Inference test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()