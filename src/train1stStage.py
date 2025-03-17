import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import gc
import argparse
import pandas as pd
from functools import partial
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
) 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model
from trainUtil import (
  ShuffleDatasetCallback,
  StreamingDatasetCallback,
  CPUOffloadCallback,
  CustomTrainer
)
from util import (
  prepareDocuments,
  answerWithRAG,
  modelUtility
)

def Train(
  dbTrainSetPath: str,
  flowTrainSetPath: str,
  savePath: str,
  RAGApiPath: str,
  RAGCodePath: str,
  modelSize: int,
  batchSize: int = 1,
  epoch: int = 2
):
  util = modelUtility("Qwen/Qwen2.5-Coder-" + str(modelSize) + "B-Instruct")
  maxSeqLength = 9554
  trainSet = pd.DataFrame(columns=["data", "code"])
  dataIndex = 0
  
  # load db data
  promptSet = pd.read_excel(dbTrainSetPath, 'prompt', header=None).iloc[1:].reset_index(drop=True)
  promptSet = promptSet.rename(columns={0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}) # 6 different tones
  codeSet = pd.read_excel(dbTrainSetPath, 'code', header=None).iloc[1:].reset_index(drop=True)
  codeSet = codeSet.rename(columns={0: "code"})
  for i in range(len(promptSet)):
    if i == 238:# Save the last 20 for testing
      break
    for j in range(6):# Loop through the 6 tones
      trainSet.loc[dataIndex, "data"] = promptSet[str(j)][i]
      trainSet.loc[dataIndex, "code"] = codeSet["code"][i]
      dataIndex += 1
  
  # load singel-stage flow data
  category = ["file", "floorplan", "io", "gpl", "mpl", "dpl", "cts", "filler", "pdn", "grt", "drt", "ir"]
  for cat in category:
    promptSet = pd.read_excel(flowTrainSetPath, cat +'_prompt', header=None).iloc[1:].reset_index(drop=True)
    promptSet = promptSet.rename(columns={0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}) # 6 different tones
    codeSet = pd.read_excel(flowTrainSetPath, cat +'_code', header=None).iloc[1:].reset_index(drop=True)
    codeSet = codeSet.rename(columns={0: "code"})
    for i in range(len(promptSet)):
      for j in range(6):# Loop through the 6 tones
        trainSet.loc[dataIndex, "data"] = promptSet[str(j)][i]
        trainSet.loc[dataIndex, "code"] = codeSet["code"][i]
        dataIndex += 1
  
  # load cross-stage data
  # Index for testing data points
  crossStageTestIndex = [208, 657, 797, 966, 1039, 1059, 1082, 1221, 1453, 1502, 1598, 1600, 1721, 1810, 1985, 2062, 2075, 2415, 2461, 2931]
  promptSet = pd.read_excel(flowTrainSetPath, 'cross_stage_prompt', header=None).iloc[1:].reset_index(drop=True)
  promptSet = promptSet.rename(columns={0: "prompt"})

  codeSet = pd.read_excel(flowTrainSetPath, 'cross_stage_code', header=None).iloc[1:].reset_index(drop=True)
  codeSet = codeSet.rename(columns={0: "code"})
  for i in range(len(promptSet)):
    if i in crossStageTestIndex: # filter out the testing data
      continue
    trainSet.loc[dataIndex, "data"] = promptSet["prompt"][i]
    trainSet.loc[dataIndex, "code"] = codeSet["code"][i]
    dataIndex += 1
  
  del promptSet, codeSet
  gc.collect()

  apiDf = pd.read_csv(RAGApiPath)
  apiDocuments, apiDocumentsDict = prepareDocuments(df=apiDf)
  templateDf = pd.read_csv(RAGCodePath)
  templateDocuments, templateDocumentsDict = prepareDocuments(df=templateDf, api=False)
  allSplits = apiDocuments + templateDocuments
  allDict = {**apiDocumentsDict, **templateDocumentsDict}

  embeddingModel = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
  embeddings = embeddingModel.encode(allSplits)

  trainingCodeString = """
```python
{code}
```<|im_end|>
"""
  
  for i in range(len(trainSet)):
    question = trainSet["data"][i]
    RAGContext = answerWithRAG(
      question,
      embeddings,
      embeddingModel,
      allSplits,
      allDict
    )
    if RAGContext == "":
      data = util.ragPromptTemplateWithoutContext.format(
        question=question,
        system_prompt=util.systemPrompt
      )
      data += trainingCodeString.format(code=trainSet["code"][i])
    else:
      data = util.ragPromptTemplateWithContext.format(
        question=question,
        context=RAGContext,
        system_prompt=util.systemPrompt
      )
      data += trainingCodeString.format(code=trainSet["code"][i])
    trainSet.loc[i, "data"] = data
  trainSet.drop(columns=["code"], inplace=True)
  gc.collect()

  trainSet.to_csv("temp_training_data.csv", index=False)
  del trainSet
  gc.collect()
  trainSet = load_dataset("csv", data_files={"train": "temp_training_data.csv"})["train"]
  trainSet = trainSet.shuffle(seed=118) # Hard code the seed for reproducibility
  
  trainingArguments = TrainingArguments(
    output_dir = "./OpenROAD-Agent-" + str(modelSize) + "B-1stStage-log",
    num_train_epochs = epoch,
    per_device_train_batch_size = batchSize,
    gradient_accumulation_steps = 1,
    optim = "adamw_bnb_8bit",
    save_steps = int(((len(trainSet)//batchSize + int(len(trainSet)%batchSize != 0)) * 1) / (1 * 1)),
    logging_steps = 1,
    learning_rate = 1e-4,
    logging_dir='./logs/',
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    remove_unused_columns = False,
    dataloader_pin_memory = True,  # Manage memory manually
    gradient_checkpointing = True,  # Enable gradient checkpointing
    gradient_checkpointing_kwargs={'use_reentrant':False}
  )

  tokenizer = AutoTokenizer.from_pretrained(
    util.modelName,
    pad_token = '<|im_end|>',
    eos_token = '<|im_end|>',
    cache_dir = None,
    truncation = True,
    padding_side = "left",
    trust_remote_code = True,
    max_length=maxSeqLength,
    device_map = "balanced_low_0"
  )
  tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})

  model = AutoModelForCausalLM.from_pretrained(
    util.modelName,
    device_map="balanced_low_0",  # Change to "balanced" or "auto" if your GPU has more memory
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
  )

  peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    inference_mode=False,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
      "q_proj",
      "up_proj",
      "o_proj",
      "k_proj",
      "down_proj",
      "gate_proj",
      "v_proj",
    ]
  )

  model = get_peft_model(model, peft_config)
  for name, param in model.named_parameters():
    if "lora" in name:
      param.requires_grad = True
    else:
      param.requires_grad = False

  def tokenizeFunction(trainSet, tokenizer):
    tokenizedInputs = tokenizer(
      trainSet["data"],
      truncation=False,
      return_tensors="pt",
      padding="max_length",
      padding_side = "left",
      max_length=maxSeqLength,
    )
    tokenizedInputs = {key: val for key, val in tokenizedInputs.items()}
    tokenizedInputs["labels"] = tokenizedInputs["input_ids"].clone()
    for key in ['input_ids', 'attention_mask', 'labels']:
      for i in range(len(tokenizedInputs[key])):
        if len(tokenizedInputs[key][i]) < maxSeqLength:
          pad_length = maxSeqLength - len(tokenizedInputs[key][i])
          pad_value = 0 if key == 'attention_mask' else tokenizer.pad_token_id.to(torch.float32)
          tokenizedInputs[key][i] = torch.cat([torch.full((pad_length,), pad_value, dtype=torch.float32), tokenizedInputs[key][i].to(torch.float32)])
        elif len(tokenizedInputs[key][i]) > maxSeqLength:
          tokenizedInputs[key][i] = tokenizedInputs[key][i][-maxSeqLength:]

    for key in tokenizedInputs:
      if isinstance(tokenizedInputs[key], torch.Tensor):
        tokenizedInputs[key] = tokenizedInputs[key].cpu()
    
    return tokenizedInputs

  tokenizeWithTokenizer = partial(tokenizeFunction, tokenizer=tokenizer)
  tokenizedPromptDataset = trainSet.map(tokenizeWithTokenizer, batched=True, cache_file_name="cached_tokenized_dataset")

  for i in range(len(tokenizedPromptDataset)):
    if len(tokenizedPromptDataset[i]['input_ids']) != maxSeqLength:
      print(f"The input_ids size is: {len(tokenizedPromptDataset[i]['input_ids'])}")
  # Ensure dataset is on CPU by explicitly setting device
  tokenizedPromptDataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], device='cpu')
  total_tokens = sum(len(example['input_ids']) for example in tokenizedPromptDataset)
  print(f"Total number of tokens in tokenizedPromptDataset: {total_tokens}")
  print(f"Dataset device: CPU (explicitly enforced)")

  shuffleDatasetCallback = ShuffleDatasetCallback()
  streamingDatasetCallback = StreamingDatasetCallback()
  cpuOffloadCallback = CPUOffloadCallback()
  trainer = CustomTrainer(
    model = model,
    train_dataset = tokenizedPromptDataset,
    args = trainingArguments,
    processing_class = tokenizer,
    callbacks=[shuffleDatasetCallback, streamingDatasetCallback, cpuOffloadCallback]
  )
  trainer.train()
  trainer.model.save_pretrained(savePath + "OpenROAD-Agent-" + str(modelSize) + "B-1stStage", saveEmbeddingLayers = True, safeSerialization=False, fromPt=True)
  tokenizer.save_pretrained(savePath + "OpenROAD-Agent-" + str(modelSize) + "B-1stStage")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "parsing the path")
  parser.add_argument("--dbTrainSetPath", type = str, default = "../EDA-Corpus-v2/DB-v2.xlsx")
  parser.add_argument("--flowTrainSetPath", type = str, default = "../EDA-Corpus-v2/Flow-v2.xlsx")
  parser.add_argument("--savePath", type = str, default = "./Saved_Model/")
  parser.add_argument("--RAGApiPath", type = str, default = "../RAGData/RAGAPIs.csv")
  parser.add_argument("--RAGCodePath", type = str, default = "../RAGData/RAGCodePiece.csv")
  parser.add_argument("--modelSize", type = int, default = 32)
  parser.add_argument("--batchSize", type = int, default = 1)
  parser.add_argument("--epoch", type = int, default = 2)
  pyargs = parser.parse_args()
  Train(
    dbTrainSetPath = pyargs.dbTrainSetPath,
    flowTrainSetPath = pyargs.flowTrainSetPath,
    RAGApiPath = pyargs.RAGApiPath,
    RAGCodePath = pyargs.RAGCodePath,
    savePath = pyargs.savePath,
    modelSize = pyargs.modelSize,
    batchSize = pyargs.batchSize,
    epoch = pyargs.epoch
  )
