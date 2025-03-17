import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import gc
import argparse
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache  
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from peft import PeftModel
import multiprocessing
import time
import threading
import queue
from openpyxl import Workbook
import re
import math
from util import (
  readOpenROADOutput,
  runOpenROADShell,
  sendCommandOpenROAD,
  processCodeString,
  clearQueue,
  modelUtility,
  prepareDocuments,
  answerWithRAG,
  generate
)

def prepareTrainSet(
  dbTrainSetPath: str,
  flowTrainSetPath: str,
  flowTrainSetCombinationPath: str
):
  trainingCodeString = """
```python
{code}
```<|im_end|>
"""

  trainSet = pd.DataFrame(
    columns=[
      "prompt",
      "context",
      "wrongCode",
      "message",
      "correctCode",
      "loadDesignType"
    ]
  )
  dataIndex = 0

  # This is used to select which .odb file to load
  loadDesignTypeList = []
  with open(flowTrainSetCombinationPath, "r") as f:
    for line in f:
      loadDesignType = line.split(",")[0]
      if loadDesignType[-1].isdigit():
        loadDesignType = loadDesignType[:-1]
      if loadDesignType == "read odb":
        loadDesignType = "file"
      loadDesignTypeList.append(loadDesignType)

  # Load db data
  promptSet = pd.read_excel(dbTrainSetPath, 'prompt', header=None).iloc[1:].reset_index(drop=True)
  promptSet = promptSet.rename(columns={0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}) # 6 different tones
  codeSet = pd.read_excel(dbTrainSetPath, 'code', header=None).iloc[1:].reset_index(drop=True)
  codeSet = codeSet.rename(columns={0: "code"})
  for i in range(len(promptSet)):
    if i == 238: # Save the last 20 for testing
      break
    for j in range(6): # Loop through the 6 tones
      trainSet.loc[dataIndex, "prompt"] = promptSet[str(j)][i]
      trainSet.loc[dataIndex, "context"] = ""
      trainSet.loc[dataIndex, "wrongCode"] = ""
      trainSet.loc[dataIndex, "message"] = ""
      trainSet.loc[dataIndex, "correctCode"] = trainingCodeString.format(code=codeSet["code"][i])
      trainSet.loc[dataIndex, "loadDesignType"] = ""
      dataIndex += 1
  # Load cross-stage flow data
  # Index for testing data points
  crossStageTestIndex = [208, 657, 797, 966, 1039, 1059, 1082, 1221, 1453, 1502, 1598, 1600, 1721, 1810, 1985, 2062, 2075, 2415, 2461, 2931]
  promptSet = pd.read_excel(flowTrainSetPath, 'cross_stage_prompt', header=None).iloc[1:].reset_index(drop=True)
  promptSet = promptSet.rename(columns={0: "prompt"})

  codeSet = pd.read_excel(flowTrainSetPath, 'cross_stage_code', header=None).iloc[1:].reset_index(drop=True)
  codeSet = codeSet.rename(columns={0: "code"})
  for i in range(len(promptSet)):
    if i in crossStageTestIndex: # Filter out the testing data
      continue
    trainSet.loc[dataIndex, "prompt"] = promptSet["prompt"][i]
    trainSet.loc[dataIndex, "context"] = ""
    trainSet.loc[dataIndex, "wrongCode"] = ""
    trainSet.loc[dataIndex, "message"] = ""
    trainSet.loc[dataIndex, "correctCode"] = trainingCodeString.format(code=codeSet["code"][i])
    trainSet.loc[dataIndex, "loadDesignType"] = loadDesignTypeList[i] # Help decide which .odb file to load
    dataIndex += 1
  
  del promptSet, codeSet
  return trainSet

def prepareRAGData(
  RAGApiPath: str,
  RAGCodePath: str,
  embeddingModel: SentenceTransformer
):
  apiDf = pd.read_csv(RAGApiPath)
  apiDocuments, apiDocumentsDict = prepareDocuments(df=apiDf)
  del apiDf # Clean up memory
  gc.collect()
  
  templateDf = pd.read_csv(RAGCodePath)
  templateDocuments, templateDocumentsDict = prepareDocuments(df=templateDf, api=False)
  del templateDf # Clean up memory
  gc.collect()
  
  allSplits = apiDocuments + templateDocuments
  allDict = {**apiDocumentsDict, **templateDocumentsDict}
  
  del apiDocuments, templateDocuments, apiDocumentsDict, templateDocumentsDict # Clean up memory
  gc.collect()
  
  embeddings = embeddingModel.encode(allSplits)
  return embeddings, allSplits, allDict

def computeReward(generatedCommand, correctCode, traceback):
  with torch.no_grad():
    # Constants for reward calculation
    alpha = 1e-3 # Penalty for length difference
    beta = 1e-2 # Penalty for cross-entropy loss

    # Process on CPU first to minimize GPU memory usage
    logits = generatedCommand.squeeze()
    labels = correctCode.squeeze()
    
    # Calculate the difference in sequence lengths
    lengthDiff = abs(labels.shape[0] - logits.shape[0])

    # Adjust dimensions by padding or truncating
    minLength = min(logits.shape[0], labels.shape[0])
    logits = logits[:minLength]
    labels = labels[:minLength]
    
    # Compute cross-entropy loss on CPU first
    lossCEValue = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]), 
        labels.view(-1), 
        reduction="mean"
    ).item()
    
    # Calculate reward based on presence of traceback
    rewardValue = 10/lossCEValue if not traceback else -lossCEValue
    
    # Calculate final reward value
    finalReward = rewardValue - alpha * lengthDiff - beta * lossCEValue
    print("ce: ", lossCEValue, ", lengthDiff: ", lengthDiff, ", reward: ", rewardValue, ", finalReward: ", finalReward)
    # Clean up CPU memory
    del logits, labels
    gc.collect()
  # Return reward as a tensor with requires_grad=True
  reward = torch.tensor(finalReward, device="cuda", requires_grad=True)
  # Ensure memory is freed
  torch.cuda.empty_cache()
  return reward

def policyGradientStep(model, optimizer, query, response, reward, stepCount=None):
  optimizer.zero_grad()
  # Move response to GPU once and remove query prefix
  response = response.squeeze()[query.shape[-1]:].to("cuda")
  tokenCount = response.shape[0]
  
  # Process query once
  query_cuda = query.to("cuda")
  
  # Initialize pastKeyValues and process first token
  pastKeyValues = DynamicCache()
  
  # Process first token (query)
  with torch.no_grad():
    outputs = model(
      input_ids=query_cuda,
      past_key_values=pastKeyValues,
      use_cache=True
    )
    pastKeyValues = outputs.past_key_values
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    # Process response tokens and collect log probabilities
    all_log_probs = []
    for i in range(tokenCount):
      if i > 0:  # Skip first iteration as we already processed the query
        nextToken = response[i-1].unsqueeze(0).unsqueeze(0)
        outputs = model(
          input_ids=nextToken,
          past_key_values=pastKeyValues,
          use_cache=True
        )
        pastKeyValues = outputs.past_key_values
        nextTokenLogits = outputs.logits[:, -1, :]
        logProbs = torch.nn.functional.log_softmax(nextTokenLogits, dim=-1)
        tokenIdx = response[i].unsqueeze(0).unsqueeze(0)
        tokenLogProb = logProbs.gather(1, tokenIdx).squeeze()
        all_log_probs.append(tokenLogProb)
        
        # Clean up memory
        del nextToken, outputs, nextTokenLogits, logProbs, tokenIdx
        torch.cuda.empty_cache()
      else:
        # For the first response token, use the last token from query outputs
        nextToken = response[i].unsqueeze(0).unsqueeze(0)
        outputs = model(
          input_ids=nextToken,
          past_key_values=pastKeyValues,
          use_cache=True
        )
        pastKeyValues = outputs.past_key_values
        nextTokenLogits = outputs.logits[:, -1, :]
        logProbs = torch.nn.functional.log_softmax(nextTokenLogits, dim=-1)
        if i+1 < tokenCount:
          tokenIdx = response[i+1].unsqueeze(0).unsqueeze(0)
          tokenLogProb = logProbs.gather(1, tokenIdx).squeeze()
          all_log_probs.append(tokenLogProb)
          del tokenIdx
          gc.collect()
        # Clean up memory
        del nextToken, outputs, nextTokenLogits, logProbs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Compute average log probability
    if all_log_probs:
      logProbSum = torch.stack(all_log_probs).sum()
      avgLogProb = logProbSum / len(all_log_probs)
      del logProbSum, all_log_probs
    else:
      avgLogProb = torch.tensor(0.0, device="cuda")
      del all_log_probs
    gc.collect()
    torch.cuda.empty_cache()
  # Compute loss
  loss = -avgLogProb * reward.to("cuda")
  del avgLogProb
  gc.collect()
  torch.cuda.empty_cache()

  # Check for NaN
  if torch.isnan(loss):
    del query, query_cuda, response, pastKeyValues, loss, reward
    gc.collect()
    torch.cuda.empty_cache()
    return

  # Backpropagate and update
  loss.backward()
  optimizer.step()
  
  # Clean up memory
  del query, query_cuda, response, pastKeyValues, loss, reward
  gc.collect()
  torch.cuda.empty_cache()

  return

def TrainRL(
  flowTrainSetCombinationPath: str,
  dbTrainSetPath: str,
  flowTrainSetPath: str,
  savePath: str,
  loadModelPath: str,
  RAGApiPath: str,
  RAGCodePath: str,
  OpenROADPath: str,
  logSavePath: str,
  modelSize: int,
  epoch: int,
  loadDesignTime: float,
  maxTestCaseWaitTime: float,
  commandFlushTime: float
):
  util = modelUtility("Qwen/Qwen2.5-Coder-" + str(modelSize) + "B-Instruct")
  
  embeddingModel = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
  trainSet = prepareTrainSet(
    dbTrainSetPath,
    flowTrainSetPath,
    flowTrainSetCombinationPath
  )
  embeddings, allSplits, allDict = prepareRAGData(
    RAGApiPath,
    RAGCodePath,
    embeddingModel
  )
  gc.collect()

  RAGContext = ""
  for i in range(len(trainSet)):
    question = trainSet["prompt"][i]
    RAGContext = answerWithRAG(
      question,
      embeddings,
      embeddingModel,
      allSplits,
      allDict
    )
    trainSet.loc[i, "context"] = RAGContext
  # Free memory after each iteration
  del RAGContext
  gc.collect()
  torch.cuda.empty_cache()

  del embeddings, allSplits, allDict, embeddingModel
  torch.cuda.empty_cache()
  gc.collect()

  trainSet = Dataset.from_pandas(trainSet).shuffle(seed = 118)

  tokenizer = AutoTokenizer.from_pretrained(
    util.modelName,
    pad_token = '<|endoftext|>',
    eos_token = '<|im_end|>',
    cache_dir = None,
    truncation = True,
    padding_side = "right",
    trust_remote_code = True,
    device_map = "balanced_low_0"
  )
  tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})
  
  model = AutoModelForCausalLM.from_pretrained(
    util.modelName,
    device_map="balanced_low_0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
  )
  model = PeftModel.from_pretrained(
    model,
    loadModelPath,
    is_trainable=True,
    device_map="balanced_low_0"
  )
  
  for name, param in model.named_parameters():
    if "lora" in name:
      param.requires_grad = True
    else:
      param.requires_grad = False

  #RL environment setup
  model.train()
  learning_rate = 1e-5
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
  stepCount = 0
  multiprocessing.set_start_method('spawn', force=True)
  masterOpenROAD, slaveOpenROAD = None, None
  OpenROADProcess = None
  
  # Set sleep count to prepare the process
  # Start the OpenROAD and LLM shell and keep it running
  masterOpenROAD, slaveOpenROAD = os.openpty()  # Create a pseudo-terminal

  OpenROADOutputQueue = queue.Queue()

  stopEvent = threading.Event()
  # Create threads to read both stdout and stderr into a single queue
  OpenROADStdoutThread = threading.Thread(
    target=readOpenROADOutput, 
    args=(
      masterOpenROAD,
      OpenROADOutputQueue,
      'STDOUT',
      stopEvent
    )
  )
  
  OpenROADStdoutThread.daemon = True
  OpenROADStdoutThread.start()

  time.sleep(loadDesignTime)

  trainingLog = Workbook()
  trainingLogIter = trainingLog.active
  trainingLogIter["A1"] = "prompt"
  trainingLogIter["B1"] = "correct code"
  trainingLogIter["C1"] = "previously generated code"
  trainingLogIter["D1"] = "previously received message"
  trainingLogIter["E1"] = "newly generated code"
  trainingLogIter["F1"] = "newly received message"
  trainingLogIter["G1"] = "load design type"

  totalEpisode = math.ceil(len(trainSet) * epoch)
  for episode in range(totalEpisode):
    print("==================="+str(episode+1)+"/"+str(totalEpisode)+"===================")
    prompt = ""
    if trainSet["context"][episode] == "":
      prompt = util.ragPromptTemplateWithoutContext.format(
        question=trainSet["prompt"][episode],
        system_prompt=util.systemPrompt
      )
    else:
      prompt = util.ragPromptTemplateWithContext.format(
        question=trainSet["prompt"][episode],
        context=trainSet["context"][episode],
        system_prompt=util.systemPrompt
      )

    trainingLogIter["A"+str(stepCount+2)] = trainSet["prompt"][episode]
    trainingLogIter["B"+str(stepCount+2)] = trainSet["correctCode"][episode]
    if OpenROADProcess is not None:
      clearQueue(OpenROADOutputQueue)
      OpenROADProcess.terminate()
      OpenROADProcess.wait()
    # Start the OpenROAD process
    OpenROADProcess = runOpenROADShell(
      OpenROADPath,
      loadDesignTime,
      slaveOpenROAD,
      trainSet["loadDesignType"][episode]
    )
    
    episodeLength = 5
    while episodeLength > 0:
      time.sleep(loadDesignTime)
      clearQueue(OpenROADOutputQueue)

      tokenizedPrompt = tokenizer.encode(prompt, return_tensors="pt").to("cpu")
      outputIds, allTokenLogits = generate(
        tokenizer = tokenizer,
        model = model,
        prompt = prompt,
        pastKeyValues = DynamicCache(),
        temperature=0.3,
        topP=0.9,
        maxNewTokens=2048,
        returnLogits = True
      )
      generatedCommand = tokenizer.decode(outputIds[0]).split("```python")[-1].split("```")[0]
      trainingLogIter["E"+str(stepCount+2)] = generatedCommand
      processedGeneratedCommand = processCodeString(generatedCommand)

      # Send commands to OpenROAD and handle potential process termination
      while True:
        try:
          stdout, traceback = sendCommandOpenROAD(
            OpenROADProcess,
            processedGeneratedCommand,
            OpenROADOutputQueue,
            maxTestCaseWaitTime,
            commandFlushTime
          )
          break  # Exit loop if the command runs successfully
        except RuntimeError:
          print("OpenROADProcess terminated. Restarting the OpenROAD shell.")
          OpenROADProcess = runOpenROADShell(
            OpenROADPath,
            loadDesignTime,
            slaveOpenROAD,
            trainSet["loadDesignType"][episode]
          )  # Restart the OpenROAD subprocess
          # Clear the output queues
          time.sleep(loadDesignTime)
          clearQueue(OpenROADOutputQueue)
      # Update the initial question based on output or errors
      stdout = stdout.encode('utf-8')
      stdout = re.sub(b'\x1b\].*?\n', b'', stdout)
      stdout = re.sub(b'\x1b\].*?\x07', b'', stdout)
      stdout = stdout.decode('utf-8').strip()
      trainingLogIter["F"+str(stepCount+2)] = stdout

      # Compute reward based on execution result and the generated quality
      correctCodeTokens = tokenizer.encode(trainSet["correctCode"][episode], return_tensors="pt").to("cpu")
      reward = computeReward(allTokenLogits, correctCodeTokens, traceback).to("cpu")
      # Free memory immediately after computing reward
      del allTokenLogits, correctCodeTokens
      gc.collect()
      torch.cuda.empty_cache()
      
      # Take an update step using the reward
      policyGradientStep(
        model = model,
        optimizer = optimizer,
        query = tokenizedPrompt,
        response = outputIds[0],
        reward = reward,
        stepCount = stepCount
      )
      
      # Free memory after policy gradient step
      del reward, tokenizedPrompt, outputIds
      gc.collect()
      torch.cuda.empty_cache()

      # Save the model and tokenizer
      model.save_pretrained(
        savePath + "OpenROAD-Agent-" + str(modelSize) + "B-3rdStage",
        saveEmbeddingLayers = True,
        safeSerialization=False,
        fromPt=True
      )
      tokenizer.save_pretrained(savePath + "OpenROAD-Agent-" + str(modelSize) + "B-3rdStage")
      # Save the interaction log (Offline manual selection is needed to enrich the dataset)
      trainingLog.save(logSavePath)

      stepCount += 1

      # Move onto the next episode if the generated code passes
      if not traceback:
        # Free memory before breaking the loop
        del stdout, traceback, processedGeneratedCommand, generatedCommand
        torch.cuda.empty_cache()
        gc.collect()
        break

      # restructure prompt to include error message and the previous generated code
      if trainSet["context"][episode] == "":
        prompt = util.ragWrongCodePromptTemplateWithoutContext.format(
          question=trainSet["prompt"][episode],
          system_prompt=util.systemPrompt,
          wrongCode=generatedCommand,
          message=stdout
        )
        # Log the prompt into the training log
        trainingLogIter["A"+str(stepCount+2)] = trainSet["prompt"][episode]
      else:
        prompt = util.ragWrongCodePromptTemplateWithContext.format(
          question=trainSet["prompt"][episode],
          context=trainSet["context"][episode],
          system_prompt=util.systemPrompt,
          wrongCode=generatedCommand,
          message=stdout
        )
        # Log the prompt into the training log
        trainingLogIter["A"+str(stepCount+2)] = trainSet["prompt"][episode]
      # Log the correct code, the previously generated code, and the previously received message into the training log
      trainingLogIter["B"+str(stepCount+2)] = trainSet["correctCode"][episode]
      trainingLogIter["C"+str(stepCount+2)] = generatedCommand
      trainingLogIter["D"+str(stepCount+2)] = stdout
      trainingLogIter["G"+str(stepCount+2)] = trainSet["loadDesignType"][episode]
      # Free memory before next iteration
      del generatedCommand, stdout, traceback, processedGeneratedCommand
      torch.cuda.empty_cache()
      gc.collect()
      episodeLength -= 1
      
  # Clean up resources after all episodes
  if OpenROADProcess is not None:
    OpenROADProcess.terminate()
    OpenROADProcess.wait()
    OpenROADProcess = None
  
  stopEvent.set()
  OpenROADStdoutThread.join(timeout=2)
  os.close(masterOpenROAD)
  os.close(slaveOpenROAD)
  trainingLog.save(logSavePath)

  torch.cuda.empty_cache()
  gc.collect()
  model.save_pretrained(
    savePath + "OpenROAD-Agent-" + str(modelSize) + "B-3rdStage",
    saveEmbeddingLayers = True,
    safeSerialization=False,
    fromPt=True)
  tokenizer.save_pretrained(savePath + "OpenROAD-Agent-" + str(modelSize) + "B-3rdStage")
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "parsing the path")
  parser.add_argument("--flowTrainSetCombinationPath", type = str, default = "../EDA-Corpus-v2/task_combinations.txt")
  parser.add_argument("--dbTrainSetPath", type = str, default = "../EDA-Corpus-v2/DB-v2.xlsx")
  parser.add_argument("--flowTrainSetPath", type = str, default = "../EDA-Corpus-v2/Flow-v2.xlsx")
  parser.add_argument("--savePath", type = str, default = "./Saved_Model/")
  parser.add_argument("--loadModelPath", type = str, default = "./Saved_Model/OpenROAD-Agent-32B-2ndStage")
  parser.add_argument("--RAGApiPath", type = str, default = "../RAGData/RAGAPIs.csv")
  parser.add_argument("--RAGCodePath", type = str, default = "../RAGData/RAGCodePiece.csv")
  parser.add_argument('--OpenROADPath', type=str, default='../OpenROAD/build/src/openroad')
  parser.add_argument("--logSavePath", type = str, default = "./training_log.xlsx")
  parser.add_argument("--modelSize", type = int, default = 32)
  parser.add_argument("--epoch", type = int, default = 0.01)
  parser.add_argument('--loadDesignTime', type=float, default=2)
  parser.add_argument('--maxTestCaseWaitTime', type=float, default=120)
  parser.add_argument('--commandFlushTime', type=float, default=0.1)
  pyargs = parser.parse_args()
  TrainRL(
    flowTrainSetCombinationPath = pyargs.flowTrainSetCombinationPath,
    dbTrainSetPath = pyargs.dbTrainSetPath,
    flowTrainSetPath = pyargs.flowTrainSetPath,
    RAGApiPath = pyargs.RAGApiPath,
    RAGCodePath = pyargs.RAGCodePath,
    savePath = pyargs.savePath,
    loadModelPath = pyargs.loadModelPath,
    OpenROADPath = pyargs.OpenROADPath,
    logSavePath = pyargs.logSavePath,
    modelSize = pyargs.modelSize,
    epoch = pyargs.epoch,
    loadDesignTime = pyargs.loadDesignTime,
    maxTestCaseWaitTime = pyargs.maxTestCaseWaitTime,
    commandFlushTime = pyargs.commandFlushTime
  )












