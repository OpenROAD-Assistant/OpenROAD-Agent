import os
import torch
import numpy as np
import pandas as pd
from transformers.cache_utils import DynamicCache
import time
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer
)
import time
import subprocess
import queue
import gc
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def readOpenROADOutput(
  master,
  outputQueue,
  pipeName,
  stopEvent
):
  # Continuously read output from the OpenROAD process
  try:
    while not stopEvent.is_set():
      # Read from the master end of the pty
      try:
        data = os.read(master, 1024).decode("utf-8")  # Handle decoding errors
        if data:
          # Put the read data into the output queue
          outputQueue.put((pipeName, data.strip()))
      except Exception as e:
        print(f"Error in readOpenROADOutput: {e}")
        break
  except Exception as e:
    print(f"Error in readOpenROADOutput: {e}")
  return

def runOpenROADShell(
  OpenROADPath,
  sleepTime,
  slave,
  loadDesignType
):
  #Start the Python shell process and keep it alive.
  try:
    # Start the OpenROAD process using subprocess
    process = subprocess.Popen(
      [OpenROADPath, "-python", "-u", "-i"],
      stdin=subprocess.PIPE,
      stdout=slave,
      stderr=slave,
      text=True,
      bufsize=1,
      universal_newlines=True
    )
    time.sleep(sleepTime)  # Wait for the shell to fully initialize
    loadDesignCommand = loadDesign("6_final", loadDesignType)
    for command in loadDesignCommand:
      # Send the design loading commands to the OpenROAD process
      process.stdin.write(command + "\n")
      process.stdin.flush()

    return process
  except subprocess.SubprocessError as e:
    print(f"Failed to start OpenROAD shell: {e}")
  raise

def sendCommandOpenROAD(
  process,
  command,
  outputQueue,
  maxWaitTime = 60,
  sleepTime = 0.05
):
  #Send a command to the persistent OpenROAD shell and capture the output.
  if process.poll() is not None:
    raise RuntimeError("OpenROAD shell process has terminated")

  stdoutOutput = list()
  traceback = False

  if len(command) == 0:
    return "No command generated!!!", 0, False, True

  for i, cmd in enumerate(command):
    # Send each command to the OpenROAD process
    process.stdin.write(cmd + "\n")
    process.stdin.flush()
    start_time = time.time()
    while True:
      if time.time() - start_time > maxWaitTime:
        print("Time out!!!!!!")
        stdoutOutput.append("Time out, might be caused by running:")
        for j in range(i+1):
          stdoutOutput.append(command[j])
        traceback = True
        break
      try:
        if not outputQueue.empty():
          _, line = outputQueue.get_nowait()
          # Check for errors in the output
          if "traceback" in line.lower() or "syntaxerror" in line.lower():
            traceback = True
          # Detect end of command prompt
          if ">>>" in line or "..." in line:
            line = line.split(">>>")[0]
            line = line.split("...")[0]
            stdoutOutput.append(line)
            break
          else:
            stdoutOutput.append(line)
        else:
          time.sleep(sleepTime)  # Avoid busy-waiting
      except queue.Empty:
        pass

    if traceback:
      print("Warnings or traceback")
      break

    if process.poll() is not None:
      raise RuntimeError("OpenROAD shell process terminated unexpectedly")
  outputMessage = '\n'.join(stdoutOutput)
  return outputMessage, traceback

def processCodeString(
  codeString: str
):
  # Process the code string to manage indentation
  codeString = codeString.strip()
  lines = codeString.splitlines()  # Split the string into lines
  result = list()
  indentationStack = list()
  indentationLevel = list()
  for line in lines:
    strippedLine = line.lstrip()  # Remove leading whitespace only
    if strippedLine:  # Skip completely empty lines
      currentIndentation = len(line) - len(strippedLine)
      indentationLevel.append(currentIndentation)
      # Check indentation and close the blocks as necessary
      while indentationStack and currentIndentation < indentationStack[-1]:
        if currentIndentation == 0:
          if not strippedLine.startswith(("else", "elif")):
            result.append("")
        indentationStack.pop()
      # Track the current indentation
      if not indentationStack or currentIndentation > indentationStack[-1]:
        indentationStack.append(currentIndentation)
      result.append(line)
  # Ensure only one empty string is added if there are remaining indentations
  if indentationStack:
    result.append("")

  if len(indentationLevel) > 0: #if the generated code is empty, return empty list
    if len(indentationLevel) > 1:
      indentationLevel = [abs(indentationLevel[i] - indentationLevel[i-1]) for i in range(1, len(indentationLevel))]
    indentationLevel = np.min(indentationLevel)
    if indentationLevel != 0:
      for i in range(len(result)):
        if result[i] != "":
          stripedLine = result[i].lstrip()
          level = (len(result[i]) - len(stripedLine)) / indentationLevel
          result[i] = "    " * int(level) + stripedLine

  return result

def loadDesign(
  designName: str,
  loadDesignType: str
):
  if loadDesignType == "":
    loadDesignCommand = [
    'import openroad',
    'import odb',
    'from openroad import Tech, Design, Timing',
    'from pathlib import Path',
    'design_name = "' + designName + '"',
    'tech = Tech()',
    'libDir = Path("../Design/nangate45/lib/")',
    'lefDir = Path("../Design/nangate45/lef/")',
    'designDir = Path("../Design/")',
    'libFiles = libDir.glob("*.lib")',
    'lefFiles = lefDir.glob("*.lef")',
    'for libFile in libFiles:',
    '    tech.readLiberty(libFile.as_posix())',
    '',
    'tech.readLef("%s/NangateOpenCellLibrary.tech.lef"%lefDir.as_posix())',
    '',
    'for lefFile in lefFiles:',
    '    tech.readLef(lefFile.as_posix())',
    '',
    'design = Design(tech)',
    'defFile = "%s/%s.def"%(designDir.as_posix(), design_name)',
    'design.readDef(defFile)',
    'design.evalTclString("create_clock -period 20 [get_ports clk] -name core_clock")',
    'design.evalTclString("set_propagated_clock [get_clocks {core_clock}]")',
    'design.evalTclString("source ../Design/nangate45/setRC.tcl")',
    'VDDNet = design.getBlock().findNet("VDD")',
    'if VDDNet is None:',
    '    VDDNet = odb.dbNet_create(design.getBlock(), "VDD")',
    '',
    'VDDNet.setSpecial()',
    'VDDNet.setSigType("POWER")',
    'VSSNet = design.getBlock().findNet("VSS")',
    'if VSSNet is None:',
    '    VSSNet = odb.dbNet_create(design.getBlock(), "VSS")',
    '',
    'VSSNet.setSpecial()',
    'VSSNet.setSigType("GROUND")',
    'design.getBlock().addGlobalConnect(None, ".*", "VDD", VDDNet, True)',
    'design.getBlock().addGlobalConnect(None, ".*", "VSS", VSSNet, True)',
    'design.getBlock().globalConnect()',
    'timing = Timing(design)',
    'block = design.getBlock()',
    'block.findInst("FILLER_9_11").setLevel(1, False)',
    'del block',
    'db = ord.get_db()',
    'filler_cells_prefix = "FILLCELL_.*"',
    'for lib in db.getLibs():',
    '    for master in lib.getMasters():',
    '        master_name = master.getConstName()',
    '        if re.fullmatch(filler_cells_prefix, master_name) != None:',
    '            master.setType("CORE_SPACER")',
    'del db',
    ]
    return loadDesignCommand
  elif loadDesignType != "file":
    designNameList = {"floorplan": "1_synth", "io": "2_floorplan", "mpl": "3_io", 
    "gpl": "4_mpl", "dpl": "5_gpl", "pdn": "6_dpl", "cts": "7_pdn", "filler": "8_cts",
    "grt": "9_filler", "drt": "10_grt"}

    loadDesignCommand = [
    'from openroad import Tech, Design, Timing',
    'from pathlib import Path',
    'tech = Tech()',
    'libDir = Path("../Design/nangate45/lib")',
    'lefDir = Path("../Design/nangate45/lef")',
    'designDir = Path("../Design/")',
    'libFiles = libDir.glob("*.lib")',
    'techLefFiles = lefDir.glob("*.tech.lef")',
    'lefFiles = lefDir.glob("*.lef")',
    'for libFile in libFiles:',
    '    tech.readLiberty(libFile.as_posix())',
    '',
    'for techLefFile in techLefFiles:',
    '    tech.readLef(techLefFile.as_posix())',
    '',
    'for lefFile in lefFiles:',
    '    tech.readLef(lefFile.as_posix())',
    '',
    'design = Design(tech)',
    'odbFile = designDir/str("' + designNameList[loadDesignType] + '.odb")',
    'design.readDb(odbFile.as_posix())',
    'design.evalTclString("create_clock -period 20 [get_ports clk] -name core_clock")',
    'design.evalTclString("set_propagated_clock [get_clocks {core_clock}]")',
    ]
    return loadDesignCommand
  else:
    return []

def clearQueue(q):
  while not q.empty():
    try:
      q.get_nowait()
    except queue.Empty:
      break

def generate(
  model: AutoModelForCausalLM,
  tokenizer: AutoTokenizer,
  prompt: str,
  pastKeyValues: DynamicCache,
  maxNewTokens: int = 32768,
  temperature: float = 0.2,
  topP: float = 0.7,
  returnLogits: bool = False
):
  inputIds = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

  outputIds = inputIds.clone()
  nextToken = inputIds
  pastKeyValues = pastKeyValues

  all_token_logits = []

  with torch.no_grad():
    for _ in range(maxNewTokens):
      outputs = model(
        input_ids = nextToken, 
        past_key_values = pastKeyValues,
        use_cache = True
      )
      nextTokenLogits = outputs.logits[:, -1, :]

      if returnLogits:
        all_token_logits.append(nextTokenLogits.clone())

      # Add temperature scaling
      if temperature != 1.0:
        nextTokenLogits = nextTokenLogits / temperature
      
      # Add top-p (nucleus) sampling
      if topP < 1.0:
        sortedLogits, sortedIndices = torch.sort(nextTokenLogits, descending=True)
        cumulativeProbs = torch.cumsum(torch.softmax(sortedLogits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sortedIndicesToRemove = cumulativeProbs > topP
        # Shift the indices to the right to keep first token above threshold
        sortedIndicesToRemove[..., 1:] = sortedIndicesToRemove[..., :-1].clone()
        sortedIndicesToRemove[..., 0] = 0
        
        indicesToRemove = sortedIndicesToRemove.scatter(
          -1, sortedIndices, sortedIndicesToRemove
        )
        nextTokenLogits[indicesToRemove] = -float("inf")

      # Sample from the modified logits
      probs = torch.softmax(nextTokenLogits, dim=-1)
      nextToken = torch.multinomial(probs, num_samples=1)
      nextToken = nextToken.to("cuda")

      pastKeyValues = outputs.past_key_values
      outputIds = torch.cat([outputIds, nextToken], dim=1)
      if nextToken.item() == tokenizer.eos_token_id:
        break
    if returnLogits:
      allTokenLogits = torch.stack(all_token_logits).clone().detach().to("cpu")
      outputIds = outputIds.to("cpu")
      for tensor in all_token_logits:
        del tensor
      gc.collect()
      torch.cuda.empty_cache()

  if returnLogits:
    return outputIds, allTokenLogits
  else:
    generatedText = tokenizer.decode(outputIds[:, 0:][0],
                                    skip_special_tokens = False,
                                    temperature = None)
    return generatedText


class modelUtility:
  def __init__(self, modelName: str):
    self.modelName = modelName
    if modelName == "OpenROAD-Assistant/Script_Adaptor":
      self.systemPrompt = """You are a tutor specializing in the knowledge of OpenROAD, the open-source EDA tool. You will be asked about general OpenROAD questions and OpenROAD Python API-related questions.
"""
    else:  
      self.systemPrompt = """You are a OpenROAD Python code generator. You will generate the Python code to complete the task. Follow the following guidelines:
1. Only generate Python code, do not include any other text.
2. First generate ```python before the start of the Python code.
3. Generate ``` when finished the Python code.
4. If you don't know the answer, respond with:
```python
```
"""
    if "llama" not in modelName.lower() and "script_adaptor" not in modelName.lower() and "retrained" not in modelName.lower():
      self.ragPromptTemplateWithContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here are the OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
      self.ragPromptTemplateWithoutContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
      self.ragWrongCodePromptTemplateWithContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

Here are some OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
      self.ragWrongCodePromptTemplateWithoutContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
    else:
      self.ragPromptTemplateWithContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here are the OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
      self.ragPromptTemplateWithoutContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
      self.ragWrongCodePromptTemplateWithContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

Here are some OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.
If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
      self.ragWrongCodePromptTemplateWithoutContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.
If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

  def isOpenROADAssistant(self):
    return "OpenROAD-Assistant/Script_Adaptor" == self.modelName
  
def prepareDocuments(df, descriptionColumn="Description:", api = True):
  documents = list()
  documentsDict = dict()
  for _, row in df.iterrows():
    content = ""
    if api:
      content = "OpenROAD Python API Description:" + row[descriptionColumn]
    else:
      content = "OpenROAD Code Example Description:" + row[descriptionColumn]
    if pd.notna(content):
      metadata = row.to_dict()
      if api:
        metadata["OpenROAD Python API Description:"] = metadata.pop("Description:")
      else:
        metadata["OpenROAD Code Example Description:"] = metadata.pop("Description:")
      documentsDict[content] = metadata
      documents.append(content)
  return documents, documentsDict

def answerWithRAG(
    question,
    embeddings,
    embeddingModel,
    allSplits,
    allDict
  ):
  questionEmbedding = embeddingModel.encode(question)
  scores = cos_sim(questionEmbedding, embeddings)
  npData = scores.numpy().flatten()
  topIndices = np.argsort(npData)[-10:][::-1]
  relevantDocs = list()
  for i in range(len(topIndices)):
    if npData[topIndices[i]] < 0.7:
      break
    else:
      relevantDocs.append(allSplits[topIndices[i]])
  finalDocs = list()
  for doc in relevantDocs:
    content = doc
    finalDocs.append("\n".join(f"# {key} {value}" for key, value in allDict[content].items()))

  context = ""
  if len(finalDocs) > 0:
    context += "".join([f"\n\n" + doc for i, doc in enumerate(finalDocs)])

  return context