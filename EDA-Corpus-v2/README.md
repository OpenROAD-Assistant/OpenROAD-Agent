# EDA-Corpus-v2

This directory contains an augmented version of the [EDA-Corpus dataset](https://ieeexplore.ieee.org/document/10691774) used in this paper. We have reorganized the data into separate sheets. The *prompt* sheets contain the prompts, the *code* sheets contain the corresponding correct script for each prompt, and the *wrong_message* sheets contain the incorrect script along with the OpenROAD messages introduced in this paper.

## Table of Content
  - [DB-v2.xlsx](./DB-v2.xlsx) contains the database-based OpenROAD Python prompt-script data points.
  - [Flow-v2.xlsx](./Flow-v2.xlsx) contains the single-stage and cross-stage physical design flow-based OpenROAD Python prompt-script data points.
  - [TestSet.xlsx](./TestSet.xlsx) is the test set for this paper.
  - [task_combination.txt](./task_combinations.txt) is an auxiliary file to assist in running training and testing scripts in [src](../src).

## Database and Single-Stage Flow Script Data Points

The row indices align across the prompt, code, and wrong_message sheets.

### Prompts
![tones](../Image/tones.png)

We include different tones of prompts in this version of the EDA-Corpus dataset. All prompts in [DB-v2.xlsx](./DB-v2.xlsx) and [Flow-v2.xlsx](./Flow-v2.xlsx) are paraphrased and organized in this order.

  - In the *xxx_prompt* sheets of [Flow-v2.xlsx](./Flow-v2.xlsx) and the *prompt* sheet of [DB-v2.xlsx](./DB-v2.xlsx), prompt0 to prompt5 consist of the original EDA-Corpus prompt and its paraphrased versions, arranged in the same order as shown in the above figure.

### Correct Script

In the *xxx_code* sheets of [Flow-v2.xlsx](./Flow-v2.xlsx) and the *code* sheet of [DB-v2.xlsx](./DB-v2.xlsx), each row contains the correct OpenROAD Python script corresponding to the prompts with the same row index in the *prompt* and *xxx_prompt* sheets.

### Incorrect Script and Corresponding Message from OpenROAD

We include six pairs of incorrect OpenROAD Python scripts and their corresponding OpenROAD messages in addition to the prompt-code pairs in the EDA-Corpus dataset.

  - In the *xxx_wrong_message* sheets of [Flow-v2.xlsx](./Flow-v2.xlsx) and the *wrong_messagee* sheet of [DB-v2.xlsx](./DB-v2.xlsx), each *wrong_codex* and *wrong_messagex* is a pair of incorrect OpenROAD Python script and its corresponding OpenROAD message. There are 6 incorrect script-message pairs for each prompt-script pair with the same row index.

## Cross Stage Flow Script Data Points
For cross-stage flow operations, we provide only one prompt for each script data point.