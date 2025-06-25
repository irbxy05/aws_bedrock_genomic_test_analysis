# Genetic Test
## Using AWS Bedrock, categorize a list of medical tests and mark them as genetic tests, non-genetic tests, or needs more evaluation.

- app_test.py - A standalone GUI application for processing csv files through the LLM
- compare_data.py - report statistics on output from genetic_test_llama.py, including accuracy, precision, recall, and f1 score
- genetic_test_llama.py - processes csv file through the LLM, printing result output in real time to the terminal
