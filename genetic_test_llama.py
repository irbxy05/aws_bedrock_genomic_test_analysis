import boto3
import json
import pandas as pd
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-2")

# Set the model ID.
model_id = "us.meta.llama4-maverick-17b-instruct-v1:0"

# Load the CSV file.
csv_file = "TestsList_modified.csv"
output_file = "llama_TestsList_with_results.csv"

df = pd.read_csv(csv_file)
tests_list = df["test_name"].tolist()

results = []

# Function to invoke the model.
def get_model_response(prompt: str):
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 3,
        "temperature": 0.1,
    }

    request = json.dumps(native_request)

    try:
        response = client.invoke_model(modelId=model_id, body=request)
        model_response = json.loads(response["body"].read())
        response_text = model_response["generation"]
        response_text = response_text[0:response_text.find("<")]
        return response_text.strip()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None

# Run through the list of tests
for test in tests_list:
    print(test)

    text = (
        f"You are a medical genetics expert specializing in laboratory diagnostics. "
        f"Your task is to classify genetic tests based on their name.\n\n"
        f"Classify the '{test} Test' using the following rules:\n"
        f"- Output 1 if it is a clearly defined, non-cancer, molecular genetic test that detects inherited or congenital genetic conditions "
        f"(e.g., using DNA sequencing, genotyping, chromosomal microarrays, karyotyping, or deletion/duplication analysis). These tests typically assess variation in DNA, RNA, or chromosome structure.\n"
        f"- Output 0 if it relates to cancer, tumors, infectious diseases, drug sensitivity, or if it involves enzyme activity, protein levels, metabolite panels, or other biomarkers that do not directly assess inherited genetic variation.\n"
        f"- Output 2 if the test is not clearly defined, the name is incoherent or nonsensical, or it’s unclear what the test does.\n"
        f"Do not explain your reasoning. Respond only with one of: 0, 1, or 2."
    )

    attempt = 0
    max_attempts = 5
    result = None

    while attempt < max_attempts:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_model_response, text)
            try:
                result = future.result(timeout=20).strip()
                if result in ['0', '1', '2']:
                    print(result)
                    results.append(result)
                    break
                else:
                    print(f"Attempt {attempt + 1}: Invalid or empty response")
            except FuturesTimeoutError:
                print(f"Attempt {attempt + 1}: Timeout occurred")
        
        attempt += 1

    if attempt == max_attempts:
        print("All attempts failed, defaulting to 2")
        results.append('2')

# Finalize the results.
df = df.iloc[:len(results)].copy()
df["Results"] = results
df.to_csv(output_file, index=False)





# Tweak prompt to ask if the test is a non cancer molecular genetic test

# Is the test genetic? If yes- is it related to cancer?
