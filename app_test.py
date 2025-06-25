from tkinter import ttk
from tkinter import *
from tkinter import filedialog
import pandas as pd
import boto3
import json
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

root = Tk(screenName=None, baseName=None, className='Tk', useTk=1)
root.geometry('400x300')
root.resizable(0,0)

menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Open')

filepath = None
output_file = None

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-2")

    # Set the model ID.
model_id = "us.meta.llama4-maverick-17b-instruct-v1:0"

def select_file(event=None):
    global filepath, output_file
    filename = filedialog.askopenfilename()
    filepath = filename
    print(filepath)
    output_file = filepath[:filepath.find(".csv")] + "_results.csv"
    selected_file.config(text=filename[filename.rfind('/')+1:]+' selected')
    selected_file.place(relx=0.5, rely=0.5, anchor='center')
    process_btn.place(relx=0.5, rely=0.7, anchor='center')

def process_file():
    global progress_bar, root, results, process_btn

    print(filepath)
    df = pd.read_csv(filepath)

    tests_list = df.iloc[:, 0]

    results = []
    progress_bar["maximum"] = len(tests_list)
    process_btn.config(text="Processing...", state=DISABLED)
    upload_btn.config(state=DISABLED)
    root.update()
    process_btn.place(relx=0.5, rely=0.7, anchor='center')
    i = 0

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
                        progress_bar["value"] = i + 1
                        i += 1
                        progress_bar.place(relx=0.5, rely=0.6, anchor='center')
                        root.update()
                        break
                    else:
                        print(f"Attempt {attempt + 1}: Invalid or empty response")
                except FuturesTimeoutError:
                    print(f"Attempt {attempt + 1}: Timeout occurred")
            
            attempt += 1
            

        if attempt == max_attempts:
            print("All attempts failed, defaulting to 2")
            results.append('2')
            progress_bar["value"] = i + 1
            i += 1
            progress_bar.place(relx=0.5, rely=0.6, anchor='center')
            root.update()

    # Finalize the results.
    process_btn.config(text="Processing Complete!")
    root.update()
    df = df.iloc[:len(results)].copy()
    df["Results"] = results
    df.to_csv(output_file, index=False)

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


'''Widgets'''
upload_text = Label(root, text='Upload a .csv file')
upload_text.place(relx=0.5, rely=0.3, anchor='center')

upload_btn = Button(root, text='Upload File', width=25, command=select_file)
upload_btn.place(relx=0.5, rely=0.4, anchor='center')

selected_file = Label(root)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.place(relx=0.5, rely=0.6, anchor='center')

process_btn = Button(root, text="Process File", width=25, command=process_file)




root.mainloop()