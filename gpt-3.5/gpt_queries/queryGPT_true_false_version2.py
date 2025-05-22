# This file is used for 0-shot true/false querying (only claims are used, no abstracts)
# Change the gpt model as needed
# GPT is prompted with the prompts made in prompt_generation(), then the results are stored in a csv file
# Results are stored in folder data2/true_false/{gpt-model-number}. GPT output is stored in csv file named "combined_tf_gpt{model}_all2_domain.csv"
# confusion matrix and accuracy/precision/recall/f1 for those results is stored in "combined_tf_gpt{model}_all2_domain_conf_matrix.csv"

import time, os, openai, random
import pandas as pd

#api key should be stored as environment variable for security
openai.api_key = ""

file_path   = 'C:\\Users\\dzham\\Desktop\\ODUWuProject\\GPTredo\\msvec_v2\\gpt-3.5\\data2\\ground_truth_datasets\\domain\\'

#change to correct gpt version
export_path = 'C:\\Users\\dzham\\Desktop\\ODUWuProject\\GPTredo\\msvec_v2\\gpt-3.5\\data2\\true_false\\gpt4o\\'

#change gpt version (model) if needed
def get_completion(prompt, model="gpt-4o"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def prompt_generation():
    #Given a folder of CSVs, with each CSV being a unique domain, read each CSV and generate prompts for each domain
    for file in os.listdir(file_path):
        if file.endswith("all2_domain.csv") and file != "out_domain.csv":
            print("Starting prompt generation...")
            
            # file = "claim_level_ground_truth-v1.csv"
            domain = file
            df = pd.read_csv(file_path + domain)
            df_testing = df
            
            prompts = []    
            #generate prompts for that domain
            for index, test_sample in df_testing.iterrows():

                prompt = (
                          f"Read the claim below, then answer the question at the end:\n\n"
                          f"Claim: {test_sample['claim']}\n"
                          f"Question: Is this claim true or false? Answer with TRUE or FALSE.\n\n") 
                prompts.append(prompt)
            print(f"Prompts for {domain}:")

            #print the training sample
            #print(f"Training sample: {training_sample['claim']}")
            #print()
            #for p in prompts:
                #print(p)
                #print("\n---\n")
            
            shot_query(domain,prompts)
            result_df = pd.read_csv(export_path + domain)
            domain_df = pd.read_csv(file_path + domain, usecols=['id', 'claim', 'true/false'])
            result_df['GPT_Response_1'] = result_df['GPT_Response_1'].apply(parse_response)
            result_df['id'] = domain_df['id'].reset_index(drop=True)
            result_df['claim'] = domain_df['claim'].reset_index(drop=True)
            result_df['true/false'] = domain_df['true/false'].reset_index(drop=True)
            combined_df = result_df

            #change export path to reflect gpt version used
            combined_export_path = export_path + "combined_tf_gpt4o_" + domain
            combined_df.to_csv(combined_export_path, index=False)
            # eval(combined_export_path) #do not run eval
            print("File combined and exported: " + combined_export_path)
            print("Done with domain: " + domain)
            #END MOVE TO NEXT DOMAIN
            print("Finished processing prompts.")
    

def shot_query(domain,prompts):
    promptsDF = pd.DataFrame(prompts, columns=['prompts'])
    responses = []
    requests = 0
    col_name = 'GPT_Response_1'
    for index,prompt in enumerate(prompts):
        if requests>0 and requests%50 == 0:
            print("Sleeping for 20 seconds...") 
            time.sleep(20)
            print("Continuing...currently on request " + str(requests))

        requests += 1

        responses.append(get_completion(prompt))
        promptsDF.at[index, col_name] = responses[index] # Name of column in output file

    # Export to CSV(creates CSV with additional column(s))
    promptsDF.to_csv(export_path + domain)
    print("Done with export " + domain + "!\n")
    
def parse_response(response):
    if pd.isna(response):
        print("error: not a response (line 114)")
        return None
    # response = response.strip()
    # if "SUPPORT" in response:
    #     return "SUPPORT"
    # elif "CONTRADICT" in response:
    #     return "CONTRADICT"
    # return None  
    return response

#not used
def calculate_metrics(trueP, trueN, falseP, falseN):
    total = trueP + trueN + falseP + falseN
    accuracy = (trueP + trueN) / total
    precision = trueP / (trueP + falseP)  
    recall = trueP / (trueP + falseN)  
    f1 = 2 * ((precision * recall) / (precision + recall)) 

    print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")
    print(f"accuracy = {format(accuracy*100, '.2f')}% \nprecision = {format(precision*100,'.2f')}%\nrecall = {format(recall*100,'.2f')}%\nf1 = {format(f1*100,'.2f')}%")
    results = {'accuracy': format(accuracy*100, '.2f'), 'precision':format(precision*100,'.2f'), 'recall': format(recall*100,'.2f'), 'f1':format(f1*100,'.2f')}
    return results

#not used
def eval(file_path):
    trueP = falseP = trueN = falseN = total = 0
    df = pd.read_csv(file_path)

    ground_truth = df['true/false'].astype(bool).tolist()
    labels = df['GPT_Response_1'].tolist()
    #ids = df['id'].astype(int).tolist()

    # Loop through both lists and calculate the counts
    for gt, label in zip(ground_truth, labels):
        total += 1
        if gt and label == 'SUPPORT': #todo: add NEI as valid gpt response
            trueP += 1
            #print(f"{id} is a trueP, gt = {gt} and label = {label}")
        elif not gt and label == 'CONTRADICT': 
            trueN += 1
            #print(f"{id} is a trueN, gt = {gt} and label = {label}")
        elif not gt  and (label == 'SUPPORT' or label == 'NEI'):
            falseP += 1
            #print(f"{id} is a falseP, gt = {gt} and label = {label}")
        elif gt  and (label == 'CONTRADICT'or label == 'NEI'):
            falseN += 1
            #print(f"{id} is a falseN, gt = {gt} and label = {label}")
        print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")

    print()
    print('Support Class Stance:\n')
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results
    
    
prompt_generation()