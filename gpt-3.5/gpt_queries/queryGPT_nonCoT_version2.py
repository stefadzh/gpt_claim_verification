# This file is used for few-shot prompting
# Change the number of shots and gpt model as needed
# GPT is prompted with the prompts made in prompt_generation(), then the results are stored in a csv file
# Results are stored in folder data2/zero_shot/gpt-model-number. GPT output is stored in csv file named "newstance_combined_{num shots}shot_gpt{model}_all2_domain.csv"
# confusion matrix and accuracy/precision/recall/f1 for those results is stored in "conf_matrix_newstance_combined_{num shots}shot_gpt{model}_all2_domain.csv"


import time, os, openai, random
import pandas as pd

#api key should be stored as environment variable for security
openai.api_key = ""

# openai.api_key = os.getenv('')
file_path   = 'C:\\Users\\dzham\\Desktop\\ODUWuProject\\GPTredo\\msvec_v2\\gpt-3.5\\data2\\ground_truth_datasets\\domain\\'
#file_path2 = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1.csv'
export_path = 'C:\\Users\\dzham\\Desktop\\ODUWuProject\\GPTredo\\msvec_v2\\gpt-3.5\\data2\\zero_shot\\gpt4o\\'

num_shot = 3 #change for number of shots

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
        if file.endswith("all2_domain.csv") and file != "out_domain.csv": #done: change back to go through all files
            domain = file
            df = pd.read_csv(file_path + domain)
            #set the first 15 rows as the training data
            df_training = df.head(15)
            #set the rest of the rows as the testing data
            df_testing = df.iloc[15:] #did: change this to exclude the first 15 so that we always test on the same set 
            training_sample = df_training.head(num_shot * 3) #num-shot * 3claims (sup, refute, nei)
            print(training_sample)
            prompts = []    
            #generate prompts for that domain

            for index, test_sample in df_testing.iterrows(): #done: edit so that it's if training_sample['consensus'] == "SUPPORTS" , etc
                

                prompt = "Read the following examples and answer the question at the end:\n\n"
                
                for _, train_sample in training_sample.iterrows(): 
                            if ("Supports") in train_sample['stance']:
                                # print(train_sample)
                                ans = "SUPPORT"
                            elif train_sample['stance'] == "Refutes":
                                ans = "REFUTES"
                            elif train_sample['stance'] == "NEI":
                                ans = "NOT ENOUGH INFORMATION"
                            # print(ans)
                            #ans = "SUPPORT" if train_sample['true/false'] else "CONTRADICT"
                            prompt += (f"Claim: {train_sample['claim']}\n"
                                    f"Abstract: {train_sample['published_paper_abstract']}\n"
                                    f"Question: Does the abstract of the scientific paper support or refute the claim?\n\n"
                                    f"Answer: {ans}\n\n")
               
                prompt += (f"Read the claim and abstract below, then answer the question at the end:\n\n"
                            f"Claim: {test_sample['claim']}\n"
                            f"Abstract: {test_sample['published_paper_abstract']}\n"
                            f"Question: Does the abstract of the scientific paper support or refute the claim? Answer with SUPPORTS or REFUTES or NOT ENOUGH INFORMATION.\n\n")
                prompts.append(prompt)
                
            # print(f"Prompts for {domain}:")
            # print(prompts[5])
            #print the training sample
            # print(f"Training sample: {training_sample['claim']}")
            # print()
            # for p in prompts:
            #     print(p)
            #     print("\n---\n")
            
            shot_query(domain,prompts)
            result_df = pd.read_csv(export_path + domain)
            domain_df = pd.read_csv(file_path + domain, usecols=['id', 'claim', 'domain', 'stance', 'true/false'])
            result_df['GPT_Response_1'] = result_df['GPT_Response_1'].apply(parse_response)
            result_df['id'] = domain_df['id'].iloc[15:].reset_index(drop=True)
            result_df['claim'] = domain_df['claim'].iloc[15:].reset_index(drop=True)
            result_df['stance'] = domain_df['stance'].iloc[15:].reset_index(drop=True)
            result_df['true/false'] = domain_df['true/false'].iloc[15:].reset_index(drop=True)
            combined_df = result_df
            combined_export_path = export_path + "newstance_combined_3shot_gpt4o_" + domain
            combined_df.to_csv(combined_export_path, index=False)
            # eval(combined_export_path) #comment out, dont run validation (do manual)
            print("File combined and exported: " + combined_export_path) 
            print("Done with domain: " + domain) 
            #END MOVE TO NEXT DOMAIN
    

def shot_query(domain,prompts):
    promptsDF = pd.DataFrame(prompts, columns=['prompts'])
    responses = []
    requests = 0
    col_name = 'GPT_Response_1'
    for index,prompt in enumerate(prompts):
        if requests>0 and requests%50 == 0:
            print("Sleeping for 20 seconds...") #probably not needed for 3.5
            time.sleep(20)
            print("Continuing...currently on request " + str(requests))

        requests += 1

        responses.append(get_completion(prompt))
        promptsDF.at[index, col_name] = responses[index] # Name of column in output file

    # Export to CSV(creates CSV with additional column(s))
    promptsDF.to_csv(export_path + domain)
    print("Done with export " + domain + "!\n")
    
def parse_response(response):
    # if pd.isna(response):
    #     return None
    # response = response.strip()
    # if "SUPPORT" in response:
    #     return "SUPPORT"
    # elif "REFUTE" in response:
    #     return "REFUTE"
    # elif "NOT ENOUGH INFORMATION" in response:
    #     return "NEI"
    # else:
    #     return response
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
        if gt and label == 'SUPPORT':
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
#Iterate trough 



