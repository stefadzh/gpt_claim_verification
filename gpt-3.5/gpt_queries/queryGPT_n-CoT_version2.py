# This file is used for n-chain of thought prompting
# Change the number of shots and the gpt model as needed
# GPT is prompted with the prompts made in prompt_generation(), then the results are stored in a csv file
# Results are stored in data2/num-shots-CoT/gpt-model-number. GPT output is stored in file named "combined_{numshots}cot_gpt{model}_all2_domain.csv"
# confusion matrix and accuracy/precision/recall/f1 for those results is stored in "conf_mat_combined_{numshots}cot_gpt{model}_all2_domain.csv"

import time, os, openai, random
import pandas as pd

#api key can be stored as environment variable for safer security or pasted into the blank (do not push to public repos)
openai.api_key = ""

#change number of shots to 1, 2, or 3
num_shots = 3

#change file paths 
file_path   = 'C:\\Users\\dzham\\Desktop\\ODUWuProject\\GPTredo\\msvec_v2\\gpt-3.5\\data2\\ground_truth_datasets\\domain'

#change export path to match the GPT version
export_path = f'C:\\Users\\dzham\\Desktop\\ODUWuProject\\GPTredo\\msvec_v2\\gpt-3.5\\data2\\{num_shots}-CoT\\gpt4o\\'

#change GPT version as needed
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
            domain = file
            df = pd.read_csv(file_path +'\\'+ domain)

            # reserve the first 15 rows as potential training data
            potential_training_data = df.head(15)

            # Select fixed number of training examples based on num_shots
            # training data is number of shots * 3 because each set of training data has 1 support, 1 refute, and 1 NEI
            selected_training = potential_training_data.head(num_shots * 3)
            # Set the rest of the rows as the testing data
            df_testing = df.iloc[15:]


            prompts = []
            # Generate prompts for that domain
            for index, test_sample in df_testing.iterrows():
                prompt = "Read the following example(s) and answer the question at the end:\n\n"
                
                # Add each selected training example to the prompt
                for _, train_sample in selected_training.iterrows():
                    #ans = "SUPPORT" if train_sample['true/false'] else "CONTRADICT"
                    prompt += (f"Claim: {train_sample['claim']}\n"
                               f"Abstract: {train_sample['published_paper_abstract']}\n"
                               f"Question: Does the abstract of the scientific paper support the claim, refute the claim, or is there not enough information??\n\n"
                               f"Answer: {train_sample['rationale']}\n\n")
                
                # Add the test question
                prompt += (f"Read the claim and abstract and answer the question by mimicking the process previously outlined.\n\n"
                           f"Claim: {test_sample['claim']}\n"
                           f"Abstract: {test_sample['published_paper_abstract']}\n"
                           f"Question: Does the abstract of the scientific paper support the claim, refute the claim, or is there not enough information?\n\n") #TODO: change to support or refute the claim, answer with SUPPORTS, REFUTES or NOT ENOUGH INFORMATION
                prompts.append(prompt)

            #change domain to be just the first 5 letters of the filename:
            # domain2 = domain[0:5]
            print(f"Prompts for {domain}:")
            #print the training sample
            print(f"Training sample(s):" + str(selected_training['claim'].tolist()) + "\n")
            print()
            # for p in prompts:
            #    print(p)
            #    print("\n---\n")
            # write prompts to text file called output.txt

            # print(export_path + domain)

            directory = os.path.dirname(export_path + domain)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(directory + "\\_prompts.txt", "w", encoding = 'utf-8') as output:
                output.write(f"Prompts for {domain}:\n\n")
                output.write(f"Training sample(s):" + str(selected_training['claim'].tolist()) + "\n\n")
                for p in prompts:
                    output.write(p + "\n---\n")
            
            print("resultdf is " + export_path + domain)
            # print("filepath is" + file_path + domain)

            shot_query(domain,prompts)
            result_df = pd.read_csv(export_path + domain)
            domain_df = pd.read_csv(file_path + '\\' + domain, usecols=['claim', 'stance', 'id', 'true/false'])
            result_df['GPT_Response_1'] = result_df['GPT_Response_1'].apply(parse_response)
            result_df['id'] = domain_df['id'].iloc[15:].reset_index(drop=True)
            result_df['claim'] = domain_df['claim'].iloc[15:].reset_index(drop=True)
            result_df['stance'] = domain_df['stance'].iloc[15:].reset_index(drop=True)
            result_df['true/false'] = domain_df['true/false'].iloc[15:].reset_index(drop=True)
            combined_df = result_df

            #change number of CoT and gpt version in file name to reflect the trial
            combined_export_path = export_path + "combined_3cot_gpt4o_" + domain #CHANGE NUMBER OF COTS

            combined_df.to_csv(combined_export_path, index=False)
            # eval(combined_export_path) # commented out, not used
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
        return None

    response = response.strip()
    # Find where the conclusion starts in the response, following 'Step 4:'
    conclusion_start = response.find("Step 4:") 
    if conclusion_start != -1:
        conclusion_text = response[conclusion_start:]
        return conclusion_text
    
        #the following approach can be used if made more robust, but GPT sometimes didn't follow the format outlined so a simple word search is not accurate enough
        # Check for keywords indicating support or contradiction
        # if "supports the claim" in conclusion_text or "Conclusion: The abstract supports the claim" in conclusion_text:
        #     return "SUPPORT"
        # elif "refutes the claim" in conclusion_text or "Conclusion: The abstract refutes the claim" in conclusion_text:
        #     return "CONTRADICT"
    #-------------------------------------------------------------------------------------------------------------------
    # support_count = response.lower().count("support") + response.lower().count("supports")
    # refute_count = response.lower().count("refute") + response.lower().count("refutes")
    # if support_count > refute_count:
    #     return "SUPPORT"
    # elif refute_count > support_count:
    #     return "CONTRADICT"
    
    return None
    

#function below NOT used due to variability of GPT responses. Can be adjusted to be used for future trials
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

#function below NOT used due to variability of GPT responses. Can be adjusted to be used for future trials
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
            print(f"{id} is a trueP, gt = {gt} and label = {label}")
        elif not gt and label == 'CONTRADICT': 
            trueN += 1
            print(f"{id} is a trueN, gt = {gt} and label = {label}")
        elif not gt  and (label == 'SUPPORT' or label == 'NEI'):
            falseP += 1
            print(f"{id} is a falseP, gt = {gt} and label = {label}")
        elif gt  and (label == 'CONTRADICT'or label == 'NEI'):
            falseN += 1
            print(f"{id} is a falseN, gt = {gt} and label = {label}")
        print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")

    print()
    print('Support Class Stance:\n')
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results
    
    
#run
prompt_generation()








