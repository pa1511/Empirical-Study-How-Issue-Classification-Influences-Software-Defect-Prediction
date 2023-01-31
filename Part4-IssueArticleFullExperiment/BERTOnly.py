#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import os
import jsonpickle
from ast import literal_eval as make_tuple


# In[ ]:


import re


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


from sklearn.model_selection import train_test_split
#
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


# In[ ]:


from datasets import Dataset
#
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
#
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from transformers import Trainer
import evaluate


# In[ ]:


from sklearn.manifold import TSNE
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


# In[ ]:


from tabulate import tabulate


# In[ ]:


import warnings
warnings.simplefilter("ignore")


# ## Constants

# In[ ]:


DATA_FOLDER = "./../data/"
#
FILE_OF_INTEREST = "files_of_interest.json"
FILE_OF_INTEREST_SOURCE = "files_of_interest_source_lookup.json"
LABELED_FILE_KEY = "labeled_issues_of_interest_"
EMBEDDING_FILE_KEY = "file_of_interest_embedding_lookup_"
FILE_OF_INTEREST_WITH_EMBEDDINGS = "files_of_interest_with_embeddings.json"


# ## Utils

# In[ ]:


def find_file(commit, file_name):
    for file in commit["files"]:
        if file["name"] == file_name:
            return file
    return None


# ## Load data

# In[ ]:


# Load files of interest 
with open(os.path.join(DATA_FOLDER, FILE_OF_INTEREST), "r") as f_in:
    for line in f_in:
        file_of_interest_data = jsonpickle.decode(line)


# In[ ]:


REPO_TO_ID = {}


# In[ ]:


# Load labeled issues
labeled_issues_of_interest_data = {}
for file in os.listdir(DATA_FOLDER):
    if LABELED_FILE_KEY in file:
        repoId = file.replace(LABELED_FILE_KEY, "").replace(".json", "")
        with open(os.path.join(DATA_FOLDER, file), "r") as f_in:
            for line in f_in:
                repo_labeled_issues = jsonpickle.decode(line)
        for repo in repo_labeled_issues:
            REPO_TO_ID[repo] = repoId
        labeled_issues_of_interest_data.update(repo_labeled_issues)
#===
for repo in labeled_issues_of_interest_data:
    if "list" in str(type(labeled_issues_of_interest_data[repo])):
        adjusted_labeled_issues_of_interest = {}
        for issue in labeled_issues_of_interest_data[repo]:
            if issue is not None:
                adjusted_labeled_issues_of_interest[str(issue["number"])] = issue
        labeled_issues_of_interest_data[repo] = adjusted_labeled_issues_of_interest


# In[ ]:


# Load files of interest with embeddings
with open(os.path.join(DATA_FOLDER, FILE_OF_INTEREST_WITH_EMBEDDINGS), "r") as f_in:
    for line in f_in:
        file_states = jsonpickle.decode(line)


# In[ ]:


adjusted_file_states = {}
for entry in file_states:
    t = make_tuple(entry)
    adjusted_file_states[t] = file_states[entry]
#
file_states = adjusted_file_states


# ## Construct datasets

# In[ ]:


dataset = {}
for repo in file_of_interest_data:
    if repo not in labeled_issues_of_interest_data:
        continue
    dataset[repo] = {}
    for file_name in file_of_interest_data[repo]:
        commits = file_of_interest_data[repo][file_name]
        commits = sorted(commits, key=lambda c: c["date"])
        #
        all_refs_cnt = 0
        for commit in commits:
            file = find_file(commit, file_name)
            if file is None:
                continue
            all_refs_cnt = all_refs_cnt + len(commit["refs"])        
        #
        for commit in commits:
            file = find_file(commit, file_name)
            if file is None:
                continue
            #
            key = (repo, file["sha"], commit["sha"])
            has_source = key in file_states and "source" in file_states[key]
            has_embedding = key in file_states and "embedding" in file_states[key] and file_states[key]["embedding"] is not None 
            if not has_source or not has_embedding:
                continue
            #
            bug_cnt = 0
            undefined_cnt = 0
            for ref in commit["refs"]:
                if ref in labeled_issues_of_interest_data[repo] and labeled_issues_of_interest_data[repo][ref]:
                    if "type" in labeled_issues_of_interest_data[repo][ref]:
                        if labeled_issues_of_interest_data[repo][ref]["type"] == "Bug":
                            bug_cnt = bug_cnt + 1
                    else:
                        undefined_cnt = undefined_cnt + 1
                else:
                        undefined_cnt = undefined_cnt + 1
            #
            if has_source and has_embedding and (bug_cnt>0 or undefined_cnt==0):
                if file_name not in dataset[repo]:
                    dataset[repo][file_name] = []
                #
                source = file_states[key]["source"]
                lines_of_code = len([line for line in source.split("\n") if len(line.strip()) > 0 ])
                embedding = file_states[key]["embedding"]
                bug = 1 if bug_cnt > 0 else 0
                #
                dataset[repo][file_name].append((source, lines_of_code, embedding, len(commits), all_refs_cnt, len(commit["refs"]), commit["refs"], bug))   


# In[ ]:


print_data = []
for repo in dataset:
    cnt = 0
    bug_cnt = 0
    for file in dataset[repo]:
        cnt = cnt + 1
        for version in dataset[repo][file]:
            source, lines_of_code, embedding, commit_cnt, all_refs_cnt, refs_cnt, refs, bug = version
            #
            if bug > 0:
                bug_cnt = bug_cnt + 1
                break
            #
        r = bug_cnt/cnt
        random_f1 = 2*r/(r+1)
    print_data.append([repo, bug_cnt, cnt, f"{round(100*bug_cnt/cnt, 2)}%", f"{round(100*random_f1, 2)}%"])
print(tabulate(print_data, headers=["Repo", "BugCnt", "Cnt", "Share", "MaxF1"]))


# ## Classification experiment

# In[ ]:


def get_prop_from_version(prop, version):
    source, lines_of_code, embedding, commit_cnt, all_refs_cnt, refs_cnt, refs, bug = version
    if prop == "source":
        return source
    if prop == "loc":
        return lines_of_code
    if prop == "embedding":
        return embedding
    if prop == "commit_cnt":
        return commit_cnt
    if prop == "all_refs_cnt":
        return all_refs_cnt
    if prop == "refs_cnt":
        return refs_cnt
    if prop == "refs":
        return refs
    if prop == "bug":
        return bug
    return None


# In[ ]:


def calc_diff_vec(embeddings):
    if len(embeddings) < 2:
        return np.zeros(embeddings[0].shape)
    else:
        diffs = []
        for i in range(len(embeddings)-1):
            diff = embeddings[i] - embeddings[i+1]
            diffs.append(diff)
        return np.asarray(sum(diffs))


# In[ ]:


def calc_x(versions):
    locs = [get_prop_from_version("loc", v) for v in versions]
    avg_loc = np.asarray([sum(locs)/len(locs)])
    #
    embeddings = [get_prop_from_version("embedding", v) for v in versions]
    x_mean = np.asarray(sum(embeddings) / len(versions))
    x_diff = calc_diff_vec(embeddings)
    #
    commit_cnt = np.asarray([get_prop_from_version("commit_cnt", versions[0])])
    #
    all_refs_cnt = np.asarray([get_prop_from_version("all_refs_cnt", versions[0])])
    #
    ref_cnts = [get_prop_from_version("refs_cnt", v) for v in versions]
    avg_ref_cnt = np.asarray([sum(ref_cnts) / len(ref_cnts)])
    #
    x = np.concatenate((avg_loc, x_mean, x_diff, commit_cnt, all_refs_cnt, avg_ref_cnt))
    return x


# In[ ]:


def calc_y(versions):
    bug_cnt = sum([get_prop_from_version("bug", v) for v in versions])
    return 1 if bug_cnt > 0 else 0


# In[ ]:


MODELS = [("LogisticRegression", lambda: LogisticRegression()), 
          ("KNeighborsClassifier", lambda: KNeighborsClassifier(1)), 
          ("GaussianNB", lambda: GaussianNB()), 
          ("DecisionTreeClassifier", lambda: DecisionTreeClassifier())]


# In[ ]:


columns=["Model", "Precision", "Recall", "F1", "MCC"]


# ### Experiment NLP - BERT

# In[ ]:


def calc_y_bert(versions, lookup):
    for version in versions:
        refs = get_prop_from_version("refs", version)
        for ref in refs:
            ref = int(ref)
            if ref in lookup:
                if lookup[ref] > 0:
                    return 1
    return 0


# In[ ]:


for repo in labeled_issues_of_interest_data:
    for ref in labeled_issues_of_interest_data[repo]:
        issue = labeled_issues_of_interest_data[repo][ref]
        if issue is None or "type" not in issue or issue["type"] is None:
            continue
        issue_title = issue["title"]
        issue_description = issue["body"]
        issue_title = "" if issue_title is None else issue_title
        issue_description = "" if issue_description is None else issue_description
        #
        CLEANR = re.compile('<.*?>') 
        text = re.sub(CLEANR, ' ', issue_description)
        text = issue_title + " " + text
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = re.sub(' +', ' ', text)
        text = re.sub(' http.*? ', ' [link] ', text)
        #
        issue["text"] = text


# In[ ]:


print_data = []
nlp_datasets = {}
for repo in dataset:
    if repo not in labeled_issues_of_interest_data:
        continue
    #
    nlp_datasets[repo] = {
        "train": [],
        "test": []
    }
    #
    train_bug_cnt = 0
    for other_repo in dataset:
        if repo == other_repo or other_repo not in labeled_issues_of_interest_data:
            continue
        for ref in labeled_issues_of_interest_data[other_repo]:
            issue = labeled_issues_of_interest_data[other_repo][ref]
            if issue is None or "type" not in issue or issue["type"] is None or "text" not in issue:
                continue
            label = 1 if issue["type"] == 'Bug' else 0
            train_bug_cnt = train_bug_cnt + label
            nlp_datasets[repo]["train"].append({"text": issue["text"], "label": label})
    #
    test_bug_cnt = 0
    for ref in labeled_issues_of_interest_data[repo]:
        issue = labeled_issues_of_interest_data[repo][ref]
        if issue is None or "type" not in issue or issue["type"] is None or "text" not in issue:
            continue
        label = 1 if issue["type"] == 'Bug' else 0
        test_bug_cnt = test_bug_cnt  + label
        nlp_datasets[repo]["test"].append({"issueId": issue["number"], "text": issue["text"], "label": label})
    #
    print_data.append([repo, train_bug_cnt, len(nlp_datasets[repo]["train"]), test_bug_cnt, len(nlp_datasets[repo]["test"])])
print(tabulate(print_data, headers=["Repo", "TrainBugCnt", "TrainCnt", "TestBugCnt", "TestCnt"]))


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("roberta-base")


# In[ ]:


def tokenize(entry):
    return tokenizer(entry["text"], padding="max_length", truncation=True)


# In[ ]:


metric = evaluate.load("f1")
#
def compute_metrics(eval_pred):
    o, y = eval_pred
    yp = np.argmax(o, axis=-1)
    #
    return metric.compute(predictions=yp, references=y)


# In[ ]:


bert_experimental_results = {}
for repo in dataset:
    bert_experimental_results[repo] = []
#
bert_model_results = {}
bert_labels = {}


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:


TRAIN_SIZE = 0.8
REP_CNT = 30
for repo in dataset:
    print(repo)
    #===============================
    # Train BERT
    train_nlp = Dataset.from_pandas(pd.DataFrame(nlp_datasets[repo]["train"]))
    #
    permutation = torch.randperm(len(train_nlp)).tolist()
    train_cnt = int(len(train_nlp) * TRAIN_SIZE)
    train_indices = permutation[:train_cnt]
    val_indices = permutation[train_cnt:]
    val_nlp = train_nlp.select(val_indices)
    train_nlp = train_nlp.select(train_indices)
    #
    train_nlp = train_nlp.map(tokenize, batched=True)
    val_nlp = val_nlp.map(tokenize, batched=True)
    #
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    if device == "cuda":
        model.cuda()
    #
    training_args = TrainingArguments(
                         overwrite_output_dir=True,
                         output_dir=f"roberta-issue-classifier-{REPO_TO_ID[repo]}",
                         evaluation_strategy="epoch",
                         learning_rate=2e-5,
                         logging_strategy='epoch',
                         per_device_train_batch_size=4,
                         per_device_eval_batch_size=4,
                         save_total_limit=3,
                         num_train_epochs=6, 
                         gradient_accumulation_steps=4,
                         gradient_checkpointing=True,
                         weight_decay=1e-3,
                         save_strategy='epoch',
                         load_best_model_at_end=True)
    #
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_nlp,
                eval_dataset=val_nlp,
                compute_metrics=compute_metrics)
    #
    trainer.train()
    #
    model.save_pretrained(f"roberta-issue-classifier-{REPO_TO_ID[repo]}/evaluated_model")
    #
    test_nlp = Dataset.from_pandas(pd.DataFrame(nlp_datasets[repo]["test"]))
    test_nlp = test_nlp.map(tokenize, batched=True)
    #
    copy_test_nlp = test_nlp.select([i for i in range(len(test_nlp))])
    copy_test_nlp = copy_test_nlp.remove_columns(["text"]).rename_column("label", "labels")
    copy_test_nlp.set_format("torch")
    #
    del train_nlp
    del val_nlp
    if device == "cuda":
        torch.cuda.empty_cache()
    #
    eval_dataloader = DataLoader(copy_test_nlp, batch_size=16)
    #
    model.eval()
    #
    all_labels = []
    all_preds = []
    all_issueId = []
    for batch in eval_dataloader:
        all_labels.append(batch['labels'].detach().cpu())
        all_issueId.append(batch['issueId'].detach().cpu())
        del batch["issueId"]
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            hs = model(**batch, output_hidden_states=True)
            logits = hs.logits
            predictions = torch.argmax(logits, dim=-1)
            last_hiddens = hs.hidden_states[-1][:,0,:]
            all_preds.append(predictions)
    #
    all_labels = torch.cat(all_labels, 0).detach().cpu().numpy()
    all_preds = torch.cat(all_preds, 0).detach().cpu().numpy()                          
    all_issueId = torch.cat(all_issueId, 0).detach().cpu().numpy()
    #
    bert_precision = precision_score(all_labels, all_preds)
    bert_recall = recall_score(all_labels, all_preds)
    bert_f1 = f1_score(all_labels, all_preds)
    bert_mcc = matthews_corrcoef(all_labels, all_preds)   
    #
    bert_model_results[repo] = [bert_precision, bert_recall, bert_f1, bert_mcc]
    #
    issue_lbl_lookup = {}
    for i in range(len(test_nlp)):
        issue_bert_lbl = all_preds[i]
        issue_id = all_issueId[i]
        issue_lbl_lookup[issue_id] = issue_bert_lbl
    #===============================
    X = []
    YT = []
    YH = []
    for file in dataset[repo]:
        versions = dataset[repo][file]
        x = calc_x(versions)
        yt = calc_y(versions)
        yh = calc_y_bert(versions, issue_lbl_lookup)
        #
        X.append(x)
        YT.append(yt)
        YH.append(yh)
    X = np.asarray(X)
    y = np.asarray(YT)
    yh = np.asarray(YH)
    #======
    bert_labels[repo] = {
        "Y": YT,
        "YE": YH
    }
    #======
    for rep in range(REP_CNT):
        if (rep+1) % 5 == 0:
            print(f"\t {rep+1}/{REP_CNT}")
        #==
        X_train, X_test, y_train, _, _, y_test  = train_test_split(X, yh, y, test_size=0.2)
        #
        for model_name, model_provider in MODELS:
            classifier = model_provider()
            #
            classifier.fit(X_train, y_train)
            #
            yp = classifier.predict(X_test)
            #
            classifier_precision = precision_score(y_test, yp)
            classifier_recall = recall_score(y_test, yp)
            classifier_f1 = f1_score(y_test, yp)
            classifier_mcc = matthews_corrcoef(y_test, yp)
            #
            bert_experimental_results[repo].append([model_name, classifier_precision, classifier_recall, classifier_f1, classifier_mcc])


# In[ ]:


print_data = []
for repo in bert_model_results:
    print_data.append([repo]+bert_model_results[repo])
print(tabulate(print_data, headers=["Repo", "Precision", "Recall", "F1", "MCC"]))


# In[ ]:


for repo in bert_experimental_results:
    df = pd.DataFrame(bert_experimental_results[repo], columns=columns)
    print(repo)
    print(df.groupby(["Model"]).mean())
    #boxplot = df.boxplot(column=columns) 
    #plt.show()
    print()

