# File that implements flask server
import os
import re
import flask
import pickle
import json
import time
import random
import pandas as pd
import numpy as np
import pathlib
import unicodedata
from sentence_transformers import SentenceTransformer

# Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Load the needed files
all_keywords_data = pd.read_parquet(os.path.join(model_path, "keywords_latest.parquet"))

final_mapping_for_api = all_keywords_data[['keywords','keyword_id']].drop_duplicates(subset=['keywords'])\
    .set_index('keywords').to_dict()['keyword_id']

print("Loaded keywords file")

# Load the tokenizer and embedding model
emb_model = SentenceTransformer('baai/BGE-M3')

print("Embedding model initialized")

def name_to_keep_ind(groups):
    """
    Function to determine if a text should be kept or not.

    Input:
    groups: list of character groups

    Output:
    0: if text should be not used
    1: if text should be used
    """
    # Groups of characters that do not perform well
    groups_to_skip = ['HIRAGANA', 'CJK', 'KATAKANA','ARABIC', 'HANGUL', 'THAI','DEVANAGARI','BENGALI',
                      'THAANA','GUJARATI','CYRILLIC']
    
    if any(x in groups_to_skip for x in groups):
        return 0
    else:
        return 1

def remove_non_latin_characters(text):
    """
    Function to remove non-latin characters.

    Input:
    text: string of characters

    Output:
    final_char: string of characters with non-latin characters removed
    """
    final_char = []
    groups_to_skip = ['HIRAGANA', 'CJK', 'KATAKANA','ARABIC', 'HANGUL', 'THAI','DEVANAGARI','BENGALI',
                      'THAANA','GUJARATI','CYRILLIC']
    for char in text:
        try:
            script = unicodedata.name(char).split(" ")[0]
            if script not in groups_to_skip:
                final_char.append(char)
        except:
            pass
    return "".join(final_char)
    
def group_non_latin_characters(text):
    """
    Function to group non-latin characters and return the number of latin characters.

    Input:
    text: string of characters

    Output:
    groups: list of character groups
    latin_chars: number of latin characters
    """
    groups = []
    latin_chars = []
    text = text.replace(".", "").replace(" ", "")
    for char in text:
        try:
            script = unicodedata.name(char).split(" ")[0]
            if script == 'LATIN':
                latin_chars.append(script)
            else:
                if script not in groups:
                    groups.append(script)
        except:
            if "UNK" not in groups:
                groups.append("UNK")
    return groups, len(latin_chars)

def check_for_non_latin_characters(text):
    """
    Function to check if non-latin characters are dominant in a text.

    Input:
    text: string of characters

    Output:
    0: if text should be not used
    1: if text should be used
    """
    groups, latin_chars = group_non_latin_characters(str(text))
    if name_to_keep_ind(groups) == 1:
        return 1
    elif latin_chars > 20:
        return 1
    else:
        return 0

def clean_title(old_title):
    """
    Function to check if title should be kept and then remove non-latin characters. Also
    removes some HTML tags from the title.
    
    Input:
    old_title: string of title
    
    Output:
    new_title: string of title with non-latin characters and HTML tags removed
    """
    keep_title = check_for_non_latin_characters(old_title)
    if (keep_title == 1) & isinstance(old_title, str):
        new_title = remove_non_latin_characters(old_title)
        if '<' in new_title:
            new_title = new_title.replace("<i>", "").replace("</i>","")\
                                 .replace("<sub>", "").replace("</sub>","") \
                                 .replace("<sup>", "").replace("</sup>","") \
                                 .replace("<em>", "").replace("</em>","") \
                                 .replace("<b>", "").replace("</b>","") \
                                 .replace("<I>", "").replace("</I>", "") \
                                 .replace("<SUB>", "").replace("</SUB>", "") \
                                 .replace("<scp>", "").replace("</scp>", "") \
                                 .replace("<font>", "").replace("</font>", "") \
                                 .replace("<inf>","").replace("</inf>", "") \
                                 .replace("<i /> ", "") \
                                 .replace("<p>", "").replace("</p>","") \
                                 .replace("<![CDATA[<B>", "").replace("</B>]]>", "") \
                                 .replace("<italic>", "").replace("</italic>","")\
                                 .replace("<title>", "").replace("</title>", "") \
                                 .replace("<br>", "").replace("</br>","").replace("<br/>","") \
                                 .replace("<B>", "").replace("</B>", "") \
                                 .replace("<em>", "").replace("</em>", "") \
                                 .replace("<BR>", "").replace("</BR>", "") \
                                 .replace("<title>", "").replace("</title>", "") \
                                 .replace("<strong>", "").replace("</strong>", "") \
                                 .replace("<formula>", "").replace("</formula>", "") \
                                 .replace("<roman>", "").replace("</roman>", "") \
                                 .replace("<SUP>", "").replace("</SUP>", "") \
                                 .replace("<SSUP>", "").replace("</SSUP>", "") \
                                 .replace("<sc>", "").replace("</sc>", "") \
                                 .replace("<subtitle>", "").replace("</subtitle>", "") \
                                 .replace("<emph/>", "").replace("<emph>", "").replace("</emph>", "") \
                                 .replace("""<p class="Body">""", "") \
                                 .replace("<TITLE>", "").replace("</TITLE>", "") \
                                 .replace("<sub />", "").replace("<sub/>", "") \
                                 .replace("<mi>", "").replace("</mi>", "") \
                                 .replace("<bold>", "").replace("</bold>", "") \
                                 .replace("<mtext>", "").replace("</mtext>", "") \
                                 .replace("<msub>", "").replace("</msub>", "") \
                                 .replace("<mrow>", "").replace("</mrow>", "") \
                                 .replace("</mfenced>", "").replace("</math>", "")

            if '<mml' in new_title:
                all_parts = [x for y in [i.split("mml:math>") for i in new_title.split("<mml:math")] for x in y if x]
                final_parts = []
                for part in all_parts:
                    if re.search(r"\>[$%#!^*\w.,/()+-]*\<", part):
                        pull_out = re.findall(r"\>[$%#!^*\w.,/()+-]*\<", part)
                        final_pieces = []
                        for piece in pull_out:
                            final_pieces.append(piece.replace(">", "").replace("<", ""))
                        
                        final_parts.append(" "+ "".join(final_pieces) + " ")
                    else:
                        final_parts.append(part)
                
                new_title = "".join(final_parts).strip()
            else:
                pass

            if '<xref' in new_title:
                new_title = re.sub(r"\<xref[^/]*\/xref\>", "", new_title)

            if '<inline-formula' in new_title:
                new_title = re.sub(r"\<inline-formula[^/]*\/inline-formula\>", "", new_title)

            if '<title' in new_title:
                new_title = re.sub(r"\<title[^/]*\/title\>", "", new_title)

            if '<p class=' in new_title:
                new_title = re.sub(r"\<p class=[^>]*\>", "", new_title)
            
            if '<span class=' in new_title:
                new_title = re.sub(r"\<span class=[^>]*\>", "", new_title)

            if 'mfenced open' in new_title:
                new_title = re.sub(r"\<mfenced open=[^>]*\>", "", new_title)
            
            if 'math xmlns' in new_title:
                new_title = re.sub(r"\<math xmlns=[^>]*\>", "", new_title)

        if '<' in new_title:
            new_title = new_title.replace(">i<", "").replace(">/i<", "") \
                                 .replace(">b<", "").replace(">/b<", "") \
                                 .replace("<inline-formula>", "").replace("</inline-formula>","")
        if new_title.isupper():
            new_title = new_title.title()
        
        return new_title
    else:
        return ''
    
def clean_abstract(raw_abstract, inverted=False):
    """
    Function to clean abstract and return it in a format for the model.
    
    Input:
    raw_abstract: string of abstract
    inverted: boolean to determine if abstract is inverted index or not
    
    Output:
    final_abstract: string of abstract in format for model
    """
    if inverted:
        if isinstance(raw_abstract, dict) | isinstance(raw_abstract, str):
            if isinstance(raw_abstract, dict):
                invert_abstract = raw_abstract
            else:
                invert_abstract = json.loads(raw_abstract)
            
            if invert_abstract.get('IndexLength'):
                ab_len = invert_abstract['IndexLength']

                if ab_len > 20:
                    abstract = [" "]*ab_len
                    for key, value in invert_abstract['InvertedIndex'].items():
                        for i in value:
                            abstract[i] = key
                    final_abstract = " ".join(abstract)[:2500]
                    keep_abs = check_for_non_latin_characters(final_abstract)
                    if keep_abs == 1:
                        pass
                    else:
                        final_abstract = None
                else:
                    final_abstract = None
            else:
                if len(invert_abstract) > 20:
                    abstract = [" "]*1200
                    for key, value in invert_abstract.items():
                        for i in value:
                            try:
                                abstract[i] = key
                            except:
                                pass
                    final_abstract = " ".join(abstract)[:2500].strip()
                    keep_abs = check_for_non_latin_characters(final_abstract)
                    if keep_abs == 1:
                        pass
                    else:
                        final_abstract = None
                else:
                    final_abstract = None
                
        else:
            final_abstract = None
    else:
        if raw_abstract:
            ab_len = len(raw_abstract)
            if ab_len > 30:
                final_abstract = raw_abstract[:2500]
                keep_abs = check_for_non_latin_characters(final_abstract)
                if keep_abs == 1:
                    pass
                else:
                    final_abstract = None
            else:
                final_abstract = None
        else:
            final_abstract = None
            
    return final_abstract

def get_top_keywords(title, abstract, cand_embs_df):
    """
    Function to use title, abstract, and candidate keyword embeddings to return scores.
    
    Input:
    title: title of paper
    abstract: abstract of paper
    cand_embs_df: dataframe containing keywords and embeddings (filtered by paper topics)
    
    Output:
    final_abstract: string of abstract in format for model
    """
    cand_embs_df = cand_embs_df.copy()
    if title.isupper():
        title = title.title()
    if abstract:
        title_and_abstract = f"{title}\n {abstract}"
    else:
        if title:
            title_and_abstract = f"{title}"
        else:
            title_and_abstract = ""

    if title_and_abstract:
        # Get title/abstract embedding
        title_abs_emb = emb_model.encode(title_and_abstract)
    
        # Get scores for each candidate keyword
        cand_embs_df['cand_scores'] = cand_embs_df['embedding'].apply(lambda x: np.dot(title_abs_emb, x))
    else:
        cand_embs_df['cand_scores'] = -1
    
    return cand_embs_df

def get_candidate_keywords(candidate_topics):
    """
    Function to get keywords based on the topics
    
    Input:
    candidate_topics: topics of paper
    
    Output:
    keywords_data_copy: filtered df of keywords and embeddings
    """
    keywords_data_copy = all_keywords_data[all_keywords_data['topic_id'].isin(candidate_topics)]\
        .drop_duplicates(subset=['keywords'])[['keywords','keyword_id','embedding']].copy()
    return keywords_data_copy

def get_all_keywords(candidate_topics, paper_title, abstract, invert_abstract=False, topk=5):
    """
    Function to get keywords that match title/abstract
    
    Input:
    candidate_topics: topic ids for a paper
    paper_title: title of a paper
    abstract: abstract of a paper
    invert_abstract: whether or not the abstract is being input as an inverted index (True/False)
    topk: maximum number of keywords to pull for a paper
    
    Output:
    final_keywords
    """
    # Process title and abstract
    paper_title = clean_title(paper_title)
    abstract = clean_abstract(abstract, inverted=invert_abstract)
    
    # Get candidate keywords
    keywords_data = get_candidate_keywords(candidate_topics)
    if keywords_data.shape[0]>0:
    
        # Get candidate scores
        cand_scores = get_top_keywords(paper_title, abstract, keywords_data)
        if cand_scores[cand_scores['cand_scores']>=0].shape[0] > 0:
            top_k = cand_scores[cand_scores['cand_scores']>=0].sort_values('cand_scores', ascending=False).head(topk)
            top_k = top_k.drop_duplicates(subset=['keyword_id'])
            keywords = top_k['keywords'].tolist()
            scores = top_k['cand_scores'].tolist()
    
            final_keywords = []
            _ = [final_keywords.append({"keyword_id": final_mapping_for_api[keyword], "score": round(score, 6)}) 
                 for keyword, score in zip(keywords, scores) if score > 0.50]
    
            if final_keywords:
                return final_keywords
            else:
                if scores[0] > 0.40:
                    return [{"keyword_id": final_mapping_for_api[keywords[0]], "score": round(scores[0], 6)}]
                else:
                    return []
        else:
            return []
    else:
        return []

def process_data_as_df(new_df):
    """
    Function to process data as a dataframe (in batch).
    
    Input:
    new_df: dataframe of data
    
    Output:
    input_df: dataframe of data with predictions
    """
    input_df = new_df.copy()
    
     # Get keywords
    input_df['keywords'] = input_df.apply(lambda x: get_all_keywords(x.topics, x.title, 
                                                                     x.abstract_inverted_index, 
                                                                     x.inverted, topk=5), axis=1)
    
    return input_df[['UID','keywords']].copy()

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy.
    """
    # Check if the classifier was loaded correctly
    try:
        _ = emb_model.device
        status = 200
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Return a prediction for the model.
    
    Input:
    JSON of data
    
    Output:
    JSON of predictions
    """
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    if isinstance(input_json, list):
        pass
    else:
        input_json = json.loads(input_json)
    
    input_df = pd.DataFrame.from_dict(input_json).reset_index().rename(columns={'index': 'UID'})

    final_df = process_data_as_df(input_df)

    # Transform predictions to JSON
    result = json.dumps(final_df['keywords'].tolist())
    return flask.Response(response=result, status=200, mimetype='application/json')
