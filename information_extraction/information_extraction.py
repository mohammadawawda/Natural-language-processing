import os
import json
import random

import spacy
import wikipedia
from google import genai

nlp = spacy.load("en_core_web_sm")


def get_propn_phrase(token):
    """
    the proper noun of a head token: the token and its children with the compound label, in text order
    """
    compounds = [child for child in token.children if child.dep_ == "compound"]
    all_tokens = sorted(compounds + [token], key=lambda t: t.i)
    return " ".join(t.text for t in all_tokens)


def dependency_extractor(doc):
    """
    (subject, relation, object) triplets from the dependency tree of every sentence
    """
    triplets = []
    for sent in doc.sents:
        # proper noun heads: PROPN tokens that are not a compound of another token
        proper_noun_heads = [token for token in sent if token.pos_ == "PROPN" and token.dep_ != "compound"]

        # every ordered pair, the subject can also come after the object in the text
        for h1 in proper_noun_heads:
            for h2 in proper_noun_heads:
                if h1 == h2:
                    continue

                # condition 1: same head h, h1 is nsubj and h2 is dobj
                h = h1.head
                if h2.head == h and h1.dep_ == "nsubj" and h2.dep_ == "dobj":
                    triplets.append((get_propn_phrase(h1), h.text, get_propn_phrase(h2)))
                    continue

                # condition 2: h1 is nsubj of h, h2 is pobj of h0 and h0 is prep of h
                h0 = h2.head
                if h1.dep_ == "nsubj" and h2.dep_ == "pobj" and h0.dep_ == "prep" and h0.head == h:
                    triplets.append((get_propn_phrase(h1), h.text + " " + h0.text, get_propn_phrase(h2)))
    return triplets


def pos_extractor(doc):
    """
    (subject, relation, object) triplets from the pos tags only
    two proper nouns in a row with no punctuation between them and at least one verb between them,
    the relation is the verbs and prepositions between them
    """
    triplets = []
    propn_spans = []
    i = 0
    while i < len(doc):
        if doc[i].pos_ == "PROPN":
            start = i
            while i < len(doc) and doc[i].pos_ == "PROPN":
                i += 1
            propn_spans.append((start, i))
        else:
            i += 1

    for j in range(len(propn_spans) - 1):
        subj_start, subj_end = propn_spans[j]
        obj_start, obj_end = propn_spans[j + 1]
        between_tokens = doc[subj_end:obj_start]
        if any(token.pos_ == "PUNCT" for token in between_tokens):
            continue
        if not any(token.pos_ == "VERB" for token in between_tokens):
            continue
        subject = " ".join(token.text for token in doc[subj_start:subj_end])
        object_ = " ".join(token.text for token in doc[obj_start:obj_end])
        relation = " ".join(token.text for token in between_tokens if token.pos_ in ("VERB", "ADP"))
        triplets.append((subject, relation, object_))
    return triplets


def extract_from_wikipedia(page_title):
    """
    runs both extractors on a wikipedia page
    """
    try:
        page_content = wikipedia.page(title=page_title, auto_suggest=False).content
    except Exception:
        print(f"Page Error: {page_title} does not exist.")
        return [], []
    doc = nlp(page_content)
    return dependency_extractor(doc), pos_extractor(doc)


# the prompt we used
prompt = (
    "Extract all (Subject, Relation, Object) triplets from the following Wikipedia article text. "
    "Each triplet should be represented as a tuple in the form (Subject, Relation, Object). "
    "Return your answer as a JSON array of such tuples. "
    "Ensure that the triplets accurately reflect factual relationships mentioned in the text. "
    "For example, if the text mentions 'Neil Gorsuch was appointed to the Supreme Court', "
    "one valid triplet is (\"Neil Gorsuch\", \"appointed to\", \"Supreme Court\")."
)


def llm_extractor_gemini(text, prompt):
    """
    sends the prompt and the page to gemini, the key is read from GEMINI_API_KEY
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    contents = str(prompt) + "\n\nText:\n" + text
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )
    return response.text


def parse_llm_output(output):
    """
    the json list from the answer, the model often puts it inside a ```json block
    returns None if it's not valid json
    """
    output = output.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("Error parsing LLM output as JSON. Raw output:")
        print(output)
        return None


def extract_from_wikipedia_llm_gemini(page_title, prompt):
    """
    triplets of a wikipedia page from gemini, if the answer is not json every line is taken as a triplet
    """
    try:
        page_content = wikipedia.page(title=page_title, auto_suggest=False).content
    except Exception:
        print(f"PageError: '{page_title}' does not exist.")
        return None
    output = llm_extractor_gemini(page_content, prompt)
    triplets = parse_llm_output(output)
    if triplets is None:
        triplets = [line.strip() for line in output.split('\n') if line.strip()]
    return triplets


wiki_pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]

results = {}
for page in wiki_pages:
    print(f"\nExtracting triplets from: {page}")
    dep_triplets, pos_triplets = extract_from_wikipedia(page)
    print(f"Dependency-based extractor: Total triplets extracted = {len(dep_triplets)}")
    print(f"POS-based extractor: Total triplets extracted = {len(pos_triplets)}")

    # 5 random triplets of each extractor from every page, to check by hand
    sample_dep = random.sample(dep_triplets, min(5, len(dep_triplets)))
    sample_pos = random.sample(pos_triplets, min(5, len(pos_triplets)))

    print("\nDependency-based extractor sample triplets:")
    for t in sample_dep:
        print(t)

    print("\nPOS-based extractor sample triplets:")
    for t in sample_pos:
        print(t)

    results[page] = {
        "dependency_total": len(dep_triplets),
        "dependency_sample": sample_dep,
        "pos_total": len(pos_triplets),
        "pos_sample": sample_pos
    }

llm_results = {}
for page in wiki_pages:
    print(f"\nExtracting triplets using Gemini from: {page}")
    triplets_list = extract_from_wikipedia_llm_gemini(page, prompt)
    if triplets_list is None:
        print("No output returned.")
        continue
    print(f"Gemini extractor: Total triplets = {len(triplets_list)}")

    sample_triplets = random.sample(triplets_list, min(5, len(triplets_list)))
    print("Sample triplets from Gemini extractor:")
    for t in sample_triplets:
        print(t)
    llm_results[page] = {
        "total": len(triplets_list),
        "sample": sample_triplets
    }
