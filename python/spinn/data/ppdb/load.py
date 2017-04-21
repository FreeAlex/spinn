
import json
import codecs
import csv

SENTENCE_PAIR_DATA = True

LABEL_MAP = {
    "Independent": 0,
    "ForwardEntailment": 1,
    "ReverseEntailment": 2,
    "Equivalence": 3,
    "OtherRelated": 4,
    # "Exclusion": 1,
}

def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions

def load_ppdb(path, lowercase=False, type='tsv', limit=None):
    print "Loading", path
    examples = []
    with codecs.open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect='excel-tab')
        count = 0
        skip = 0
        for line in reader:
            loaded_example = line
            if loaded_example["relation"] not in LABEL_MAP:
                skip += 1
                continue

            example = {}
            example["label"] = loaded_example["relation"]
            example["premise"] = loaded_example["source"]
            example["hypothesis"] = loaded_example["target"]
            (example["premise_tokens"], example["premise_transitions"]) = convert_binary_bracketing(loaded_example["source_bp"], lowercase=lowercase)
            (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert_binary_bracketing(loaded_example["target_bp"], lowercase=lowercase)
            example["example_id"] = loaded_example["id"]
            examples.append(example)
            count += 1
            if limit is not None and count >= limit:
                break
        print 'read: {}, skip: {}'.format(count, skip)

    return examples, None


if __name__ == "__main__":
    # Demo:
    examples,_ = load_ppdb('/Users/Alex/Data/ppdb/ppdb2-s-parsed.tsv', limit=10)
    print examples[0]
