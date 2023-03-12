import os
import re

import torch

device = torch.device('cuda')

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

def parse_prompt(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      preceeding '\' disables char parsing, i.e. \( = literal character '('
      anything else - just text
    >>> parse_prompt('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [("", 1.0)]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

def txt_clean(txt):
    if isinstance(txt, list):
        txt = [t.strip() for t in txt if len(t.strip()) > 0]
        txt = ' '.join(txt).strip()
    return ''.join(e for e in txt.replace(' ', '_') if (e.isalnum() or e in ['_','-']))

def read_txt(txt):
    if os.path.isfile(txt):
        with open(txt, 'r', encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        lines = [txt]
    return lines

def parse_line(txt):
    subs = []
    for subtxt in txt.split('|'):
        if ':' in subtxt:
            [subtxt, wt] = subtxt.split(':')
            wt = float(wt)
        else: wt = 1e-4 if len(subtxt.strip())==0 else 1.
        subs.append([subtxt.strip(), wt])
    return subs # list of tuples

def read_multitext(in_txt, prefix=None, postfix=None):
    if in_txt is None or len(in_txt)==0: return [[('', 1.)]]
    prompts = [parse_line(tt) for tt in read_txt(in_txt) if tt.strip()[0] != '#']
    if prefix is not None and len(prefix) > 0:
        prefixs  = read_txt(prefix)
        prompts = [parse_line(prefixs[i % len(prefixs)]) + prompts[i]   for i in range(len(prompts))]
    if postfix is not None and len(postfix) > 0:
        postfixs = read_txt(postfix)
        prompts = [prompts[i] + parse_line(postfixs[i % len(postfixs)]) for i in range(len(prompts))]
    return prompts # [list of [list of (text, weight)]]

def multiencode(pipe, prompt):
    maxlen = pipe.tokenizer.model_max_length
    bos = pipe.tokenizer.bos_token_id # 49406
    eos = pipe.tokenizer.eos_token_id # 49407

    tokens = [bos]
    weights = [1.]
    for text, weight in prompt: # prompt = [list of (text, weight)]
        token = pipe.tokenizer(text).input_ids[1:-1] # tokenize and discard start/end token
        tokens += token
        weights += [weight] * len(token) # copy the weight by length of token
    if len(tokens) > maxlen-1:
        tokens = tokens[:maxlen-1]
        weights = weights[:maxlen-1]
    tokens  += [eos] * (maxlen - len(tokens))
    weights += [1.0] * (maxlen - len(weights))

    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    embedding = pipe.text_encoder(tokens)[0]
    weights = torch.tensor([weights], dtype=embedding.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    previous_mean = embedding.float().mean(axis=[-2, -1]).to(embedding.dtype)
    embedding *= weights.unsqueeze(-1)
    current_mean  = embedding.float().mean(axis=[-2, -1]).to(embedding.dtype)
    embedding *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    return embedding

def multiprompt(pipe, in_txt, pretxt='', postxt='', parens=False):
    if parens:
        prompts = [parse_prompt(''.join([pretxt, line, postxt])) for line in read_txt(in_txt)]
    else:
        prompts = read_multitext(in_txt, pretxt, postxt)

    embeddings = [multiencode(pipe, p) for p in prompts] # list of [1,77,768]
    texts = [txt_clean([p[0] for p in prompt]) for prompt in prompts]
    return embeddings, texts

