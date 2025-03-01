import os
import re

import torch

is_mac = torch.backends.mps.is_available() and torch.backends.mps.is_built() # M1/M2 chip?
is_cuda = torch.cuda.is_available()
device = 'mps' if is_mac else 'cuda' if is_cuda else 'cpu'

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

def encode_tokens(pipe, prompt):
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
    embedding *= weights.unsqueeze(-1).repeat_interleave(embedding.shape[-1], dim=-1) # MPS-friendly ?
    current_mean = embedding.float().mean(axis=[-2, -1]).to(embedding.dtype)
    embedding *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1).repeat_interleave(embedding.shape[-1], dim=-1) # MPS-friendly ?

    return embedding

def read_txt(txt):
    if os.path.isfile(txt):
        with open(txt, 'r', encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        lines = [txt]
    return lines

def parse_line(txt, splitter='~'):
    subs = []
    wts = []
    for subtxt in txt.split('|'):
        if splitter in subtxt:
            [subtxt, wt] = subtxt.split(splitter)
            wt = float(wt)
        else: 
            wt = 1.
        subs += [subtxt.strip()]
        wts  += [wt]
    if all([w > 0 for w in wts]): wts = [w / sum(wts) for w in wts] # norm weights if all positive
    return subs, wts

def txt_clean(txt):
    if isinstance(txt, list):
        txt = [t.strip() for t in txt if len(t.strip()) > 0]
        txt = ' '.join(txt).strip()
    return ''.join(e for e in txt.replace(' ', '_') if (e.isalnum() or e in ['_','-']))

def multiprompt(pipe, in_txt, pretxt='', postxt='', repeat=1):
    prompts = []
    texts   = []
    for line in read_txt(in_txt):
        if len(line.strip()) > 0 and line.strip()[0] != '#':
            texts += [txt_clean(line)]
            line = ' '.join([pretxt, line, postxt])
            prompts += [parse_line(line)]
    if len(prompts)==0: prompts = [([''], [1.])] # uc
    embeds  = []
    weights = []
    for prompt in prompts:
        embatch = torch.cat([encode_tokens(pipe, parse_prompt(string)) for string in prompt[0]]) # [b,77,768]
        embeds += [embatch]
        wts = torch.Tensor(prompt[1])
        weights += [wts] # or [wts / wts.sum()] ?
    # MPS-friendly expand using repeat_interleave with explicit tensor copy
    embeds = torch.stack(embeds).repeat_interleave(repeat, dim=0)
    weights = torch.stack(weights).repeat_interleave(repeat, dim=0).to(embeds.device, dtype=embeds.dtype)
    return embeds, weights, texts

