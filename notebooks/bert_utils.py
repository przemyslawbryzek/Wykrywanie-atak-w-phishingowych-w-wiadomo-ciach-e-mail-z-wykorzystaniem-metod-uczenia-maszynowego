import gc
import numpy as np
import torch
from belt_nlp.splitting import split_overlapping
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

MODEL_NAME = "bert-base-uncased"
MASK_TOKENS = ["[URL]", "[EMAIL_ADDRESS]", "[DOMAIN]", "[FILE_PATH]", "[YEAR]", "[PHONE_NUMBER]"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_logging.set_verbosity_error()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": MASK_TOKENS})
    return tokenizer


def load_model(tokenizer, seed):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    original = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        for token in MASK_TOKENS:
            pieces = original(token, add_special_tokens=False)["input_ids"]
            embeddings[tokenizer.convert_tokens_to_ids(token)] = embeddings[pieces].mean(dim=0)
    return model.to(DEVICE)


def encode(tokenizer, texts, max_length=512, chunk_size=5000):
    texts = list(texts)
    sequences, lengths = [], []
    for start in range(0, len(texts), chunk_size):
        ids = tokenizer(
            texts[start:start + chunk_size], truncation=False, return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]
        for sequence in ids:
            lengths.append(len(sequence))
            if len(sequence) > max_length:
                sequence = sequence[:max_length - 1] + sequence[-1:]
            sequences.append(np.asarray(sequence, dtype=np.int32))
    return sequences, np.asarray(lengths, dtype=np.int32)


def encode_chunks(tokenizer, texts, chunk_size=510, stride=510, minimal_chunk_length=128, batch_texts=5000):
    texts = list(texts)
    documents, lengths = [], []
    cls, sep = [tokenizer.cls_token_id], [tokenizer.sep_token_id]
    for start in range(0, len(texts), batch_texts):
        ids = tokenizer(
            texts[start:start + batch_texts], add_special_tokens=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]
        for sequence in ids:
            lengths.append(len(sequence) + 2)
            chunks = split_overlapping(torch.tensor(sequence, dtype=torch.long), chunk_size, stride, minimal_chunk_length)
            documents.append([np.asarray(cls + chunk.tolist() + sep, dtype=np.int32) for chunk in chunks] or
                             [np.asarray(cls + sep, dtype=np.int32)])
    return documents, np.asarray(lengths, dtype=np.int32)


def _as_documents(sequences):
    return [sequence if isinstance(sequence, list) else [sequence] for sequence in sequences]


def _pad(tokenizer, sequences):
    length = max(len(sequence) for sequence in sequences)
    ids = np.full((len(sequences), length), tokenizer.pad_token_id, dtype=np.int64)
    mask = np.zeros((len(sequences), length), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = sequence
        mask[row, :len(sequence)] = 1
    return torch.from_numpy(ids).to(DEVICE), torch.from_numpy(mask).to(DEVICE)


def _chunk_logits(model, tokenizer, chunks):
    ids, mask = _pad(tokenizer, chunks)
    with torch.autocast(DEVICE.type, dtype=torch.float16):
        return model(input_ids=ids, attention_mask=mask).logits.squeeze(-1).float()


def _groups(documents, batch, max_chunks):
    groups, current, count = [], [], 0
    for index in batch:
        if current and count + len(documents[index]) > max_chunks:
            groups.append(current)
            current, count = [], 0
        current.append(index)
        count += len(documents[index])
    return groups + [current]


def training_batches(lengths, batch_size, rng, pool=50):
    order = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(order), batch_size * pool):
        chunk = order[start:start + batch_size * pool]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


@torch.no_grad()
def predict(model, tokenizer, sequences, batch_size=32):
    model.eval()
    documents = _as_documents(sequences)
    counts = np.array([len(document) for document in documents])
    chunks = [chunk for document in documents for chunk in document]
    owner = np.repeat(np.arange(len(documents)), counts)
    order = np.argsort([len(chunk) for chunk in chunks], kind="stable")
    logits = np.empty(len(chunks), dtype=np.float64)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        logits[batch] = _chunk_logits(model, tokenizer, [chunks[i] for i in batch]).cpu().numpy()
    mean_logits = np.bincount(owner, weights=logits, minlength=len(documents)) / counts
    return (1 / (1 + np.exp(-mean_logits))).astype(np.float32)


def train_bert(
    tokenizer, sequences, labels, learning_rate, seed, total_steps, batch_size=16, weight_decay=0.01, warmup=0.1,
    holdout=None, eval_every=None, patience=None, max_chunks=16, log=print,
):
    rng = np.random.default_rng(seed)
    model = load_model(tokenizer, seed)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler()
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="sum")
    documents = _as_documents(sequences)
    max_chunks = max(max_chunks, max(len(document) for document in documents))
    lengths = np.array([sum(len(chunk) for chunk in document) for document in documents])
    targets = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    history, best_score, best_step, best_state, step = [], -np.inf, total_steps, None, 0
    evaluations_since_best = 0
    while step < total_steps:
        model.train()
        for batch in training_batches(lengths, batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            for group in _groups(documents, batch, max_chunks):
                counts = [len(documents[i]) for i in group]
                chunk_logits = _chunk_logits(model, tokenizer, [chunk for i in group for chunk in documents[i]])
                logits = torch.stack([part.mean() for part in chunk_logits.split(counts)])
                loss = loss_function(logits, targets[group].to(DEVICE)) / len(batch)
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if holdout is not None and (step % eval_every == 0 or step == total_steps):
                holdout_sequences, holdout_labels, holdout_weights = holdout
                score = average_precision_score(
                    holdout_labels, predict(model, tokenizer, holdout_sequences), sample_weight=holdout_weights,
                )
                history.append((step, score))
                log(f"  krok {step}/{total_steps}: AUCPR ważone {score:.4f}")
                if score > best_score:
                    best_score, best_step, evaluations_since_best = score, step, 0
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                else:
                    evaluations_since_best += 1
                    if evaluations_since_best >= patience:
                        step = total_steps
                model.train()
            if step >= total_steps:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, scheduler, scaler, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return model, history, best_step
