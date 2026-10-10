"""Deterministic lexical fallback and strict, provider-independent vector scoring."""
from collections import Counter
import json
import math
import re

from .models import Query, Record


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run) == 1:
            words.append(run)
    return words


def record_text(record: Record) -> str:
    # Scope/time applicability remain hard gates, never inferred from a vector.
    return json.dumps({"question":record.question,"task":record.key,"event":record.event,
        "state":record.required_state,"action":record.action,"outcome":record.outcome,
        "entry":record.evolution.stages[0].model_dump(mode="json",exclude={"at","observation_ids"})
        if record.evolution else None},ensure_ascii=False,sort_keys=True)


def query_text(query: Query) -> str:
    return json.dumps({"question":query.question,"task":query.key,"event":query.event,
                       "phase":query.phase,"state":query.state},ensure_ascii=False,sort_keys=True)


def lexical_scores(query: Query, records: list[Record]) -> dict[str,float]:
    # Do not tokenize JSON field names: those would spuriously match every record.
    documents = [tokens(r.question + " " + r.key + " " + (r.event or "") + " " +
                       " ".join(r.required_state.values()) + " " +
                       (r.evolution.stages[0].phase if r.evolution else "")) for r in records]
    q = tokens(query.question)
    counts = [Counter(d) for d in documents]
    frequency = Counter(t for d in counts for t in d)
    idf = {t:math.log((1+len(records))/(1+frequency[t]))+1 for t in set(q)|set(frequency)}
    def weighted(values):
        return {t:(1+math.log(n))*idf[t] for t,n in Counter(values).items()}
    a = weighted(q)
    anorm = math.sqrt(sum(v*v for v in a.values()))
    output = {}
    for record,doc in zip(records,documents):
        b = weighted(doc)
        denom = anorm*math.sqrt(sum(v*v for v in b.values()))
        output[record.id] = sum(v*b.get(t,0) for t,v in a.items())/denom if denom else 0.0
    return output


def validate_vectors(values, count: int) -> list[list[float]]:
    if not isinstance(values,list) or len(values) != count:
        raise ValueError("Invalid embedding batch size.")
    result, dimension = [], None
    for vector in values:
        if not isinstance(vector,list) or not vector or len(vector) > 65536:
            raise ValueError("Invalid embedding vector.")
        try:
            if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in vector):
                raise ValueError("Embedding contains nonfinite or nonnumeric values.")
            norm = math.hypot(*vector)
        except OverflowError:
            raise ValueError("Embedding contains an overflowing value.") from None
        if not norm or not math.isfinite(norm):
            raise ValueError("Embedding must have a finite nonzero norm.")
        if dimension is not None and dimension != len(vector):
            raise ValueError("Embedding dimensions differ.")
        dimension = len(vector)
        result.append([v/norm for v in vector])
    return result


def vector_scores(query_vector, vectors, records):
    normalized = validate_vectors([query_vector,*vectors],len(records)+1)
    return {r.id:max(0.0,min(1.0,sum(a*b for a,b in zip(normalized[0],v))))
            for r,v in zip(records,normalized[1:])}
