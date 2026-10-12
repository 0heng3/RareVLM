"""Pinned lmms-eval ANLS semantics, including its raw-length denominator."""


def levenshtein(a,b):
    if len(a)<len(b):a,b=b,a
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def anls(prediction,references):
    if not references:raise ValueError('ANLS reference list empty')
    values=[]
    for answer in references:
        a=' '.join(answer.strip().lower().split())
        p=' '.join(prediction.strip().lower().split())
        distance=levenshtein(a,p)
        length=max(len(answer.upper()),len(prediction.upper()))
        values.append(0. if length==0 else distance/length)
    result=1-min(values)
    return 0. if result<.5 else result
