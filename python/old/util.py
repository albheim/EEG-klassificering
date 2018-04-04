import numpy as np

def kfold(n, k, shuffle=False):
    s = np.arange(n)
    if shuffle:
        np.random.shuffle(s)
    l = []
    for a in range(k):
        val = s[int(n * a / k):int(n * (a + 1) / k)]
        tr = np.concatenate((s[:int(n * a / k)], s[int(n * (a + 1) / k):]))
        l.append((tr, val))

    return l

def validation_split(n):
    s = np.arange(n)
    np.random.shuffle(s)
    val = s[int(n):]
    tr = s[:int(n)]
    return (tr, val)
