def overlap(a, b):
    for i in range(min(len(a), len(b)), -1, -1):
        if a.endswith(b[:i]):
            return i
    return 0


def shortestCommonSuperstring(string_set):
    while len(string_set) > 1:
        max_overlap, (i, j) = max(((overlap(string_set[i], string_set[j]), (i, j))
                                   for i in range(len(string_set))
                                   for j in range(i+1, len(string_set))),
                                  key=lambda x: x[0])

        string_set[i] += string_set[j][max_overlap:]
        string_set.pop(j)

    return string_set[0]







