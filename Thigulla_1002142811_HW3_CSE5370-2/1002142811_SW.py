#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Solution:
    def local_alignment(self, sequence_A: str, sequence_B:str, substitution: dict, gap: int ) -> [tuple]:
        
        # Initializing the matrix D
        n = len(sequence_A)
        m = len(sequence_B)
        D = [[0] * (m+1) for i in range(n+1)]
        for i in range(n+1):
            D[i][0] = 0
        for j in range(m+1):
            D[0][j] = 0
        max_score, max_i, max_j = 0, 0, 0
        
        # Fill the matrix D
        for i in range(1, n+1):
            for j in range(1, m+1):
                match_score = substitution[sequence_A[i-1]][sequence_B[j-1]]
                diagonal = D[i-1][j-1] + match_score
                horizontal = D[i-1][j] + gap
                vertical = D[i][j-1] + gap
                D[i][j] = max(0,diagonal, horizontal, vertical)
                if D[i][j] > max_score:
                    max_score = D[i][j]
                    max_i, max_j = i, j
                
        #printing the matrix D 
        for i in D:
            print(i)
    
        
        #alignments
        alignmentA = ""
        alignmentB = ""

        i, j = max_i, max_j
        for _ in range(i*j):
            if D[i][j] != 0:    
                if i > 0 and j > 0:
                    if D[i][j] == D[i - 1][j] + gap:
                        alignmentA += sequence_A[i - 1]
                        alignmentB += '-'
                        i = i - 1
                    elif D[i][j] == (D[i - 1][j - 1] + substitution[sequence_A[i - 1]][sequence_B[j - 1]]):
                        alignmentA = alignmentA + sequence_A[i - 1]
                        alignmentB = alignmentB + sequence_B[j - 1]
                        i = i - 1
                        j = j - 1
                    elif D[i][j] == D[i][j - 1] + gap:
                        alignmentA += '-'
                        alignmentB += sequence_B[j - 1]
                        j = j - 1
                    
                elif j > 0:
                    alignmentA += '-'
                    alignmentB += sequence_B[j - 1]
                    j = j - 1
                elif i > 0:
                    alignmentA += sequence_A[i - 1]
                    alignmentB += '-'
                    i -= 1

            alignmentA = alignmentA[::-1]
            alignmentB = alignmentB[::-1]

        return (alignmentA, alignmentB)


# In[3]:

"""
#checking with the given example
sol = Solution()

sub_matrix = {}
seqB = 'GATA'
seqC = 'CTAC'

seqA = set(seqB + seqC)
for i in seqA:
    dict = {}
    for j in seqA:
        if i == j:
            dict[j] = 1
        else:
            dict[j] = -1
    sub_matrix[i] = dict
print(sol.local_alignment(seqB, seqC, sub_matrix, gap=-2))

"""

# In[ ]:




