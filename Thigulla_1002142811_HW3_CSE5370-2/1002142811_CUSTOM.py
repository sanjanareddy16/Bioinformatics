#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
class Custom_Alignment:

    
    def local_alignment(self, sequence_A: str, sequence_B:str, substitution: dict, gap: int ):
        n = len(sequence_A)
        m = len(sequence_B)
        D = [[0] * (m+1) for i in range(n+1)]
        for i in range(n + 1):
            D[i][0] = 0
        for j in range(m + 1):
            D[0][j] = 0
        max_score= 0
        max_i = []
        max_j = []
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_score = substitution[sequence_A[i-1]][sequence_B[j-1]]
                diagonal = D[i-1][j-1] + match_score
                horizontal = D[i-1][j] + gap
                vertical = D[i][j-1] + gap
                D[i][j] = max(0,diagonal, horizontal, vertical) 
                if D[i][j] > max_score:
                    max_score = D[i][j]
                    
        coords_arr=[]           
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                  if D[i][j] == max_score:
                    max_i.append(i)
                    max_j.append(j)
                    coords = (D[0][i], D[1][i])
                    coords_arr.append(coords)
                  
        print(np.matrix(D))
        print(max_score)
        
  

    
    
    
    
        arr=[]
        for x in range(len(max_i)):
            i = max_i[x]
            j = max_j[x]
            i = i-1
            j = j-1
            str1=""
            str2=""
            str1+=sequence_A[i]
            str2+=sequence_B[j]
            while D[i][j] != 0:
                diag = D[i - 1][j - 1]
                up = D[i - 1][j]
                lt = D[i][j - 1]
                max_points = max(diag, up, lt)
                D[i][j] = 0
              # diag
                if i > 0 and j > 0 and diag == max_points:
                    str1+=sequence_A[i-1]
                    str2+=sequence_B[j-1]
                    i -= 1
                    j -= 1
              # up
                elif j > 0 and up == max_points:
                    str1+="_"
                    str2+=sequence_B[j-1]
                    j -= 1
              # left
                else:
                    str1+="_"
                    str2+=sequence2[i-1]
                    i -= 1
            arr.append((str1[::-1], str2[::-1]))
        return arr
    
    

    
seqA = 'sanjana'
seqB = 'reddy'
seqC = seqA + seqB
seqD = 'abcdefghijklmnopqrstuvwxyz'
seqE = 'thequickbrownfoxjumpsoverthelazydog'
seqF = 'sanjanareddy'


sub_matrix = {}
dict1 = {}
for i in seqD:
    dict2 = {}
    for j in seqD:
        if i in seqC and j in seqC:
            val = 1
        elif i == j:
            val = 2
        else:
            val = -1
        dict2[j] = val
    dict1[i] = dict2
    sub_matrix[i] = dict2
    
    
ans=Custom_Alignment()
ans=ans.local_alignment("sanjanareddy","thequickbrownfoxjumpsoverthelazydog",dict1,-2)
print(ans)

    
    



    
mat_main=[]
mat=[]
mat.append('-')
for i in seqD:
    mat.append(i)
mat_main.append(mat)

for i in dict1:
    mat=[]
    mat.append(i)
    dict2=dict1[i]
    for j in dict2:
        mat.append(dict2[j])
    mat_main.append(mat)

    

np.matrix(mat_main)


# In[ ]:




