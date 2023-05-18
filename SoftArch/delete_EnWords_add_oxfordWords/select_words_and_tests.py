import snowballstemmer
import pickle

def read_words(fname):
    lst = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                lst.append(line)
    return lst
            
        
with open('words_and_tests_enlarged__cleansing2.p', 'rb') as f:
    d = pickle.load(f)

# Step 1: don't include words whose only test type is EnWords
d2 = {}
for k in d:
    test_type_lst = list(set(d[k]))
    if len(test_type_lst) == 1 and 'EnWords' in test_type_lst:
        continue
    if 'EnWords' in test_type_lst:
        test_type_lst.remove('EnWords')
    d2[k] = test_type_lst

# Step 2: read Oxford 3000 and Oxford 5000
oxford3000 = read_words('oxford3000.txt')
oxford5000 = read_words('oxford5000.txt')

# Step: merge words in Oxford 3000 and Oxford 5000 into d2
d3 = d2.copy()

for word in oxford3000:
    if word in d2:
        test_type_lst = d2[word]
        test_type_lst.append('OXFORD3000')
        d3[word] = test_type_lst
    else:
        d3[word] = ['OXFORD3000']

for word in oxford5000:
    if word in d2:
        test_type_lst = d2[word]
        test_type_lst.append('OXFORD5000')
        d3[word] = test_type_lst
    else:
        d3[word] = ['OXFORD5000']

# Step:

stemmer = snowballstemmer.stemmer('english');

d4 = d3.copy()

for k in d3:
    d4[k] = sorted(list(set(d3[k])))
    original_word = k
    stem_word = stemmer.stemWord(k)
    if original_word != stem_word:
        #print('%s - %s' % (original_word, stem_word))
        #input()
        if stem_word not in d4:
            d4[stem_word] = d4[k]
        else:
            lst = d4[stem_word]
            lst.extend(d4[k])
            d4[stem_word] = sorted(list(set(lst.copy())))
            #print(stem_word)
            #print(d4[stem_word])
        if d4[original_word] == ['BBC']:
            d4[original_word] = d4[stem_word]

for k in d4:
    d4[k] = sorted(list(set(d4[k])))

for word in ['apple', 'apples', 'dog', 'dogs', 'understand', 'understood', 'leaf', 'leaves']:
    print(word)
    print(d4[word])

print(len(d4))
with open('my_words_and_tests.pickle', 'wb') as f:
    pickle.dump(d4, f)














