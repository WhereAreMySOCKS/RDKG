import pickle

with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
with open('train.pkl', 'rb') as f:
    train = pickle.load(f)
    #   构造输入数据
print(train)
input_list = []
for item in test_dataset:
    for key in item:
        if type(key) == int:
            d = key
            s = item[key]
            temp = []
            temp.append(s)
            temp.append(d)
            temp.append(item['attribute_idxs'])
            temp.append(item['attribute_text'])
            input_list.append(temp)
print(input_list)