import os
import random
 
train_percent = 0.7
test_percent = 0.3

img_path = './dataset/images'
total_img = os.listdir(img_path)

num = len(total_img)
list = range(num)

train_num = int(num * train_percent) #640 * 0.9
test_num = int(num * test_percent)#640 * 0.1

train_index = random.sample(list, train_num)

ftrain = open('./dataset/train.txt', 'w')
ftest = open('./dataset/test.txt', 'w')

for i in list:
    name = total_img[i][:-4] + '\n'
    if i in train_index:
        ftrain.write(name)
    else:
        ftest.write(name)
 
ftrain.close()
ftest.close()
