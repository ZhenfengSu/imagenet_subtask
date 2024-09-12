from nltk.corpus import wordnet as wn 
import os
import shutil
image_net_path = '/home/zhenfeng/project/swintf/imagenet_1k/imagenet/'
subtask_type = [  
    'bird.n.01',                # 鸟  
    'canine.n.02',              # 犬  
    'feline.n.01',              # 猫科动物  
    'fish.n.01',                # 鱼  
    'insect.n.01',              # 昆虫  
    'monkey.n.01',              # 猴  
    'reptile.n.01',             # 爬行动物  
    'snake.n.01',               # 蛇  
    'vehicle.n.01',             # 车辆  
    'electronic_equipment.n.01',# 电子设备  
    'instrument.n.01',          # 乐器  
    'tool.n.01',                # 工具  
    'weapon.n.01',              # 武器  
    'building.n.01',            # 建筑物  
    'furniture.n.01',           # 家具  
    'container.n.01',           # 容器  
    'clothing.n.01',            # 衣物  
    'fruit.n.01',               # 水果  
    'vegetable.n.01',           # 蔬菜  
    'plant.n.02'                # 植物  
]
def get_tar_class_map():
    tar_set =set()
    tar2class = dict()
    with open('train.txt') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line_list = line.split(' ')
            img_name = line_list[0]
            class_name = line_list[1][:-1] if line_list[1][-1] == '\n' else line_list[1]
            tar_name = img_name.split('/')[0]
            if not tar_name in tar_set:
                tar_set.add(tar_name)
                tar2class[tar_name] = class_name
    return tar2class
def get_type_list(id_num):
    synset =  wn.synset(subtask_type[id_num])  
    types_set = set([w for s in synset.closure(lambda s:s.hyponyms()) for w in s.lemma_names()])
    types = list(set([w for s in synset.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
    assert len(types_set) == len(types)

    type_list = set()
    with open('synset_words.txt') as f:
        all_lines = f.readlines()
        id_len = len('n07932039')
        for line in all_lines:
            # 获取图片前缀与labels
            address = line[0:id_len]
            labels = line[id_len+1:-1]
            # 获取不同的label
            label_list = labels.split(', ')
            # 将不同的label的空格用_拼接起来
            for i, label_item in enumerate(label_list):
                label_item = label_item.replace(' ','_')
                label_list[i] = label_item
            for label_item in label_list:
                if label_item in types_set:
                    type_list.add(address)
    print(subtask_type[id_num])
    print(len(type_list))
    return list(type_list)
if __name__ == '__main__':
    print(len(subtask_type))
    hash_map = get_tar_class_map()
    assert len(hash_map) == 1000
    for i in range(len(subtask_type)):
        if subtask_type[i] == 'vehicle.n.01' or subtask_type[i] == 'bird.n.01':
            type_list = get_type_list(i)
            # 取前20种
            type_list = type_list[:20]
            # 获取符合条件的class
            fit_class = [hash_map[item] for item in type_list]
            assert len(fit_class) == 20
            # 创建文件夹
            file_name = 'vehicle' if subtask_type[i] == 'vehicle.n.01' else 'bird'
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            # 创建train和val文件夹
            if not os.path.exists(file_name + '/train'):
                os.makedirs(file_name + '/train')
            if not os.path.exists(file_name + '/val'):
                os.makedirs(file_name + '/val')
            # 读取train文件夹下的所有子文件夹
            train_list = os.listdir(image_net_path+'train')
            counter_map = dict()
            class_counter = 0
            for class_name in train_list:
                if class_name[len('class'):] in fit_class:
                    class_counter += 1
                    counter_map[class_name[len('class'):]] = class_counter # 记录每个class的counter，避免后续val文件夹错位
                    counter = 1
                    if not os.path.exists(file_name + '/train/'+'class' + str(class_counter)):
                        os.makedirs(file_name + '/train/'+'class' + str(class_counter))
                    # 复制前面1000张图片到train文件夹下
                    img_list = os.listdir(image_net_path+'train/' + class_name)
                    for img_name in range(1,1001):
                        shutil.copy(image_net_path+'train/' + class_name + '/' + 'img'+str(img_name)+'.jpeg', file_name + '/train/'+'class' + str(class_counter)+'/')
                        print('copy',image_net_path+'train/' + class_name + '/' + 'img'+str(img_name)+'.jpeg'+' to ' +  file_name + '/train/'+'class' + str(class_counter)+'/')
                        counter += 1
                    assert counter == 1001
            
            # 读取val文件夹下的所有子文件夹
            val_list = os.listdir(image_net_path+'val')
            for class_name in val_list:
                if class_name[len('class'):] in fit_class:
                    counter = 1
                    val_counter = counter_map[class_name[len('class'):]]
                    if not os.path.exists(file_name + '/val/'+'class' + str(val_counter)):
                        os.makedirs(file_name + '/val/'+'class' + str(val_counter))
                    # 复制前面50张图片到train文件夹下
                    img_list = os.listdir(image_net_path+'val/' + class_name)
                    for img_name in range(1,51):
                        shutil.copy(image_net_path+'val/' + class_name + '/' + 'img'+str(img_name)+'.jpeg', file_name + '/val/'+ 'class' + str(val_counter)+'/')
                        print('copy',image_net_path+'val/' + class_name + '/' + 'img'+str(img_name)+'.jpeg'+' to ' +  file_name + '/val/'+'class' + str(val_counter)+'/')
                        counter += 1
                    assert counter == 51
                    
                        
            