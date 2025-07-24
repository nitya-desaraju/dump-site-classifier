import random
import numpy as np

def test_train_id(image_ids_updated_pkl: str):
    with open(image_ids_updated_pkl, 'rb') as f:
        loaded_data = pickle.load(f)

    keys = np.array(loaded_data)
    random.shuffle(keys)

    train_items = keys[:int(keys.shape[0]*0.8)]
    test_items = keys[int(keys.shape[0]*0.8):]

    train_data = []
    for i, k in enumerate(train_items):
        image_id = k.name
        caption_id = k.caption_ID[random.randint(0, 4)]
        confusor_id = train_items[random.randint(0, train_items.shape[0]-1)].name
        ar = (image_id, caption_id, confusor_id)
        train_data.append(ar)

    
    test_data = []
    for i, k in enumerate(test_items):
        image_id = k.name
        caption_id = k.caption_ID[random.randint(0, 4)]
        confusor_id = test_items[random.randint(0, test_items.shape[0]-1)].name
        ar = (image_id, caption_id, confusor_id)
        # print(ar)
        test_data.append(ar)

    return (train_data, test_data)