from cogworks_data.language import get_data_path
from pathlib import Path
import json
import pickle

class image_id:
    def __init__(self, name, url):
        self.name = name #image_ID
        self.url= url
        self.caption_ID = []
        self.caption_index = []
        self.descriptor= None
        self.W= None

    def add_caption_ID(self, caption_ID):
        self.caption_ID.append(caption_ID)

    def add_caption_index(self, index):
        self.caption_index.append(index)
    
class caption:
    def __init__(self, name, img_id, string):
        self.name = name #caption_ID
        self.image_ID = img_id
        self.caption= string
        self.W_norm = None


def create_class():
    filename = get_data_path("captions_train2014.json")
    with Path(filename).open() as f:
        coco_data = json.load(f)
    image_ids = []
    captions = []
    index_map = {}
    counter = 0
    for image in coco_data["images"]:
        image_ids.append(image_id(image["id"], image["coco_url"]))
        index_map[image["id"]] = counter
        counter+=1
        
    counter = 0
    for cap in coco_data["annotations"]:
        captions.append(caption(cap["id"], cap["image_id"], cap["caption"]))
        image_ids[index_map[cap["image_id"]]].add_caption_ID(cap["id"])
        image_ids[index_map[cap["image_id"]]].add_caption_index(counter)
        counter+=1

    with open('image_ids.pkl', 'wb') as f_img:
        pickle.dump(image_ids, f_img)

    with open('captions.pkl', 'wb') as f_cap:
        pickle.dump(captions, f_cap)

    return image_ids, captions
        
        
        
    
    
    
    
