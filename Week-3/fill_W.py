def fill_W(image_ids_updated_pkl, model):
    
    with open(image_ids_updated_pkl, 'rb') as f:
        loaded_data = pickle.load(f)

    for k in loaded_data:
        k.W = model(k.name)

    with open('image_ids_filled.pkl', 'wb') as file:
        pickle.dump(loaded_data, file)

    # return loaded_data #idk if we actually need this, but it's here just in case