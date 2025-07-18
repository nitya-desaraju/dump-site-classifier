def cos_dist(img_descriptors, database_descriptors):
    dot_product = img_descriptors @ database_descriptors.T
    magnitudes_img = np.linalg.norm(img_descriptors, axis = 1)
    magnitudes_database = np.linalg.norm(database_descriptors, axis = 1)
    magnitude_product = magnitudes_img.reshape(-1, 1) @ magnitudes_database.reshape(1, -1)
    return (1 - dot_product/magnitude_product)