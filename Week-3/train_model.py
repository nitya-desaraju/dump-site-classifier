import mygrad as mg
import mynn
import numpy as np
import matplotlib.pyplot as plt
from noggin import create_plot
import pickle
%matplotlib inline
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
from mygrad.nnet.initializers.glorot_normal import glorot_normal
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam

class Model:
    def __init__(self, input_dim, output_dim):
        self.dense = dense(input_dim, output_dim, weight_initializer=glorot_normal, bias=False)
        
    def __call__(self, x):
        return self.dense(x)
        
    @property
    def parameters(self):
        return self.dense.parameters


params = []
train_accuracies = []
test_accuracies = []

def train_model(train, test, margin, image_ids_updated_pkl, captions_pkl, batch_size=32, num_epochs=250):
    global train_accuracies
    global test_accuracies
    
    caption_ids, true_ids, confusor_ids = train # true-caption-ID, true-image-ID, conufuser-image-ID
    caption_test, true_test, confusor_test = test
    
    with open(image_ids_updated_pkl, 'rb') as f:
        image_ids = pickle.load(f)
    with open(caption_pkl, 'rb') as g:
        caption = pickle.load(g)
    
    true_descriptors = mg.tensor([list(image_ids[x].descriptor) for x in true_ids])
    confusor_descriptors = mg.tensor([list(image_ids[x].descriptor) for x in confusor_ids])
    caption_embeddings = mg.tensor([list(caption[x].W_norm) for x in caption_ids])
    
    true_descriptors_test = mg.tensor([list(image_ids[x].descriptor) for x in true_test])
    confusor_descriptors_test = mg.tensor([list(image_ids[x].descriptor) for x in confusor_test])
    caption_embeddings_test = mg.tensor([list(caption[x].W_norm) for x in caption_test])
    
    N = true_descriptors.shape[0]
    
    model = Model(true_descriptors.shape[-1], caption_embeddings.shape[-1])
    
    op = SGD(model.parameters, learning_rate=1e-3, momentum=0.9)
    
    for epoch_cnt in range(num_epochs):
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        
        for batch_cnt in range(0, N//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            
            true_batch = true_descriptors[batch_indices]
            confused_batch = confusor_descriptors[batch_indices]
            caption_batch = caption_embeddings[batch_indices]
            
            sim_true = model(true_batch) * caption_batch
            sim_confused = model(confused_batch) * caption_batch
            
            loss = margin_ranking_loss(x1=sim_true, x2=sim_confused, margin=margin)
            loss.backward()
            
            op.step()

            #plotter.set_train_batch({"loss" : loss.item()}, batch_size=batch_size)

        if epoch_cnt % 25 == 0:
            params.append(model.parameters)
            
            with mg.no_autodiff:
                sim_true_test = mg.einsum("nd,nd -> n", model(true_descriptors_test), caption_embeddings_test)
                sim_confused_test = mg.einsum("nd,nd -> n", model(confusor_descriptors_test), caption_embeddings_test)
                loss = margin_ranking_loss(x1=sim_true_test, x2=sim_confused_test, margin=margin)
                test_accuracies.append(loss.item())
                
                sim_true = mg.einsum("nd,nd -> n", model(true_descriptors), caption_embeddings)
                sim_confused = mg.einsum("nd,nd -> n", model(confusor_descriptors), caption_embeddings)
                loss = margin_ranking_loss(x1=sim_true, x2=sim_confused, margin=margin)
                train_accuracies.append(loss.item())

            #print(f'epoch {epoch_cnt:5}, loss = {loss.item():0.3f}')
            #plotter.set_train_epoch()
    
    return model


def generate_linear_encoder(image_ids_updated_pkl, captions_pkl, margin=0.5):
    test, train = Test_train_ID(image_ids_updated_pkl)
    model = train_model(test=test, train=train, image_ids_updated_pkl=image_ids_updated_pkl, captions_pkl=captions_pkl, margin=0.5)
    return model
