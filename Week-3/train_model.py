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


def train_model(train, test, image_ids_updated_pkl, captions_pkl, margin=0.25, batch_size=32, num_epochs=3):
    global train_accuracies
    global test_accuracies
    
    caption_ids, true_ids, confusor_ids = train # true-caption-ID, true-image-ID, conufuser-image-ID
    caption_test, true_test, confusor_test = test
    
    #with open(image_ids_updated_pkl, 'rb') as f:
    #    image_ids = pickle.load(f)
    #with open(captions_pkl, 'rb') as g:
    #    captions = pickle.load(g)
    #true_descriptors = mg.tensor([(image_ids[x].descriptor) for x in true_ids])
    #confusor_descriptors = mg.tensor([list(image_ids[x].descriptor) for x in confusor_ids])
    #caption_embeddings = mg.tensor([list(caption[x].W_norm) for x in caption_ids])
    true_descriptors = mg.tensor([x.descriptor for x in true_ids])
    true_descriptors = true_descriptors.reshape((true_descriptors.shape[0], true_descriptors.shape[2]))
    confusor_descriptors = mg.tensor([x.descriptor for x in confusor_ids])
    confusor_descriptors = confusor_descriptors.reshape((confusor_descriptors.shape[0], confusor_descriptors.shape[2]))
    caption_embeddings = mg.tensor([x.W_norm for x in caption_ids])
    
    #true_descriptors_test = mg.tensor([list(image_ids[x].descriptor) for x in true_test])
    #confusor_descriptors_test = mg.tensor([list(image_ids[x].descriptor) for x in confusor_test])
    #caption_embeddings_test = mg.tensor([list(caption[x].W_norm) for x in caption_test])
    true_descriptors_test = mg.tensor([x.descriptor for x in true_test])
    true_descriptors_test = true_descriptors_test.reshape((true_descriptors_test.shape[0], true_descriptors_test.shape[2]))
    confusor_descriptors_test = mg.tensor([x.descriptor for x in confusor_test])
    confusor_descriptors_test = confusor_descriptors_test.reshape((confusor_descriptors_test.shape[0], confusor_descriptors_test.shape[2]))
    caption_embeddings_test = mg.tensor([x.W_norm for x in caption_test])

    print(true_descriptors.shape, confusor_descriptors.shape, caption_embeddings.shape)
    print(true_descriptors_test.shape, confusor_descriptors_test.shape, caption_embeddings_test.shape)
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
            #print(true_batch.shape, predictions.shape, caption_batch.shape)
            
            sim_true = mg.einsum("nd,nd -> n", model(true_batch), caption_batch)
            sim_confused = mg.einsum("nd,nd -> n", model(confused_batch), caption_batch)
            loss = margin_ranking_loss(x1=sim_true, x2=sim_confused, y=1, margin=margin)
            loss.backward()
            
            op.step()
            if batch_cnt%10==0:
                print(f"Epoch {epoch_cnt}, Batch {batch_cnt} : {loss.item()}")
                
            #plotter.set_train_batch({"loss" : loss.item()}, batch_size=batch_size)

        if epoch_cnt % 1 == 0:
            params.append(model.parameters)
            
            with mg.no_autodiff:
                print(true_descriptors_test.shape)
                #predictions_test = model(true_descriptors_test)
                #predictions_test = predictions.reshape((predictions_test.shape[0], predictions_test.shape[2]))
                #confused_predictions_test = model(confusor_descriptors_test)
                #confused_predictions_test = confused_predictions.reshape((confused_predictions_test.shape[0], confused_predictions_test.shape[2]))
                
                sim_true_test = mg.einsum("nd,nd -> n", model(true_descriptors_test), caption_embeddings_test)
                sim_confused_test = mg.einsum("nd,nd -> n", model(confusor_descriptors_test), caption_embeddings_test)
                loss_test = margin_ranking_loss(x1=sim_true_test, x2=sim_confused_test, y=1, margin=margin)
                test_accuracies.append(loss_test.item())

                #predictions_train = model(true_descriptors)
                #predictions_train = predictions.reshape((predictions_train.shape[0], predictions_train.shape[2]))
                #confused_predictions_train = model(confusor_descriptors)
                #confused_predictions_train = confused_predictions.reshape((confused_predictions_train.shape[0], confused_predictions_train.shape[2]))
                
                sim_true_train = mg.einsum("nd,nd -> n", model(true_descriptors), caption_embeddings)
                sim_confused_train = mg.einsum("nd,nd -> n", model(confusor_descriptors), caption_embeddings)
                loss_train = margin_ranking_loss(x1=sim_true_train, x2=sim_confused_train, y=1, margin=margin)
                train_accuracies.append(loss_train.item())

                print(f'Epoch {epoch_cnt}, Train Accuracy: {loss_train.item()}, Test Accuracy: {loss_test.item()}')

            #print(f'epoch {epoch_cnt:5}, loss = {loss.item():0.3f}')
            #plotter.set_train_epoch()
    
    return model

def generate_linear_encoder(image_ids_updated_pkl, captions_pkl, margin=0.25):
    test, train = test_train_id(image_ids_updated_pkl, captions_pkl)
    model = train_model(test=test, train=train, image_ids_updated_pkl=image_ids_updated_pkl, captions_pkl=captions_pkl, margin=margin)
    return model
