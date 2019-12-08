car_drive.py contains the DeepRL, ReplayBuffer, and CarRacer classes.

CarRacer creates an instance of DeepRL (the DDQN neural network) and loads an existing
model if specified. CarRacer can train the DeepRL NN, saving predicted Q-values and model
weights periodically. CarRacer can also load an existing model and run episodes of the 
game using the model and plot Q-values.

DeepRL is the DDQN neural network. It has a model and a target_model, as specified in the
Google DeepMind paper. The models take in inputs of size 96 x 96*3 x NUM_FRAMES, representing 
10 stacked 96x96 pixel RGB images, and output a vector of size equal to the number of possible
actions.

ReplayBuffer stores experience tuples, and batches of tuples can be sampled from ReplayBuffer
to train model weights.

Google Drive link for saved models: 
https://drive.google.com/file/d/1pkdQBTPNHZKKpjfQu2vouqSpAhlaxRYB/view?usp=sharing
