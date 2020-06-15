
def create_model():
   model = Transformer()
   model.add(Dense(64, input_dim=14, init='uniform'))
   model.add(LeakyReLU(alpha=0.3))
   model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
   model.add(Dropout(0.5)) 
   model.add(Dense(64, init='uniform'))
   model.add(LeakyReLU(alpha=0.3))
   model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
   model.add(Dropout(0.5))
   model.add(Dense(2, init='uniform'))
   model.add(Activation('softmax'))
   return model

transformer = create_model()
transformer.load_weights("TM.h5")