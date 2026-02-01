from keras_facenet import FaceNet

embedder = FaceNet()
print(embedder.model.output_shape)
