import tfcoreml as tf_converter
#tf_converter.convert(tf_model_path = 'EmoNet.pb', mlmodel_path = 'EmoNet.mlmodel', output_feature_names = ['softmax:0'], input_name_shape_dict = {'image:0' : [1, 48, 48, 1]})
#image_input_names='image:0'
tf_converter.convert(tf_model_path = 'EmoNet.pb',
	mlmodel_path = 'EmoNet.mlmodel',
	output_feature_names = ['softmax:0'],
	image_input_names='image:0',
	image_scale = 1.0/255.0,
	class_labels = 'labels.txt',
	gray_bias = 0.0)
