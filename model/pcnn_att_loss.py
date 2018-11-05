from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_att_loss(is_training, is_dev=False):
	if is_training:
		framework = Framework(is_training=True)
	else:
		framework = Framework(is_training=False, use_bag=False)

	word_embedding = framework.embedding.word_embedding()
	pos_embedding = framework.embedding.pos_embedding()
	embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
	x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)

	if is_training:
		logit, repre, weights_by_atten = framework.selector.weighted_loss(x, framework.scope, framework.label_for_select)
		loss = framework.classifier.weighted_softmax_cross_entropy(logit, weights_by_atten)
		output = framework.classifier.output(logit)
		framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
		framework.load_train_data()
		framework.train()
	else:
		logit, repre = framework.selector.no_bag(x)
		framework.init_test_model(logit) # TODO
		if is_dev:
			framework.load_dev_data()
			framework.test()
		else:
			framework.load_test_data()
			framework.test()

