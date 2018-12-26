import tensorflow as tf


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

f = x * x * y + y + 2


# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)


# Creating a new graph

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is tf.get_default_graph())
