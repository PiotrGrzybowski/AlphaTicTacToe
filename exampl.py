import tensorflow as tf
import numpy as np

x = tf.placeholder('float', name='X')
tf.summary.scalar('addition', x)
summary_op = tf.summary.merge_all()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('Graphs', session.graph)
    for i in range(100):
        var1 = np.random.rand()
        add, s_ = session.run([x, summary_op], feed_dict={x: var1})
        writer.add_summary(s_, i)
