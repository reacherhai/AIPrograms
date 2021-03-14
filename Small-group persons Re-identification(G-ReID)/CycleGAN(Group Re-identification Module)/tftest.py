import tensorflow as tf
import timeit



out = tf.random.normal([100,10])
out = tf.nn.softmax(out,axis = 1)
pred = tf.argmax(out, axis = 1)

y = tf.random.uniform([100],dtype = tf.int64, maxval = 10)
out = tf.equal(pred,y)

out = tf.cast(out,dtype = tf.float32)
correct = tf.reduce_sum(out)

x = tf.random.normal([4,28,28,1])
y = tf.pad(x,[[0,0],[5,5],[5,5],[0,0]])
print(y)

for epoch in range(epochs):
    for _ in range(5):
        batch_z =


exit(0)

#chapter 5 finished!
#tomorrow: learn the neural network,CycleGAN



with tf.GradientTape() as tape:
    w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
    b1 = tf.Variable(tf.zeros(256))

    w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
    b2 = tf.Variable(tf.zeros(128))

    w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
    b3 = tf.Variable(tf.zeros(10))


    x = tf.reshape(x, [-1,28*28])

    h1 = x@w1 +tf.broadcast_to(b1, [x.shape[0],256])
    h1 = tf.nn.relu(h1)

    h2 = h1@w2 + b2
    h2 = tf.nn.relu(h2)

    out = h2@w3 + b3

    loss = tf.square(y_onehot - out)
    loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
    w1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])
    w2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    w3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])


with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([10000,1000])
    cpu_b = tf.random.normal([1000,2000])
    print(cpu_a.device, cpu_b.device)

with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000,1000])
    gpu_b = tf.random.normal([1000,2000])
    print(gpu_a.device, gpu_b.device)

def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a,cpu_b)
    return c

def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a,gpu_b)
    return c

print(timeit.timeit(gpu_run,number=10))
print(timeit.timeit(cpu_run,number=10))

print(timeit.timeit(gpu_run,number=10))
print(timeit.timeit(cpu_run,number=10))
exit(0)

x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    y = a**2 * x + b * x +c

[dy_da, dy_db, dy_dc] = tape.gradient(y, [a,b,c])
print(dy_da,dy_db,dy_dc)
