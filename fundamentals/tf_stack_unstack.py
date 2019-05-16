"""
https://blog.csdn.net/weixin_35718886/article/details/79013372
"""
import tensorflow as tf

session = tf.Session()
x = [1., 4.]
y = [2., 5.]
z = [3., 6.]

stack_rslt = tf.stack([x, y, z],axis=0)
axis0 = session.run(stack_rslt)
print("axis0:", "\n", axis0)
print("axis0.shape:", "\n", axis0.shape)

stack_rslt = tf.stack([x, y, z],axis=1)
axis1 = session.run(stack_rslt)
print("axis1:", "\n", axis1)
print("axis1.shape:", "\n", axis1.shape)

x = [[1,1,1,1],[2,2,2,2],[3,3,3,3]]
unstack_rslt = tf.unstack(x, axis=0)
axis0 = session.run(unstack_rslt)
print("u_axis0:", "\n", axis0)
print("u_axis0[1]:", "\n", axis0[1])
print("u_axis0[0].shape:", "\n", axis0[0].shape)

x = [[1,1,1,1],[2,2,2,2],[3,3,3,3]]
unstack_rslt = tf.unstack(x, axis=1)
axis1 = session.run(unstack_rslt)
print("u_axis1:", "\n", axis1)
print("u_axis1[1]:", "\n", axis1[1])
print("u_axis1[0].shape:", "\n", axis1[0].shape)


x2 = [0., 1., 2., 3., 4., 5., 6., 7.]
y = tf.reshape(x2, [2, 2, 2])

