from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()
#
print(f"텐서플로 버전: {tf.__version__}")
# print(f"즉시 실행: {tf.executing_eagerly()}")


column_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', 'bool']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("특성: {}".format(feature_names))
print("레이블: {}".format(label_name))

batch_size = 99

train_dataset = tf.contrib.data.make_csv_dataset(
    'medi120.csv',
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)


# features, labels = next(iter(train_dataset))


#
# print(features)
#
# plt.scatter(features['13'].numpy(),
#             features['14'].numpy(),
#             c=labels.numpy(),
#             cmap='viridis')
#
# plt.xlabel('13')
# plt.ylabel('14')
#
# plt.show()


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(28, activation=tf.nn.relu, input_shape=(14,)),
    tf.keras.layers.Dense(14, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2)
])

predictions = model(features)


print(predictions[:5])
print(type(predictions[:5]))
print("소프트맥스", tf.nn.softmax(predictions[:5]))

print("  예측: {}".format(tf.argmax(predictions, axis=1)))
print("레이블: {}".format(labels))


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
print("손실 테스트: {}".format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.contrib.eager.Variable(0)

loss_value, grads = grad(model, features, labels)

print("단계: {}, 초기 손실: {}".format(global_step.numpy(),
                                 loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

print("단계 : {},    손실: {}".format(global_step.numpy(),
                                  loss(model, features, labels).numpy()))

from tensorflow import contrib

tfe = contrib.eager

# 도식화를 위해 결과를 저장합니다.
train_loss_results = []
train_accuracy_results = []

num_epochs = 100000
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # 훈련 루프 - 32개의 배치를 사용합니다.
    for x, y in train_dataset:
        # 모델을 최적화합니다.
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)

        # 진행 상황을 추적합니다.
        epoch_loss_avg(loss_value)  # 현재 배치 손실을 추가합니다.
        # 예측된 레이블과 실제 레이블 비교합니다.
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # epoch 종료
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("에포크 {:03d}: 손실: {:.3f}, 정확도: {:.3%}".format(epoch,
                                                           epoch_loss_avg.result(),
                                                           epoch_accuracy.result()))

model.save('my_model.hdf5')


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()
