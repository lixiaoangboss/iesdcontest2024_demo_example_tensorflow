import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset  # 您可能需要调整这部分，以确保数据加载和转换与 TensorFlow 兼容
from models.model_tf import AFNet
from tensorflow.keras.optimizers.schedules import CosineDecay


def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    # tf.config.experimental.set_visible_devices([], 'GPU')
    # print(tf.config.list_physical_devices('GPU'))

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiating NN
    net = AFNet()
    
    # 创建余弦退火调度器
    lr_schedule = CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=15000,
        alpha=0.01  # 最小学习率为 initial_learning_rate * alpha
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)# adam优化器
    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Start dataset loading
    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)
    # trainloader = trainloader.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
    testloader = create_dataset(testset, BATCH_SIZE)
    # testloader = testloader.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    number=0
    print("Start training")
    best=0
    # history = net.fit(trainloader, epochs=EPOCH, validation_data=testloader, verbose=1)
    for epoch in range(EPOCH):
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        number+=1
        for step, (x, y) in enumerate(trainloader):
            with tf.GradientTape() as tape:
                logits = net(x, training=True)
                loss = loss_object(y, logits)
                grads = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(grads, net.trainable_variables))
                pred = tf.argmax(logits, axis=1)
                correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
                accuracy += correct / x.shape[0]
                correct = 0.0

                running_loss += loss
                i += 1
        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))
        print("lr:",optimizer.lr)
        Train_loss.append(running_loss / i)
        Train_acc.append(accuracy / i)

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        for x, y in testloader:
            logits = net(x, training=False)
            test_loss = loss_object(y, logits)
            pred = tf.argmax(logits, axis=1)
            total += y.shape[0]
            correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
            running_loss_test += test_loss
            i += x.shape[0]

        print('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total))
    # Save model
        if correct / total > best:
            best=correct / total
            net.save('./saved_models/ECG_net_tf'+str(number)+'.h5')

    # Write results to file
    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    print('Finish training')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)# 训练轮数
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)# 训练学习率
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=80)# 批量大小
    argparser.add_argument('--size', type=int, default=1251)# 总共训练的行数
    argparser.add_argument('--path_data', type=str, default='./training_dataset/')# 训练集
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')# 标签集
    args = argparser.parse_args()

    main()
