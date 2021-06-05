import tempfile
import pickle
import tensorflow as tf
from botocore.exceptions import ClientError
from decimal import Decimal
from io import BytesIO
import numpy as np
import boto3
import json
import gzip
import sys

from tensorflow.keras import optimizers
sys.path.insert(0, '/mnt/efs/tensorflow')


bucket_name = 'storage-cirrus'
s3_client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='AKIAXAB365XRJIRUI7HP',
                         aws_secret_access_key='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')

dynamodb = boto3.resource(
    'dynamodb', region_name='us-east-1', aws_access_key_id='AKIAXAB365XRJIRUI7HP',
    aws_secret_access_key='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')
lbd = boto3.client('lambda', region_name='us-east-1', aws_access_key_id='AKIAXAB365XRJIRUI7HP',
                   aws_secret_access_key='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')

num_epochs = 5
batch_size = 50
minibatch_num = 5  # batch = 5 minibatchs
minibatch_size = int(batch_size / minibatch_num)


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output


def FMNISTload_data():
    base = '/mnt/efs/datasets/fashion-mnist/'
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(base + fname)

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


class FMNISTLoader():
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_data, self.train_label), (self.test_data,
                                              self.test_label) = FMNISTload_data()
        # Normalization
        self.train_data = np.expand_dims(
            self.train_data.astype(np.float32) / 255.0, axis=-1)

        self.test_data = np.expand_dims(
            self.test_data.astype(np.float32) / 255.0, axis=-1)
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)

        self.num_train_data, self.num_test_data = self.train_data.shape[
            0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


data_loader = FMNISTLoader()


def update_info(field, seq, batch, fin_count_increase, dynamodb=dynamodb):
    if not dynamodb:
        dynamodb = boto3.resource(
            'dynamodb', region_name='us-east-1', aws_access_key_id='AKIAXAB365XRJIRUI7HP',
            aws_secret_access_key='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')

    table = dynamodb.Table('ParameterServer')

    response = table.update_item(
        Key={
            'field': field,
            'seq': seq
        },
        UpdateExpression="set info.btc=:b, info.finish_count=info.finish_count + :f",
        ExpressionAttributeValues={
            ':b': Decimal(batch),
            ':f': Decimal(fin_count_increase),
        },
        ReturnValues="UPDATED_NEW"
    )
    return response


def get_info(field, seq, dynamodb=dynamodb):
    if not dynamodb:
        dynamodb = boto3.resource(
            'dynamodb', region_name='us-east-1', aws_access_key_id='AKIAXAB365XRJIRUI7HP',
            aws_secret_access_key='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')

    table = dynamodb.Table('ParameterServer')

    try:
        response = table.get_item(Key={'field': field, 'seq': seq})
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        return response['Item']


def put_info(field, seq, batch, finish_count, dynamodb=dynamodb):
    if not dynamodb:
        dynamodb = boto3.resource(
            'dynamodb', region_name='us-east-1', aws_access_key_id='AKIAXAB365XRJIRUI7HP',
            aws_secret_access_key='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')

    table = dynamodb.Table('ParameterServer')
    response = table.put_item(
        Item={
            'field': field,
            'seq': seq,
            'info': {
                'btc': batch,
                'finish_count': finish_count,
            }
        }
    )
    return response


def scheduler(event, context):
    eventRecords = event['Records']
    eventRecords = eventRecords[0]
    
    batch_train_data, batch_train_label = data_loader.get_batch(batch_size)
        
    if eventRecords['eventSource'] == 'aws:invoke':
        batch_info = get_info('batch', 0)
        batch = batch_info['info']['btc']
        responseBody = None
        
        
        if batch == 0:
            model = MLP()
            model.save_weights('/mnt/efs/ckpt')

            batch += 1
            update_info('batch', 0, batch, 0)
            put_info('finc', batch, 0, 0)
        

        if eventRecords['TrainStatus'] == 'ON':
            # sperate batch into minibatch and upload
            for i in range(minibatch_num):
                mini_idx = np.random.randint(0, batch_size, minibatch_size)
                minibatch_data = batch_train_data[mini_idx, :]
                minibatch_label = batch_train_label[mini_idx]
                with tempfile.TemporaryFile() as fp_data:
                    pickle.dump(minibatch_data, fp_data)
                    fp_data.seek(0)
                    s3_client.upload_fileobj(
                        fp_data, bucket_name, 'dataset/data/' + str(batch) + '/' +
                        str(i) + '_minibatch'
                    )

                with tempfile.TemporaryFile() as fp_label:
                    pickle.dump(minibatch_label, fp_label)
                    fp_label.seek(0)
                    s3_client.upload_fileobj(
                        fp_label, bucket_name, 'dataset/label/' + str(batch) + '/' +
                        str(i) + '_minibatch'
                    )
            responseBody = {
                'UploadData': 'Complete'
            }
        else:  # Finish Training
            responseBody = {
                'TrainStatus': 'OFF'
            }
            with tempfile.TemporaryFile() as fp_off:
                s3_client.upload_fileobj(
                    fp_off, bucket_name, 'OFF_' + str(batch)
                )
        response = {
            'statusCode': 200,
            'body': json.dumps(responseBody)
        }
        return response

    elif eventRecords['eventSource'] == 'aws:dynamodb':  # every minibatch trigger
        # if update finish_count
        batch_info = get_info('batch', 0)
        batch = batch_info['info']['btc']
        finc_info = get_info('finc', batch)
        finc = finc_info['info']['finish_count']
        response = None
        if finc >= minibatch_num:
            # average grads
            with tempfile.TemporaryFile() as fp_grads:
                s3_client.download_fileobj(
                    bucket_name, 'grads/' + str(batch) + '/' + str(minibatch_num-1), fp_grads)
                fp_grads.seek(0)
                grads = pickle.load(fp_grads)

            for miniidx in range(minibatch_num-1):
                with tempfile.TemporaryFile() as fp_grads:
                    s3_client.download_fileobj(
                        bucket_name, 'grads/' + str(batch) + '/' + str(miniidx), fp_grads)
                    fp_grads.seek(0)
                    newgrads = pickle.load(fp_grads)
                for i, ts in enumerate(newgrads):
                    grads[i] = grads[i] + ts

            for i, ts in enumerate(grads):
                grads[i] = grads[i]/minibatch_num

            model = MLP()
            mini_idx = np.random.randint(0, batch_size, minibatch_size)
            minibatch_data = batch_train_data[mini_idx, :]
            model(minibatch_data)
            optimizer = tf.keras.optimizers.SGD(0.1)
            model.load_weights('/mnt/efs/ckpt')
            optimizer.apply_gradients(
                grads_and_vars=zip(grads, model.trainable_weights))

            model.save_weights('/mnt/efs/ckpt')
            msg = "Batch %d trainings complete, waiting for client." % int(
                batch)

            s3_client.upload_fileobj(
                BytesIO(bytes(msg, encoding='utf-8')), bucket_name, 'msgs/' +
                str(batch)
            )

            response = {
                "msg": msg
            }

            batch += 1
            update_info('batch', 0, batch, 0)
            put_info('finc', batch, 0, 0)

        return response


def trainer(event, context):
    batch_info = get_info('batch', 0)
    batch = batch_info['info']['btc']

    # download minibatch
    eventRecords = event['Records']
    eventRecords = eventRecords[0]

    eventS3 = eventRecords['s3']
    eventObj = eventS3['object']
    objKey = eventObj['key']
    seq = objKey.split('_')[0][-1]

    with tempfile.TemporaryFile() as fp_data:
        s3_client.download_fileobj(
            bucket_name, 'dataset/data/' + str(batch) + '/' +
            seq + '_minibatch', fp_data)
        fp_data.seek(0)
        mb_data = pickle.load(fp_data)

    with tempfile.TemporaryFile() as fp_label:
        s3_client.download_fileobj(
            bucket_name, 'dataset/label/' + str(batch) + '/' +
            seq + '_minibatch', fp_label)
        fp_label.seek(0)
        mb_label = pickle.load(fp_label)

    # restore model
    model = MLP()
    model.load_weights('/mnt/efs/ckpt')

    # training -> grads
    with tf.GradientTape() as tape:
        y_pred = model(mb_data)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=mb_label, y_pred=y_pred)
        loss = tf.reduce_mean(loss)

    cur_grads = tape.gradient(loss, model.trainable_weights)
    print("batch:%d seq:%d after grad" % (batch, int(seq)))
    print(model.trainable_weights)
    print('\n')

    with tempfile.TemporaryFile() as fp_grads:
        pickle.dump(cur_grads, fp_grads)
        fp_grads.seek(0)
        response = s3_client.upload_fileobj(
            fp_grads, bucket_name, 'grads/' +
            str(batch) + '/' + seq
        )
    # update finish_count
    finc_info = get_info('finc', batch)
    finc = finc_info['info']['finish_count']
    if finc < minibatch_num:
        update_info('finc', batch, 0, 1)
    return response


def client(event, context):
    # num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    num_batches = 500
    batch_info = get_info('batch', 0, dynamodb)
    batch = batch_info['info']['btc']
    TrainStatus = 'ON'
    if batch >= num_batches:
        TrainStatus = 'OFF'
        model = MLP()

        model.load_weights('/mnt/efs/ckpt')
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        num_batches = int(data_loader.num_test_data // batch_size)
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * \
                batch_size, (batch_index + 1) * batch_size
            y_pred = model.predict(
                data_loader.test_data[start_index: end_index])
            sparse_categorical_accuracy.update_state(
                y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
        
        print('accuracy : %f' % sparse_categorical_accuracy.result())
                

        with tempfile.TemporaryFile() as fp_result:
            pickle.dump(sparse_categorical_accuracy.result(), fp_result)
            fp_result.seek(0)
            s3_client.upload_fileobj(
                fp_result, bucket_name, 'result_accuracy_' + str(batch)
            )

    payload = {
        "Records": [
            {
                "eventSource": "aws:invoke",
                "TrainStatus": TrainStatus
            }
        ]
    }

    response = lbd.invoke(
        FunctionName='arn:aws:lambda:us-east-1:481162096098:function:cirrus-scheduler',
        Payload=json.dumps(payload),
    )
    return json.loads(response.get('Payload').read())
