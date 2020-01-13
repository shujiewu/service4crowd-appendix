#!/usr/bin/python
# -*- coding: utf-8 -*

import pika
import functools
import threading
import logging
import time
import json
import yaml
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def ack_message(ch, delivery_tag, connection, result):
    """Note that `ch` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    LOGGER.info('complete')
    response_channel = connection.channel()
    response_channel.exchange_declare(
        exchange='SLURM_TRAINING_RESPONSE',
        exchange_type='topic',
        passive=False,
        durable=True,
        auto_delete=False)
    response_channel.basic_publish(
        exchange='SLURM_TRAINING_RESPONSE',
        routing_key='SLURM_TRAINING_RESPONSE',
        body=result,
        properties=pika.BasicProperties(content_type='application/json'))
    response_channel.close()
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def do_work(conn, ch, delivery_tag, body):
    # thread_id = threading.get_ident()
    LOGGER.info('Thread Delivery tag: %s Message body: %s', delivery_tag, body)
    # Sleeping to simulate 10 seconds of work
    res = json.loads(body)
    taskInfo = res['taskInfo']
    parameter = res['parameter']

    training_id = parameter['trainingId']
    try:
        template_config = '/home/LAB/wusj/exp/KL-Loss/configs/e2e_faster_rcnn_R-50-FPN_2x_fpn_no.yaml'
        new_config = '/home/LAB/wusj/fastwash_tmp/training/' + 'training_config_' + training_id + '.yaml'
        yaml_obj = {}
        with open(template_config, 'r') as f:
            yaml_obj = yaml.safe_load(f)
            if 'gpuNum' in parameter:
                yaml_obj['NUM_GPUS']=parameter['gpuNum']
            if 'maxIter' in parameter:
                yaml_obj['SOLVER']['MAX_ITER']=parameter['maxIter']
            # if 'gamma' in parameter:
            #     yaml_obj['SOLVER']['GAMMA']=parameter['gamma']
            # if 'baseLR' in parameter:
            #     yaml_obj['SOLVER']['BASE_LR']=parameter['baseLR']
            if 'lrs' in parameter:
                yaml_obj['SOLVER']['LRS'] = parameter['lrs']
            if 'weightDecay' in parameter:
                yaml_obj['SOLVER']['WEIGHT_DECAY']=parameter['weightDecay']
            if 'steps' in parameter:
                yaml_obj['SOLVER']['STEPS']=parameter['steps']
            yaml_obj['OUTPUT_DIR'] = "/home/LAB/wusj/fastwash_model/" + "output_" + training_id + "/"
            with open(new_config, 'wt') as new_file:
                yaml.dump(yaml_obj,new_file, default_flow_style=False,encoding='utf-8',allow_unicode=True)

        parameter_config = '/home/LAB/wusj/fastwash_tmp/training/' + 'parameter_config_' + training_id
        with open(parameter_config, 'wt') as f:
            json.dump(res, f)

        output = {}
        if 'simulate' in parameter and parameter['simulate'] == True:
            time.sleep(3)
            output['trainingId'] = training_id
            output['modelPkl'] = '/home/LAB/wusj/exp/output/test_base_plus/train/voc_2007_train/generalized_rcnn/model_final0.1.pkl'
            output['modelConfig'] = new_config
        else:
            time.sleep(3)
            output['trainingId'] = training_id
            output['modelPkl'] = ''
            output['modelConfig'] = new_config
            # (status, output) = commands.getstatusoutput(
            #     'srun --gres=gpu:V100:1 python /home/LAB/wusj/exp/KL-Loss/exp/training.py' + ' --input ' + config_file)
            # print(status)
            # print(output)
            output_file = '/home/LAB/wusj/fastwash_tmp/training/' + 'result_' + training_id
            with open(output_file, 'r') as f:
                res = json.load(f)
        res = {}
        res["taskInfo"] = taskInfo
        res["output"] = output
        res["success"] = True
        res['status'] = 'TRAINING_TASK_SUCCESS'

        res = json.dumps(res)
        cb = functools.partial(ack_message, ch, delivery_tag, conn, res)
        conn.add_callback_threadsafe(cb)
    except:
        import traceback, sys
        traceback.print_exc()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error = str(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))

        output = {"exception": error}
        res = {}
        res["taskInfo"] = taskInfo
        res["output"] = output
        res["success"] = False
        res['status'] = 'TRAINING_TASK_FAILED'
        res = json.dumps(res)
        cb = functools.partial(ack_message, ch, delivery_tag, conn, res)
        conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


if __name__ == "__main__":
    credentials = pika.PlainCredentials("guest", "guest")
    conn_params = pika.ConnectionParameters(host='10.1.1.63', port=5672, credentials=credentials, socket_timeout=500)
    # Infinite loop
    while True:
        try:
            connection = pika.BlockingConnection(conn_params)
            channel = connection.channel()
            channel.exchange_declare(
                exchange='SLURM_TRAINING_REQUEST',
                exchange_type='topic',
                passive=False,
                durable=True,
                auto_delete=False)
            channel.queue_declare(queue='SLURM_TRAINING_REQUEST', auto_delete=False)
            channel.queue_bind(
                queue='SLURM_TRAINING_REQUEST',
                exchange='SLURM_TRAINING_REQUEST',
                routing_key='SLURM_TRAINING_REQUEST')
            # channel.basic_qos(prefetch_count=1)
            threads = []
            on_message_callback = functools.partial(on_message, args=(connection, threads))
            channel.basic_consume('SLURM_TRAINING_REQUEST', on_message_callback)

            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()

            # Wait for all to complete
            for thread in threads:
                thread.join()

            connection.close()
            break
        # Do not recover if connection was closed by broker
        except pika.exceptions.ConnectionClosedByBroker:
            break
        # Do not recover on channel errors
        except pika.exceptions.AMQPChannelError:
            break
        # Recover on all other connection errors
        except pika.exceptions.AMQPConnectionError:
            continue