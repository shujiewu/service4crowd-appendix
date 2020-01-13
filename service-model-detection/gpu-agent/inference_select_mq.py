#!/usr/bin/python
# -*- coding: utf-8 -*

import pika
import functools
import threading
import logging
import time
import json
import yaml
import subprocess
import commands
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
        exchange='SLURM_SELECT_RESPONSE',
        exchange_type='topic',
        passive=False,
        durable=True,
        auto_delete=False)
    response_channel.basic_publish(
        exchange='SLURM_SELECT_RESPONSE',
        routing_key='SLURM_SELECT_RESPONSE',
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

    inference_id = parameter['inferenceId']
    model_config = parameter['modelConfig']
    model_pkl = parameter['modelPkl']

    try:
        parameter_config = '/home/LAB/wusj/fastwash_tmp/inference/' + 'parameter_config_' + inference_id
        with open(parameter_config, 'wt') as f:
            json.dump(res, f)
        # output = {}
        # # time.sleep(3)
        # output['annotationList'] = []
        # output['selectImageIdList'] = []
        # output['remainImageIdList'] = []
        # for i in range(parameter['selectNum']):
        #     output['selectImageIdList'].append(parameter['imageIdList'][i])
        # for i in range(parameter['selectNum'],len(parameter['imageIdList'])):
        #     output['remainImageIdList'].append(parameter['imageIdList'][i])
        # output['inferenceId'] = inference_id
        # output['remainImageNum'] = len(output['remainImageIdList'])
        #command_str = 'srun --gres=gpu:V100:1 python /home/LAB/wusj/exp/KL-Loss/exp/inference_entropy.py'+' --input '+parameter_config
        #process = subprocess.Popen(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            stdout = ""
            (status, stdout) = commands.getstatusoutput(
                'srun --gres=gpu:V100:1 python /home/LAB/wusj/exp/KL-Loss/exp/inference_entropy.py' + ' --input ' + parameter_config)
            LOGGER.info("status:'%s'", status)
            # for info in process.communicate():
            #     stdout = stdout + str(info, encoding="utf-8")
            #     LOGGER.info(str(info, encoding="utf-8"))
            LOGGER.info("output:'%s'", stdout)
            LOGGER.info("=== Run (ID '%s') succeeded ===", inference_id)

            result_output_dir = '/home/LAB/wusj/fastwash_tmp/inference/'
            with open(result_output_dir + 'result_' + inference_id, 'r') as f2:
                output = json.load(f2)
            res = {}
            res["taskInfo"] = taskInfo
            res["stdout"] = stdout
            res["output"] = output
            res["success"] = True
            res['status'] = 'INFERENCE_TASK_SUCCESS'
            res = json.dumps(res)
            cb = functools.partial(ack_message, ch, delivery_tag, conn, res)
            conn.add_callback_threadsafe(cb)
        except KeyboardInterrupt:
            LOGGER.error("=== Run (ID '%s') interrupted, cancelling run ===", inference_id)
            # process.cancel()
            raise
        except Exception:
            LOGGER.error("=== Run (ID '%s') failed ===", inference_id)
            raise

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
        res['status'] = 'INFERENCE_TASK_FAILED'
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
                exchange='SLURM_SELECT_REQUEST',
                exchange_type='topic',
                passive=False,
                durable=True,
                auto_delete=False)
            channel.queue_declare(queue='SLURM_SELECT_REQUEST', auto_delete=False)
            channel.queue_bind(
                queue='SLURM_SELECT_REQUEST',
                exchange='SLURM_SELECT_REQUEST',
                routing_key='SLURM_SELECT_REQUEST')
            # channel.basic_qos(prefetch_count=1)
            threads = []
            on_message_callback = functools.partial(on_message, args=(connection, threads))
            channel.basic_consume('SLURM_SELECT_REQUEST', on_message_callback)

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