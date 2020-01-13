#!/usr/bin/env python

import pika
import functools
import threading
import logging
import time
import json
import argparse
import os
import yaml
import hashlib
import process
import subprocess
from six.moves import shlex_quote
from mongo import MongoGridFS

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


MLPROJECT_FILE_NAME = "mlproject"
DEFAULT_CONDA_FILE_NAME = "conda.yaml"
MLFLOW_CONDA_HOME = "MLFLOW_CONDA_HOME"



class Parameter(object):
    """A parameter in an MLproject entry point."""
    def __init__(self, name, yaml_obj):
        self.name = name
        self.type = yaml_obj.get("type", "string")
        self.default = yaml_obj.get("default")
    # def _compute_path_value(self, user_param_value, storage_dir):
    #     basename = os.path.basename(user_param_value)
    #     dest_path = os.path.join(storage_dir, basename)
    #     if dest_path:
    #         if not os.path.exists(dest_path):
    #             raise Exception("Got value %s for parameter %s, but no such file or "
    #                                      "directory was found." % (user_param_value, self.name))
    #         return os.path.abspath(dest_path)
    #     return os.path.abspath(dest_path)
    def _compute_file_value(self, key, user_param_value, data_dir):
        dest_path = data_dir
        if user_param_value is not None and user_param_value!="":
            file = user_param_value
            db = MongoGridFS("")
            db.downLoadFilebyID("fs", file['fileId'], os.path.join(dest_path, file['fileName']))
            return os.path.abspath(os.path.join(dest_path, file['fileName']))
        else:
            return user_param_value
    def _compute_path_value(self, key, user_param_value, data_dir):
        dest_path = os.path.join(data_dir, key)
        if dest_path is not None and not os.path.exists(dest_path):
            os.makedirs(dest_path)
        if user_param_value is not None and user_param_value != "":
            path = user_param_value
            db = MongoGridFS("")
            for file in path:
                db.downLoadFilebyID("fs", file['fileId'],os.path.join(dest_path, file['fileName']))
            return os.path.abspath(dest_path)
        else:
            return user_param_value

    def _compute_object_value(self, key, user_param_value, data_dir):
        dest_path = os.path.join(data_dir, key)
        if user_param_value is not None and user_param_value!="":
            with open(dest_path, 'wt') as f:
                json.dump(user_param_value, f)
            return os.path.abspath(dest_path)
        else:
            return user_param_value
    def _compute_list_value(self, key, user_param_value, data_dir):
        dest_path = os.path.join(data_dir, key)
        if user_param_value is not None and user_param_value!="":
            with open(dest_path, 'wt') as f:
                json.dump(user_param_value, f)
            return os.path.abspath(dest_path)
        else:
            return user_param_value

    def compute_value(self, key, param_value, data_dir):
        if data_dir and self.type == "file":
            return self._compute_file_value(key, param_value, data_dir)
        elif data_dir and self.type == "path":
            return self._compute_path_value(key, param_value, data_dir)
        elif data_dir and self.type == "object":
            return self._compute_object_value(key, param_value, data_dir)
        elif data_dir and self.type == "list":
            return self._compute_list_value(key, param_value, data_dir)
        else:
            return param_value

class EntryPoint(object):
    """An entry point in an MLproject specification."""
    def __init__(self, name, parameters, command):
        self.name = name
        self.parameters = {k: Parameter(k, v) for (k, v) in parameters.items()}
        self.command = command

    def _validate_parameters(self, user_parameters):
        missing_params = []
        for name in self.parameters:
            if (name not in user_parameters and self.parameters[name].default is None):
                missing_params.append(name)
        if missing_params:
            raise Exception(
                "No value given for missing parameters: %s" %
                ", ".join(["'%s'" % name for name in missing_params]))

    def compute_parameters(self, user_parameters, data_dir):
        if user_parameters is None:
            user_parameters = {}
        self._validate_parameters(user_parameters)
        final_params = {}
        extra_params = {}

        for key, param_obj in self.parameters.items():
            if key in user_parameters and user_parameters[key] is not None and not user_parameters[key] == "":
                value = user_parameters[key]
            else:
                value = self.parameters[key].default
            final_params[key] = param_obj.compute_value(key, value, data_dir)
        for key in user_parameters:
            if key not in final_params:
                extra_params[key] = user_parameters[key]
        return self._sanitize_param_dict(final_params), self._sanitize_param_dict(extra_params)

    def compute_command(self, user_parameters, data_dir):
        params, extra_params = self.compute_parameters(user_parameters, data_dir)
        command_with_params = self.command.format(**params)
        command_arr = [command_with_params]
        # command_arr.extend(["--%s %s" % (key, value) for key, value in extra_params.items()])
        return " ".join(command_arr)

    def _sanitize_param_dict(self,param_dict):
        return {str(key): shlex_quote(str(value)) for key, value in param_dict.items()}

# def _get_conda_command(conda_env_name):
#     activate_conda_env = ["conda activate {} 1>&2".format(conda_env_name)]
#     return activate_conda_env
def _get_conda_command(conda_env_name):
    #  Checking for newer conda versions
    if os.name != 'nt' and ('CONDA_EXE' in os.environ or 'MLFLOW_CONDA_HOME' in os.environ):
        conda_path = _get_conda_bin_executable("conda")
        activate_conda_env = [
            'source {}/../etc/profile.d/conda.sh'.format(os.path.dirname(conda_path))
        ]
        activate_conda_env += ["conda activate {} 1>&2".format(conda_env_name)]
    else:
        activate_path = _get_conda_bin_executable("activate")
        # in case os name is not 'nt', we are not running on windows. It introduces
        # bash command otherwise.
        if os.name != "nt":
            return ["source %s %s 1>&2" % (activate_path, conda_env_name)]
        else:
            return ["conda activate %s" % (conda_env_name)]
    return activate_conda_env
def _get_conda_env_name(conda_env_path, env_id=None):
    conda_env_contents = open(conda_env_path).read() if conda_env_path else ""
    if env_id:
        conda_env_contents += env_id
    return "mlflow-%s" % hashlib.sha1(conda_env_contents.encode("utf-8")).hexdigest()
# def _get_or_create_conda_env(conda_env_path, env_id=None):
#     conda_path = "conda"
#     (_, stdout, _) = process.exec_cmd([conda_path, "env", "list", "--json"])
#     env_names = [os.path.basename(env) for env in json.loads(stdout)['envs']]
#     project_env_name = _get_conda_env_name(conda_env_path, env_id)
#     if project_env_name not in env_names:
#         LOGGER.info('=== Creating conda environment %s ===', project_env_name)
#         if conda_env_path:
#             process.exec_cmd([conda_path, "env", "create", "-n", project_env_name, "--file",
#                               conda_env_path], stream_output=True)
#         else:
#             process.exec_cmd(
#                 [conda_path, "create", "-n", project_env_name, "python"], stream_output=True)
#     return project_env_name
def _get_conda_bin_executable(executable_name):
    conda_home = os.environ.get(MLFLOW_CONDA_HOME)
    if conda_home:
        return os.path.join(conda_home, "bin/%s" % executable_name)
    # Use CONDA_EXE as per https://github.com/conda/conda/issues/7126
    if "CONDA_EXE" in os.environ:
        conda_bin_dir = os.path.dirname(os.environ["CONDA_EXE"])
        return os.path.join(conda_bin_dir, executable_name)
    return executable_name

def _get_or_create_conda_env(conda_env_path, env_id=None):
    conda_path = _get_conda_bin_executable("conda")
    try:
        process.exec_cmd([conda_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise Exception("Could not find Conda executable at {0}. "
                                 "Ensure Conda is installed as per the instructions at "
                                 "https://conda.io/projects/conda/en/latest/"
                                 "user-guide/install/index.html. "
                                 "You can also configure MLflow to look for a specific "
                                 "Conda executable by setting the {1} environment variable "
                                 "to the path of the Conda executable"
                                 .format(conda_path, MLFLOW_CONDA_HOME))
    (_, stdout, _) = process.exec_cmd([conda_path, "env", "list", "--json"])
    env_names = [os.path.basename(env) for env in json.loads(stdout)['envs']]
    project_env_name = _get_conda_env_name(conda_env_path, env_id)
    if project_env_name not in env_names:
        LOGGER.info('=== Creating conda environment %s ===', project_env_name)
        process.exec_cmd([conda_path, "config", "--add", "channels", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/"], stream_output=True)
        process.exec_cmd([conda_path, "config", "--add", "channels", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"],stream_output=True)
        process.exec_cmd([conda_path, "config", "--set", "show_channel_urls","yes"],stream_output=True)
        if conda_env_path:
            process.exec_cmd([conda_path, "env", "create", "-n", project_env_name, "--file",
                              conda_env_path], stream_output=True)
        else:
            process.exec_cmd(
                [conda_path, "create", "-n", project_env_name, "python"], stream_output=True)
    return project_env_name
def _find_mlproject(directory):
    filenames = os.listdir(directory)
    for filename in filenames:
        if filename.lower() == MLPROJECT_FILE_NAME:
            return os.path.join(directory, filename)
    return None

def _run(project_type, project_name,projece_version,task_id,input_parameters):
    data_dir = os.path.join('/home/data',task_id)
    if data_dir is not None and not os.path.exists(data_dir):
        os.makedirs(data_dir)

    storage_dir = os.path.join('/home/storage',task_id)
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    input_parameters['storage_dir'] = storage_dir

    work_dir = '/home/service/'+project_type + '/'+ project_name + '/' + projece_version
    if not os.path.exists(work_dir):
        raise Exception("Could not find subdirectory %s" % (work_dir))
    LOGGER.info("=== work_dir '%s'", work_dir)
    mlproject_path = _find_mlproject(work_dir)
    yaml_obj = {}
    if mlproject_path is not None:
        with open(mlproject_path) as mlproject_file:
            yaml_obj = yaml.safe_load(mlproject_file)
    project_name = yaml_obj.get("name")
    conda_path = yaml_obj.get("conda_env")
    if conda_path:
        conda_env_path = os.path.join(work_dir, conda_path)
        if not os.path.exists(conda_env_path):
            raise Exception("Project specified conda environment file %s, but no such "
                                     "file was found." % conda_env_path)
    # Parse entry points
    entry_points = {}
    for name, entry_point_yaml in yaml_obj.get("entry_points", {}).items():
        parameters = entry_point_yaml.get("parameters", {})
        command = entry_point_yaml.get("command")
        entry_points[name] = EntryPoint(name, parameters, command)

    entry_point_obj = entry_points['main']
    command_args = []
    command_separator = " && "
    if conda_path:
        conda_env_name = _get_or_create_conda_env(conda_env_path)
        command_args += _get_conda_command(conda_env_name)
    LOGGER.info("=== Create Conda Complete")
    commands = []
    commands.append(entry_point_obj.compute_command(input_parameters, data_dir=data_dir))
    command_args += commands
    command_str = command_separator.join(command_args)
    LOGGER.info("=== Running command '%s'", command_str)

    if os.name != "nt":
        process = subprocess.Popen(["bash", "-c", command_str], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir)
        try:
            stdout = ""
            for info in process.communicate():
                stdout = stdout + str(info, encoding="utf-8")
                LOGGER.info(str(info, encoding="utf-8"))
            # if process.wait():
            #     # for i in iter(process.stdout.readline, 'b'):
            #     #     LOGGER.info(i)
            #     res = process.stdout.read()
            LOGGER.info("output:'%s'",stdout)
            LOGGER.info("=== Run (ID '%s') succeeded ===", task_id)

            output_file = os.path.join(storage_dir,"output")
            output_result = {}
            if os.path.exists(output_file):
                # with open(output_file, 'r') as output:
                #     list2 = output.readlines()
                # db = MongoGridFS("")
                # for i in range(0, len(list2)):
                #     content = list2[i].strip('\n')
                #     para_map = content.split('=', 1)
                #     output_result[para_map[0]] = para_map[1]
                #     if os.path.isdir(para_map[1]):
                #         result = []  # 所有的文件
                #         for maindir, subdir, file_name_list in os.walk(para_map[1]):
                #             for filename in file_name_list:
                #                 apath = os.path.join(maindir, filename)  # 合并成一个完整路径
                #                 result.append(apath)
                #         objectIds = []
                #         for file in result:
                #             basename = os.path.basename(file)
                #             objectId = str(db.upLoadFile("fs", file, basename, task_id))
                #             objectIds.append({'fileName': basename, 'fileId': objectId})
                #         output_result[para_map[0]] = objectIds
                #     elif os.path.isfile(para_map[1]):
                #         basename = os.path.basename(para_map[1])
                #         objectId = str(db.upLoadFile("fs",para_map[1],basename,task_id))
                #         output_result[para_map[0]] = {'fileName':basename,'fileId':objectId}
                #     else:
                #         continue
                with open(output_file, 'r') as output:
                    output_result = json.load(output)
                    db = MongoGridFS("")
                    for key in output_result:
                        value = output_result[key]
                        if(isinstance(value, str)):
                            if os.path.isdir(value):
                                result = []  # 所有的文件
                                for maindir, subdir, file_name_list in os.walk(value):
                                    for filename in file_name_list:
                                        apath = os.path.join(maindir, filename)  # 合并成一个完整路径
                                        result.append(apath)
                                objectIds = []
                                for file in result:
                                    basename = os.path.basename(file)
                                    objectId = str(db.upLoadFile("fs", file, basename, task_id))
                                    objectIds.append({'fileName': basename, 'fileId': objectId})
                                output_result[key] = objectIds
                            elif os.path.isfile(value):
                                basename = os.path.basename(value)
                                objectId = str(db.upLoadFile("fs",value,basename,task_id))
                                output_result[key] = {'fileName':basename,'fileId':objectId}
                            else:
                                continue
            output_result['stdout'] = stdout
            return output_result
        except KeyboardInterrupt:
            LOGGER.error("=== Run (ID '%s') interrupted, cancelling run ===", task_id)
            process.cancel()
            raise
        except Exception:
            LOGGER.error("=== Run (ID '%s') failed ===", task_id)
            raise

def ack_message(ch, delivery_tag, connection, result):
    """Note that `ch` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    LOGGER.info('complete')
    response_channel = connection.channel()
    response_channel.exchange_declare(
        exchange='RUNTIME_RESPONSE',
        exchange_type='topic',
        passive=False,
        durable=True,
        auto_delete=False)
    response_channel.basic_publish(
        exchange='RUNTIME_RESPONSE',
        routing_key='RUNTIME_RESPONSE',
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
    LOGGER.info('Thread Delivery tag: %s Message body: %s', delivery_tag, json.loads(body))
    # Sleeping to simulate 10 seconds of work
    body = json.loads(body)
    #print(res)
    # config_file = '/home/LAB/wusj/fastwash_tmp/training/' + 'config_' + res['id']
    # with open(config_file, 'wt') as f:
    #     json.dump(res, f)
    # output_file = '/home/LAB/wusj/fastwash_tmp/training/' + 'config_' + res['id']
    # with open(output_file, 'r') as f:
    #     res = json.load(f)
    # res['status'] = 'TRAINING_TASK_SUCCESS'
    # res = json.dumps(res)
    taskInfo = body['taskInfo']
    projectInfo = body['projectInfo']
    parameter = body['parameter']
    try:
        output = _run(projectInfo['projectType'], projectInfo['projectName'], projectInfo['version'], taskInfo['taskId'], parameter)
        res = {}
        res["taskInfo"] = taskInfo
        res["projectInfo"] = projectInfo
        res["output"] = output
        res["success"] = True
        res = json.dumps(res)
        cb = functools.partial(ack_message, ch, delivery_tag, conn, res)
        conn.add_callback_threadsafe(cb)
    except:
        import traceback, sys
        traceback.print_exc()  # 打印异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error = str(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))  # 将异常信息转为字符串

        output = {"exception": error}
        res = {}
        res["taskInfo"] = taskInfo
        res["projectInfo"] = projectInfo
        res["output"] = output
        res["success"] = False
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--q',dest='queue',type=str)
    args = parser.parse_args()
    print(args.queue)
    queue_name = args.queue
    credentials = pika.PlainCredentials("guest", "guest")
    conn_params = pika.ConnectionParameters(host='10.1.1.63', port=5672, credentials=credentials, socket_timeout=500)
    # Infinite loop
    while True:
        try:
            connection = pika.BlockingConnection(conn_params)
            channel = connection.channel()
            channel.exchange_declare(
                exchange='RUNTIME_REQUEST',
                exchange_type='topic',
                passive=False,
                durable=True,
                auto_delete=False)
            channel.queue_declare(queue=queue_name, auto_delete=False)
            channel.queue_bind(
                queue=queue_name,
                exchange='RUNTIME_REQUEST',
                routing_key=queue_name)
            # channel.basic_qos(prefetch_count=1)
            threads = []
            on_message_callback = functools.partial(on_message, args=(connection, threads))
            channel.basic_consume(queue_name, on_message_callback)

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
