#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf

worker_hosts = ["111.111.1.111:2222", "111.111.1.111:2222"]
cluster = tf.train.ClusterSpec({"worker": worker_hosts})

tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    server.join()

if __name__ == "__main__":
    tf.app.run()

