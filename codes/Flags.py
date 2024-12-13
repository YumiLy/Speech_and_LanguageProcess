# -*- coding: utf-8 -*-
import tensorflow as tf

def get_flags():
    tf.compat.v1.disable_eager_execution()

    flags = tf.compat.v1.app.flags
    flags.DEFINE_float("lr", 0.0001, "learning rate")
    flags.DEFINE_integer("batch_size", 16, "batch size")
    flags.DEFINE_boolean("is_train", True, "training or testing a model")
    flags.DEFINE_string("train_dir", "train", "path to save model")
    flags.DEFINE_string("model", "ar", "choose baseline model, ar lm or sar")
    flags.DEFINE_string("gpu", "1", "choose gpu")

    FLAGS = flags.FLAGS

    return FLAGS