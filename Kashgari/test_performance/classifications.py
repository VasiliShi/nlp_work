#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/8/29 11:16 上午
# File    : classifications.py
# Project : Kashgari

import logging
import os
import time
from datetime import datetime
from typing import Type

import pandas as pd
import tensorflow as tf

from kashgari.callbacks import EvalCallBack
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.classification import ABCClassificationModel
from kashgari.tasks.classification import ALL_MODELS
from examples.tools import get_bert_path

log_root = "tf_dir/classification/" + datetime.now().strftime("%m%d-%H:%M")


class ClassificationPerformance:
    MODELS = ALL_MODELS

    def run_with_model_class(self, model_class: Type[ABCClassificationModel], epochs: int):
        bert_path = get_bert_path()

        train_x, train_y = SMP2018ECDTCorpus.load_data('train')
        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        test_x, test_y = SMP2018ECDTCorpus.load_data('test')

        bert_embed = BertEmbedding(bert_path)
        model = model_class(bert_embed)

        log_path = os.path.join(log_root, model_class.__name__)
        file_writer = tf.summary.create_file_writer(log_path + "/metrics")
        file_writer.set_as_default()
        callbacks = [EvalCallBack(model, test_x, test_y, step=1)]

        model.fit(train_x, train_y, valid_x, valid_y, epochs=epochs, callbacks=callbacks)

        report = model.evaluate(test_x, test_y)
        del model
        del bert_embed
        return report

    def run(self, epochs=10):
        logging.basicConfig(level='DEBUG')
        reports = []
        for model_class in self.MODELS:
            logging.info("=" * 80)
            logging.info("")
            logging.info("")
            logging.info(f" Start Training {model_class.__name__}")
            logging.info("")
            logging.info("")
            logging.info("=" * 80)
            start = time.time()
            report = self.run_with_model_class(model_class, epochs=epochs)
            time_cost = time.time() - start
            reports.append({
                'model_name': model_class.__name__,
                "epoch": epochs,
                'f1-score': report['f1-score'],
                'precision': report['precision'],
                'recall': report['recall'],
                'time': f"{int(time_cost // 60):02}:{int(time_cost % 60):02}"
            })

        df = pd.DataFrame(reports)
        print(df.to_markdown())


if __name__ == '__main__':
    p = ClassificationPerformance()
    p.run()
