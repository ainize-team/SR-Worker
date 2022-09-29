from kombu import Queue

task_acks_late = True
worker_prefetch_multiplier = 1
task_queues = [Queue(name="sr")]
task_ignore_result = True
