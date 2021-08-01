from TEPS_worker import main as train_function
from OOP_config import init_config 
from hpbandster.core.worker import Worker
class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        print("Init worker") 
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)
        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        res = train_function(config,int(budget))

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })
    
    @staticmethod
    def get_configspace():
        return init_config()
