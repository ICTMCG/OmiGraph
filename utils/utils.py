from collections import defaultdict
import json
    
class Recorder():
    def __init__(self, args, logger):
        self.max = .0
        self.cur = .0
        self.results = defaultdict(dict)
        self.maxindex = 0
        self.curindex = 0
        self.early_stop = args.early_stop
        self.custom_name = args.custom_name
        self.logger = logger

    def add(self, x):
        self.cur = float(x['mac_f1'])
        self.logger.info(f"Current epoch [{self.curindex}] (mac_f1) = {self.cur}, while former max = {self.max} in epoch[{self.maxindex}]")
        judge = self.judge()
        if judge == 'save':
            self.results = x
        self.curindex += 1
        return judge
    
    def judge(self):
        if self.cur > self.max:
            self.logger.info(f'From {self.max} (mac_f1 in epoch {self.maxindex}) update to {self.cur} (mac f1 in epoch {self.curindex})')
            self.max = self.cur
            self.maxindex = self.curindex
            self.logger.info(f"=== Epoch [{self.curindex}] max(mac_f1)={self.max} ===")
            return 'save'
        if self.curindex - self.maxindex >= self.early_stop:
            return 'esc'
        else:
            return 'continue'
        
    def showfinal(self):
        self.logger.info(f"\n=== {self.custom_name}")
        self.logger.info(f"=== Best epoch: {self.maxindex}, Best val results: {json.dumps(self.results)}")
