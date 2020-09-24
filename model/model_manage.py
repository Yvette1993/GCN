from functools import reduce, partial
import torch

class MOOMM:
    '''
    Multi-objective optimization model manager
    '''
    

    def __init__(self, model_class_info, model_super_params, device, model_save_dir, optimizer, loss_num):

        assert 'package_name' in model_class_info and 'class_name' in model_class_info, 'Model class information is incomplete!'
        self.model_class_info = model_class_info
        self.model_super_params = model_params
        self.device = device
        self.model_save_path = model_save_dir + '/' + '{}_{}_'.format(self.model_class_info['package_name'],self.model_class_info['class_name'])
        self.optimizer = optimizer
        self.loss_num = loss_num
        self.model_info_pool = {} #
        self.current_model:torch.nn.Module = None
        self.model_counter=0
        self.start()

    def model_init(self): 
        model_class=getattr(__import__(self.model_class_info['package_name']) , self.model_class_info['class_name'])
        temp_model = model_class(**self.model_super_params)
        return temp_model

    def current_model_is_dominated(self, current_loss):
        # 这里用到了 柯里化 functools.partial
        seleted_func = partial(MOOMM.dominated_compare, candidate_loss=current_loss)
        is_dominated_results, dominated_results, model_ids = zip(*[seleted_func(model_info) for model_info in self.model_info_pool.values()])
        if is_dominated_results.count(True) != 0 :
            return True
        else:
            for (model_id, dominated_result) in zip(model_ids, dominated_results):
                if dominated_result:
                    del self.model_info_pool[model_id]
            return False

    def start(self):
        self.current_model = self.model_init
        self.current_model = self.current_model.set_ID(self.model_counter).to(self.device)
        model_info = {}
        model_info['id'] = self.current_model.model_id
        model_info['loss'] = [10e3 for _ in range(self.loss_num)]
        model_info['path'] = self.model_save_path+model_info['id']+'.pth'
        self.model_info_pool[model_info['id']] = model_info
        return self.current_model


    def continue_or_rollback(self, current_loss):
        if self.current_model_is_dominated(current_loss):
            # rollback 随机选一个已有模型
            # 从模型池中随机选择一个进行训练
            # TODO 这里没有添加连续变坏的判断和处理
            model_id = random.choice(list(self.model_info_pool.keys())) #if len(self.model_info_pool)!=0 else self._counter_increment
            checkpoint = torch.load(self.model_info_pool[model_id]['path'])        
            self.current_model.set_ID(model_id)
            self.current_model.load_state_dict(checkpoint['model'])
            self.current_model = self.current_model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        else:
            # continue
            model_info = {}
            if self.current_model.model_id in self.model_info_pool: # 这次结果和上次结果同样好，互不支配
                model_info['id'] = self._counter_increment
                self.current_model.set_ID(model_info['id'])
            else: #说明这一次结果比上一次结果好，上一次结果已经被删除
                model_info['id'] = self.current_model.model_id

            model_info['loss'] = [i for i in current_loss]
            model_info['path'] = self.model_save_path+model_info['id']+'.pth'
            self.model_info_pool[model_info['id']] = model_info
            state = {'model': self.current_model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            torch.save(state, model_info['path'])


    @property
    def _counter_increment(self):
        self.model_counter = self.model_counter+1
        return self.model_counter


    @staticmethod
    def dominated_compare(candidate_loss, existing_model_info):
        loss_info = existing_model_info['loss']
        is_dominated = reduce(lambda x, y: x&y, [i<j for i,j in zip(loss_info, candidate_loss)])
        dominated = reduce(lambda x, y: x&y, [i>j for i,j in zip(loss_info, candidate_loss)])
        return is_dominated, dominated, existing_model_info['id']


    
