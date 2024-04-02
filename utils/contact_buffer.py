import torch

class ContactBuffer:
    """后面更好的结果会替换前面的结果，但是前面的结果不会被删除，只是被覆盖了
    """
    def __init__(self, buffer_size, content_dim=3, device=torch.device('cpu')):
        '''
        512, 8
        '''

        self.buffer_size = buffer_size
        self.content_dim = content_dim
        self.device = device
        self.buffer = torch.zeros((buffer_size, content_dim), device=device)  #这里存放的应该是点云
        self.top = 0  #这里表明的是得分较高的几个点
        
    def insert(self, batch):
        '''
        batch: {tensor} N x 8, 一般只有一个输入进来
        '''
        # print('batch.shape', batch)
        
        batch_size = batch.shape[0]
        start_random_insert = batch_size

        #这里检查还有几个点，如果还有空间，就直接插入
        if self.top+batch_size <= self.buffer_size:
            self.buffer[self.top:self.top+batch_size].copy_(batch)
            self.top += batch_size
        elif self.top < self.buffer_size:
            avl_len = self.buffer_size - self.top
            self.buffer[self.top:self.buffer_size].copy_(batch[:avl_len])
            start_random_insert = avl_len
            self.top += avl_len
        else:
            start_random_insert = 0
        
        num_insert = batch_size - start_random_insert
        if num_insert > 0 :
            i,idx = torch.sort(self.buffer[:, -1])  #对最后一列进行排序，看样子这里的最后一列应该是成绩，最后一列是rew——buf
            i,rank = torch.sort(idx)  #这里由于得分的高低，感觉这个和上一个重复了, 确实是重复了
            possibility = torch.softmax(rank.float(), dim=-1)   #替换成0-1,这里依据排序，换成了概率
            replace_idx = torch.multinomial(possibility, num_insert)  #然后根据概率，把对应的点的索引找出来
            self.buffer[replace_idx] = batch[start_random_insert:]  #把新的概率点放进去
            
    def all(self):

        return self.buffer[:self.top, :]
        
    def print(self):

        print(self.buffer[:self.top])
    
    def save(self, path):

        torch.save(self.buffer[:self.top], path)