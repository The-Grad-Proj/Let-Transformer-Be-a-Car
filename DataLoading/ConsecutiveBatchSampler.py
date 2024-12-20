#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import Sampler
import random



############### A subclass that inherits from Sampler class in torch.utils.data.sampler.py #######################################################
class ConsecutiveBatchSampler(Sampler):
    
    def __init__(self, data_source, batch_size, seq_len, drop_last=False, shuffle=True, use_all_frames=False):
        r""" Sampler to generate consecutive Batches
        
        Args:
            data_source: Source of data
            batch_size: Size of batch
            seq_len: Number of frames in each sequence (used for context for prediction)
            drop: Wether to drop the last incomplete batch
            shuffle: Wether to shuffle the data
        Return:
            List of iterators, size: [batch_size x seq_len x n_channels x height x width]
        """

        ########################### Call the __init__ function in superclass Sampler #############################
        super(ConsecutiveBatchSampler, self).__init__(data_source) ## Warning if data_source is passed 
        ##########################################################################################################


        self.data_source = data_source
        
        assert seq_len >= 1, "Invalid batch size: {}".format(seq_len) ## Raise an error if seq_len >= 1
        self.seq_len = seq_len
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.use_all_frames_ = use_all_frames
    
    def __iter__(self): # Provide a way to iterate over indices or lists of indices (batches) of dataset elements
        
        data_size = len(self.data_source) ## Determines how many total data points are there in the data source
        
        if self.use_all_frames_:
            start_indices = list(range(data_size))
        else:
            start_indices = list(range(1, data_size, self.seq_len)) #[1, 6, 11, 16, 21, ...]
            
        if self.shuffle:
            random.shuffle(start_indices) ## Shuffle start_indices list if True
        
        batch = []
        for idx, ind in enumerate(start_indices): ## Index will be stored in idx ## Value will be stored in ind
            if data_size - idx < self.batch_size and self.drop_last: ## If data remaining is less than batch size, then its a incomplete batch
                break
                
            seq = []
            if ind + 1 < self.seq_len:
                seq.extend([0]*(self.seq_len - ind - 1) + list(range(0, ind+1)))
            else:
                seq.extend(list(range(ind-self.seq_len+1, ind+1)))
            
            batch.append(seq)
            
            if len(batch) == self.batch_size or idx == data_size - 1:
               # print(batch)
                yield batch ## Return batch and without terminating the function and waits for next function call 
                batch = []

    
    def __len__(self): # Returns the length of the returned iterators
        length = len(self.data_source)
        batch_size = self.batch_size
        
        if length % batch_size == 0 or self.drop_last:
            return length // batch_size
        
        return length // batch_size + 1

