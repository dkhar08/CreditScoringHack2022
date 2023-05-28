#save and load functions for weights exchanges between autoencoder and classifier

from tqdm.notebook import tqdm
import os
import torch
import numpy as np
import bz2
from bz2 import BZ2Compressor
import gc
import shutil
import random
from models import Net


def saver_state(model, PATH):
    torch.save({
            'lstm_state_dict': model.lstm.state_dict(),
            'embedding_state_dict': model.embedding_layers.state_dict()
            }, PATH)
    
def loader_state(PATH):
    model = Net()
    
    checkpoint = torch.load(PATH)
    model.lstm.load_state_dict(checkpoint['lstm_state_dict'])
    model.embedding_layers.load_state_dict(checkpoint['embedding_state_dict'])
    
    return model


#Unarchived data allocates a lot of RAM. Function here execute RAM-positive reshuffle between epoches.
def shuffle(data_part="train", do_compression = False):
    avaliable_chunks = list(range(len(os.listdir(os.path.join("nn_data", data_part)))))
    Ninchunk = 250000
    ich = np.hstack([np.repeat(np.array([i], dtype=np.uint8), repeats=Ninchunk) for i in avaliable_chunks])
    np.random.shuffle(ich)
    path = os.path.join("nn_data","tmp")
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(os.path.join("nn_data","tmp"))
    compressors = [BZ2Compressor() for _ in avaliable_chunks]
    files = [open(os.path.join("nn_data","tmp","chunk"+str(i)+".bz2"), "wb") for i in avaliable_chunks]
    
    #for each string determine new chunk
    global_i = 0
    isfirst = [True for _ in avaliable_chunks]
    for chunk_i in tqdm(avaliable_chunks):
        with open(os.path.join("nn_data",data_part,"chunk"+str(chunk_i)+".bz2"), "rb") as f:
            lines = compressed_read(f, do_compression = do_compression)
            lines = lines.split('\n')
        for l in lines:
            tochunk = ich[global_i]
            global_i += 1

            if(isfirst[tochunk]):
                compressed_write(l, files[tochunk], compressors[tochunk].compress, do_compression = do_compression)
                isfirst[tochunk] = False
            else:
                compressed_write('\n'+l, files[tochunk], compressors[tochunk].compress, do_compression = do_compression)

        gc.collect()

    for i in range(len(avaliable_chunks)):
        if(do_compression):
            files[i].write(compressors[i].flush())
        files[i].close()
        
    #shuffle each chunk
    for chunk_i in tqdm(avaliable_chunks):
        with open(os.path.join("nn_data","tmp","chunk"+str(chunk_i)+".bz2"), "rb") as f:
            lines = compressed_read(f, do_compression = do_compression)
            lines = lines.split('\n')

        random.shuffle(lines)
        st = '\n'.join(lines)

        with open(os.path.join("nn_data","tmp","chunk"+str(chunk_i)+".bz2"), "wb") as f:
            compressed_write(st, f, bz2.compress, do_compression = do_compression)
    #rewrite train
    path = os.path.join("nn_data",data_part)
    shutil.rmtree(path)
    os.rename(os.path.join("nn_data","tmp"), os.path.join("nn_data",data_part))

#Hard drive-friendly read and write
def compressed_read(file, do_compression=True):
    if(do_compression):
        return bz2.decompress(file.read()).decode('ascii')
    else:
        return file.read().decode('ascii')
    
def compressed_write(line, file, compressor, do_compression=True):   
    if(do_compression):
        bs = compressor(bytearray(line, encoding='ascii'))
    else:
        bs = bytearray(line, encoding='ascii')
    if(len(bs)>0):
        file.write(bs)


#if not flag than do not perform compression due to speed reasons
def decompress(part = "train"):
    for chunk_i in tqdm(range(len(os.listdir(os.path.join("nn_data",part) )))):
        with open(os.path.join("nn_data",part,"chunk"+str(chunk_i)+".bz2"), "rb") as f:
            lines = compressed_read(f)
            
        with open(os.path.join("nn_data", part,"chunk"+str(chunk_i)+".bz2"), "wb") as f:
            compressed_write(lines, f, bz2.compress, False)


def renumerate_chunks(chunks, part="train"):
    for o, n in zip(chunks, range(len(chunks))):
        shutil.move(os.path.join("nn_data",part,"chunk"+str(o)+".bz2"), os.path.join("nn_data", part,"chunk"+str(n)+"_tmp.bz2"))
    for i in range(len(chunks)):
        shutil.move(os.path.join("nn_data",part,"chunk"+str(i)+"_tmp.bz2"), os.path.join("nn_data",part,"chunk"+str(i)+".bz2"))























