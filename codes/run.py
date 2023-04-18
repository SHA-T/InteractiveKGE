#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import pathlib
import time
import random
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_interactions import panhandler, zoom_factory
from multiprocessing import Process, Pipe
from collections import deque

import torch
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


PLT_INTERACTION_WINDOW = 0.1


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--visualize', action='store_true', help='Visualize embeddings (Interactive Mode)')

    parser.add_argument('--do_pretrain', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)


def override_config(args):
    """
    Override model and data configuration
    """
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args, step=0):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'+str(f"{step:06d}")), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id):
    """
    Read triples and map them into ids.
    """
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    """
    Write logs to checkpoint and console
    """
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs
    """
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args, conn):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.visualize:
        args.do_pretrain = False
        args.hidden_dim = 2
        args.save_checkpoint_steps = 1
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    if args.do_pretrain:
        pretrain_triples = read_triple(os.path.join(args.data_path, 'pretrain.txt'), entity2id, relation2id)
        logging.info('#train: %d' % len(pretrain_triples))
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_pretrain:
        # Set pretraining dataloader iterator with the pretrain-mock-data
        pretrain_dataloader_head = DataLoader(
            TrainDataset(pretrain_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        pretrain_dataloader_tail = DataLoader(
            TrainDataset(pretrain_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        pretrain_iterator = BidirectionalOneShotIterator(pretrain_dataloader_head, pretrain_dataloader_tail)

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training

    if args.do_pretrain:
        logging.info('____________________')
        logging.info('Start Pretraining...')

        training_logs = []

        # Pretraining Loop
        for step in range(init_step, args.max_steps//10):
            log = kge_model.train_step(kge_model, optimizer, pretrain_iterator, args, pretrain_finished=False)

            training_logs.append(log)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_train:
        logging.info('____________________')
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        # Training Loop
        for step in range(init_step, args.max_steps):

            if args.visualize:
                # From parent
                if conn.poll():
                    id_xy_local = conn.recv()
                    # Overwrite embedding of dragged entity
                    with torch.no_grad():
                        kge_model.entity_embedding[id_xy_local[0]] = torch.from_numpy(np.asarray(id_xy_local[1]))
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args, pretrain_finished=True)

            if args.visualize:
                # To parent
                conn.send(kge_model.entity_embedding.detach().numpy())
                # Makes sure pipe won't get filled more before it was emptied
                # and slows visualization a bit down for the human eye
                time.sleep(PLT_INTERACTION_WINDOW * 1.5)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args, step)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('____________________')
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('____________________')
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('____________________')
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


def path_object_to_str(obj):
    if isinstance(obj, pathlib.Path):
        return str(obj)


class DraggablePoint:

    lock = None     # only one can be animated at a time

    def __init__(self, point, entity_id):
        self.entity_id = entity_id
        self.point = point
        self.press = None
        self.background = None

    def connect(self):
        """
        Connect to all the events we need
        """
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes:
            return
        if DraggablePoint.lock is not None:
            return
        contains, attrd = self.point.contains(event)
        if not contains:
            return
        self.press = self.point.center, event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes:
            return
        # We only get this far if this point is pressed and then mouse moves
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """
        On release, we reset the press data
        """
        if DraggablePoint.lock is not self:
            return

        # Update embedding to the destination where the circle was dragged to
        global id_xy
        id_xy = (self.entity_id, (event.xdata, event.ydata))

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

    def disconnect(self):
        """
        Disconnect all the stored connection ids
        """
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
        

if __name__ == '__main__':
    # Tuple to be sent through multiprocessing.Pipe to child process, where the KGE training happens
    # First element: ID of the entity that is moved via drag & drop | Second element: coordinates of placement
    id_xy = (0, (0.0, 0.0))

    args = parse_args()
    conn_parent, conn_child = Pipe()

    if not args.visualize:
        main(args, conn_child)
    else:
        p = Process(target=main, args=(args, conn_child))
        p.start()

        # Create annotations from entity labels
        labels = pd.read_csv(f"{args.data_path}/entities.dict", sep='\t', header=None, names=['idx', 'entity'],
                             index_col=0)
        labels = labels.entity.tolist()
        annotations = len(labels) * [plt.Annotation("", (0, 0))]

        # Create colors for plot
        colors = []
        for i in range(len(labels)):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        # Receive initial embeddings from child process (waits until something comes)
        emb = conn_parent.recv()

        # Configure plot
        plt.ion()
        fig = plt.figure(num='InteractiveKGE', figsize=(6.8, 6.8))
        ax = fig.add_subplot(111)
        plt.title("(Use the mouse-wheel to zoom in and out. \n"
                  "Change the embeddings by dragging the points with the left mouse button.)")
        if args.data_path == "data/countries_neighb_UsaSpaDen":
            plt.suptitle("Neighbors of USA, Spain and Denmark by 'countries_S1' dataset")
        else:
            plt.suptitle("InteractiveKGE")

        # Set initial limits manually, since ax will be populated with circles
        min_lim, max_lim = np.min(emb), np.max(emb)
        max_lim_factor = 1.6
        min_lim_factor = max_lim_factor if (min_lim < 0) else 1.0/max_lim_factor
        max_lim *= max_lim_factor
        min_lim *= min_lim_factor
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)

        # Create circles and annotate
        drs = []
        circles = []
        r1 = (max_lim - min_lim) / 50.0
        r2 = (max_lim - min_lim) / 70.0
        r3 = (max_lim - min_lim) / 80.0
        r4 = (max_lim - min_lim) / 100.0
        r5 = (max_lim - min_lim) / 120.0
        r6 = (max_lim - min_lim) / 140.0
        for i in range(len(emb)):
            circles.append(patches.Circle(tuple(emb[i]), r1, fc=colors[i], alpha=1.0))
            annotations[i] = ax.annotate(labels[i], (emb[i, 0] + r2, emb[i, 1] + r2))
            annotations[i].set_color(colors[i])

        # Add circles to ax
        for i, circ in enumerate(circles):
            ax.add_patch(circ)
            dr = DraggablePoint(circ, i)
            dr.connect()
            drs.append(dr)

        plt.show()

        # Mouse-Wheel Zooming
        disconnect_zoom = zoom_factory(ax)
        pan_handler = panhandler(fig)

        # FIFO-Ringbuffer for historical plots
        maxlen = 6
        q = deque(maxlen=maxlen)

        id_xy_old = id_xy

        try:
            while p.is_alive():  # while process is still alive

                # SEND embedding that was changed via drag-and-drop to child process
                if id_xy != id_xy_old:
                    conn_parent.send(id_xy)
                    id_xy_old = id_xy

                # Only RECEIVE if something was sent to this end of the pipe
                if conn_parent.poll():
                    # If RECEIVE (=embedding changed), update historical positions in ringbuffer
                    q.append([])
                    for i in range(len(emb)):
                        q[-1].append(patches.Circle(tuple(emb[i]), r3, fc=colors[i], alpha=0.5))
                        ax.add_patch(q[-1][i])
                        if len(q) > 1:
                            q[-2][i].set(radius=r4, alpha=0.4)
                            if len(q) > 2:
                                q[-3][i].set(radius=r5, alpha=0.3)
                                if len(q) > 3:
                                    q[-4][i].set(radius=r6, alpha=0.2)
                                    if len(q) == maxlen:
                                        q[0][i].remove()
                    # RECEIVE updated embeddings from child process
                    emb = conn_parent.recv()

                # Update coordinates of circles and annotations
                for i, circ in enumerate(circles):
                    circ.center = tuple(emb[i])
                    annotations[i].set_position((emb[i, 0] + r2, emb[i, 1] + r2))

                fig.canvas.draw_idle()
                plt.pause(PLT_INTERACTION_WINDOW)

            plt.waitforbuttonpress()

        except KeyboardInterrupt:
            plt.close()
            p.terminate()
            p.close()
            print("\nProgram terminated.\n")

        p.join()
        p.close()
