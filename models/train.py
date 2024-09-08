import os
import numpy as np
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import h5py
import json
import horovod.torch as hvd # for multi GPU
import argparse

from utils_models import get_device, get_batch_size, loss_criterion, RunningAverage, EarlyStopper
from ffno_model import FFNO_3D
from mifno_model import MIFNO_3D
from dataloaders import GeologyTracesSourceDataset


parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@model_type', type=str, default="MIFNO", help="Architecture used: MIFNO or F-FNO")
parser.add_argument('@S_in', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('@S_in_z', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('@S_out', type=int, default=32, help="Size of the spatial output grid")
parser.add_argument('@T_out', type=int, default=320, help="Number of time steps")
parser.add_argument('@nlayers', type=int, help="Number of layers")
parser.add_argument('@branching_index', type=int, help="Index of the first FFNO block seeing the source")
parser.add_argument('@dv', type=int, help = "Number of channels")
parser.add_argument('@list_dv', type=int, nargs='+', help = "Number of channels in uplfit block + each Fourier block (used only in F-FNO, not in MIFNO)")
parser.add_argument('@list_D1', type=int, nargs='+', help = "Dimensions along the 1st dimension after each block")
parser.add_argument('@list_D2', type=int, nargs='+', help = "Dimensions along the 2nd dimension after each block")
parser.add_argument('@list_D3', type=int, nargs='+', help = "Dimensions along the 3rd dimension after each block")
parser.add_argument('@list_M1', type=int, nargs='+', help = "Number of modes along the 1st dimension after each block")
parser.add_argument('@list_M2', type=int, nargs='+', help = "Number of modes along the 2nd dimension after each block")
parser.add_argument('@list_M3', type=int, nargs='+', help = "Number of modes along the 3rd dimension after each block")
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of validation samples")
parser.add_argument('@batch_size', type=int, default=16, help = 'batch size')
parser.add_argument('@source_orientation', type=str, default='angle', help="angle or moment")
parser.add_argument('@normalize_source', action='store_true', help='Whether to normalize the source position and the angles (if applicable)')
parser.add_argument('@normalize_traces', action='store_true', help='Whether to normalize the traces')
parser.add_argument('@padding', type=int, default=0, help = "Number of pixels for padding on each side of x and y")
parser.add_argument('@epochs', type=int, default=350, help = 'Number of epochs')
parser.add_argument('@learning_rate', type=float, default=0.0006, help='learning rate')
parser.add_argument('@loss_weights', type=float, nargs='+', default = [1.0, 0.0], help = "Weight of L1 loss, L2 loss")
parser.add_argument('@dir_data_train', type=str, nargs='+', default=['../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_train'], help="Name of folders with training data")
parser.add_argument('@dir_data_val', type=str, nargs='+', default=['../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_val'], help="Name of folders with training data")
parser.add_argument('@dir_logs', type=str, default='../logs/', help="Path to folder to store loss and models")
parser.add_argument('@additional_name', type=str, default="", help="string to add to the configuration name for saved outputs")
parser.add_argument('@restart_model', type=str, default="", help="Path to the model to use as initialization")
parser.add_argument('@start_epoch', type=int, default=0, help="Epoch to start, >0 if initializing with a trained model")
parser.add_argument('@seed', type=int, default=0, help="Seed to initialize pytorch")
options = parser.parse_args().__dict__

# increase the reproducibility between devices and runs
torch.manual_seed(options['seed'])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


dir_logs = options['dir_logs']                    
batch_size = options['batch_size']
Ntrain = options['Ntrain']
Nval = options['Nval']
learning_rate = options['learning_rate']
weight_decay = 0.00001
patience = 60 # number of epochs to wait for the validation loss to decrease before stopping the training
epochs = options['epochs']
start_epoch = options['start_epoch']
loss_weights = options['loss_weights']
restart_model = options['restart_model']


# Model parameters
model_type = options['model_type']
source_orientation = options['source_orientation']
normalize_source = options['normalize_source']
if normalize_source:
    transform_position=[9600,9600,-9600]
else:
    transform_position = None
    
if options['normalize_traces']:
    normalize_traces = 'distance_Vs'
else:
    normalize_traces = None
nlayers = options['nlayers']
S_in = options['S_in']
S_in_z = options['S_in_z']
S_out = options['S_out']
T_out = options['T_out']
padding = options['padding']



if __name__ == '__main__':
    ### INITIALIZE HOROVOD
    hvd.init()
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
    verbose = 2 if hvd.rank() == 0 else 0

    train_data = GeologyTracesSourceDataset(options['dir_data_train'], S_in=S_in, S_in_z=S_in_z,
                                            S_out=S_out, T_out=T_out,
                                            transform_a='normal', N=Ntrain, orientation=source_orientation, 
                                            transform_position=transform_position, transform_angle='unit',
                                            transform_traces=normalize_traces)
    val_data = GeologyTracesSourceDataset(options['dir_data_val'], S_in=S_in, S_in_z=S_in_z,
                                          S_out=S_out, T_out=T_out,
                                          transform_a='normal', N=Nval, orientation=source_orientation,
                                          transform_position=transform_position, transform_angle='unit',
                                          transform_traces=normalize_traces)
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    shuffle=True,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=2,
                                               sampler=train_sampler
    )
    
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)

    if verbose:
        print(f'using {len(train_data)} training data and {len(val_data)} validation data')

    ### MODEL
    device = get_device()

    dv = options['dv']
    list_D1 = np.array(options['list_D1']).astype(int)
    list_D2 = np.array(options['list_D2']).astype(int)
    list_D3 = np.array(options['list_D3']).astype(int)
    list_M1 = np.array(options['list_M1']).astype(int)
    list_M2 = np.array(options['list_M2']).astype(int)
    list_M3 = np.array(options['list_M3']).astype(int)

    assert nlayers == list_D1.shape[0]

    name_config = f"{model_type}3D-{source_orientation}-dv{dv}-{nlayers}layers-S{S_in}-T{T_out}-"\
        f"learningrate{str(learning_rate).replace('.','p')}-Ntrain{Ntrain}-batchsize{batch_size}"
    if normalize_source:
        name_config += "-normedsource"
    if normalize_traces is not None:
        name_config += "-normedtraces"
    name_config += options['additional_name']

    ## Build the initial model
    if model_type == 'MIFNO':
        branching_index = options['branching_index']
        if source_orientation == 'angle':
            input_dim = 4 # a, x, y, z
            output_dim = 1
            source_dim = 6 # 3 coordinates + 3 angles
            model = MIFNO_3D(list_D1, list_D2, list_D3,
                             list_M1, list_M2, list_M3, dv,
                            input_dim=input_dim, output_dim=output_dim, source_dim=source_dim, 
                            n_layers=nlayers, branching_index=branching_index, padding=padding
            )

        elif source_orientation == 'moment':
            input_dim = 4 # a, x, y, z
            output_dim = 1
            source_dim = 9 # 3 coordinates + 6 moment tensor components
            model = MIFNO_3D(list_D1, list_D2, list_D3,
                             list_M1, list_M2, list_M3, dv,
                            input_dim=input_dim, output_dim=output_dim, source_dim=source_dim,
                            n_layers=nlayers, branching_index=branching_index, padding=padding
            )

        else:
            raise Exception(f"source {source_orientation} is not defined")
    
    if model_type == 'FFNO':
        list_dv = np.array(options['list_dv']).astype(int)
        if source_orientation == 'angle':
            input_dim = 10 # a, x, y, z, x_s, y_s, z_s, strike, dip, rake
            output_dim = 1
            model = FFNO_3D(list_D1, list_D2, list_D3,
                            list_M1, list_M2, list_M3, dv, list_dv,
                            input_dim=input_dim, output_dim=output_dim, list_width=list_dv,
                            n_layers=nlayers, padding=padding
            )

        elif source_orientation == 'moment':
            input_dim = 13 # a, x, y, z, x_s, y_s, z_s, Mxx, Myy, Mzz, Mxy, Mxz, Myz
            output_dim = 1
            model = FFNO_3D(list_D1, list_D2, list_D3,
                            list_M1, list_M2, list_M3, dv,
                            input_dim=input_dim, output_dim=output_dim, list_width=list_dv,
                            n_layers=nlayers, padding=padding
            )

        else:
            raise Exception(f"source {source_orientation} is not defined")
    
    
    if hvd.rank() == 0:
        if model_type == 'MIFNO':
            with open(f"{dir_logs}models/architecture-{name_config}-epochs{epochs}.json", mode='w', encoding='utf-8') as param_file:
                json.dump({'architecture':{'model type':model_type,
                                           'input dim':input_dim,
                                           'output dim':output_dim,
                                           'source dim':source_dim,
                                           'nlayers':nlayers,
                                           'branching index':branching_index,
                                           'dv':dv,
                                           'list D1':list(list_D1.astype(float)),
                                           'list D2':list(list_D2.astype(float)),
                                           'list D3':list(list_D3.astype(float)),
                                           'list M1':list(list_M1.astype(float)),
                                           'list M2':list(list_M2.astype(float)),
                                           'list M3':list(list_M3.astype(float)),
                                           'padding':padding},
                           
                           'optimization':{'weight decay':weight_decay,
                                           'patience':patience,
                                           'epochs':epochs,
                                           'learning rate':learning_rate,
                                           'loss weights (L1, L2)':loss_weights
                           },
                           
                           'data':{'Ntrain':Ntrain,
                                   'Nval':Nval,
                                   'batch size':batch_size,
                                   'source orientation':source_orientation,
                                   'normalize source':normalize_source,
                                   'transform position':transform_position,
                                   'normalize traces':normalize_traces,
                                   'data train':options['dir_data_train'],
                                   'data val':options['dir_data_val']
                           }
                }, param_file)

        if model_type == 'FFNO':
            with open(f"{dir_logs}models/architecture-{name_config}-epochs{epochs}.json", mode='w', encoding='utf-8') as param_file:
                json.dump({'architecture':{'model type':model_type,
                                           'input dim':input_dim,
                                           'output dim':output_dim,
                                           'nlayers':nlayers,
                                           'list dv':list(list_dv.astype(float)),
                                           'list D1':list(list_D1.astype(float)),
                                           'list D2':list(list_D2.astype(float)),
                                           'list D3':list(list_D3.astype(float)),
                                           'list M1':list(list_M1.astype(float)),
                                           'list M2':list(list_M2.astype(float)),
                                           'list M3':list(list_M3.astype(float)),
                                           'padding':padding},
                           
                           'optimization':{'weight decay':weight_decay,
                                           'patience':patience,
                                           'epochs':epochs,
                                           'learning rate':learning_rate,
                                           'loss weights (L1, L2)':loss_weights
                           },
                           
                           'data':{'Ntrain':Ntrain,
                                   'Nval':Nval,
                                   'batch size':batch_size,
                                   'source orientation':source_orientation,
                                   'normalize source':normalize_source,
                                   'transform position':transform_position,
                                   'normalize traces':normalize_traces,
                                   'data train':options['dir_data_train'],
                                   'data val':options['dir_data_val']
                           }
                }, param_file)

    
    if restart_model != "": # initialize weights with a trained model
        model.load_state_dict(torch.load(restart_model, map_location=device))
        if verbose:
            print('model loaded!')
  
    if torch.cuda.device_count() >= 1:
        NGPUs = torch.cuda.device_count()
    else:
        NGPUs = 0
    if verbose:
        print(f'Using {NGPUs} GPUs for training')
    model.to(device)


    if verbose:
        nb_params = 0
        for name, layer in model.named_parameters():
            nb_params += torch.numel(layer)
        print(f'Total nb of parameters : {nb_params:.2e}')

    # Store losses history
    train_history = {'loss_relative':[], 'loss_absolute':[]}
    val_history = {'loss_relative':[], 'loss_absolute':[]}
    best_loss = np.inf

    
    ### OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=learning_rate*hvd.size(), betas=(0.9, 0.999))
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True) 
    early_stopper = EarlyStopper(patience=patience, min_delta=0.0001)

    
    ### TRAINING       
    for ep in range(start_epoch, epochs):
        t1 = timeit.default_timer()
        model.train()
        train_losses_relative = RunningAverage()
        train_losses_absolute = RunningAverage()
        
        # training
        for _ in train_loader:
            a = _[0].to(device)
            uE = _[1].to(device)
            uN = _[2].to(device)
            uZ = _[3].to(device)
            s = _[4].to(device)
            
            outE, outN, outZ = model(a, s)
            loss_rel = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=True)
            loss_abs = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=False)
            
            train_losses_relative.update(loss_rel.item(), get_batch_size(a))
            train_losses_absolute.update(loss_abs.item(), get_batch_size(a))
            
            # compute gradients and update parameters
            optimizer.zero_grad()
            loss_rel.backward()
            optimizer.step()
        
        train_history['loss_relative'].append(train_losses_relative.avg)
        train_history['loss_absolute'].append(train_losses_absolute.avg)

        # validation
        if hvd.rank() == 0:
            model.eval()
            with torch.no_grad():
                val_losses_relative = RunningAverage()
                val_losses_absolute = RunningAverage()

                # training
                for _ in val_loader:
                    a = _[0].to(device)
                    uE = _[1].to(device)
                    uN = _[2].to(device)
                    uZ = _[3].to(device)
                    s = _[4].to(device)
                    outE, outN, outZ = model(a, s)
                    loss_rel_val = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=True)
                    loss_abs_val = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=False)
                    
                    val_losses_relative.update(loss_rel_val.item(), get_batch_size(a))
                    val_losses_absolute.update(loss_abs_val.item(), get_batch_size(a))

                val_history['loss_relative'].append(val_losses_relative.avg)
                val_history['loss_absolute'].append(val_losses_absolute.avg)
            
            lr_scheduler.step(val_losses_relative.avg)

            t2 = timeit.default_timer()
            print(f'Epoch {ep+1}/{epochs}: {t2-t1:.2f}s - Training loss = {train_losses_relative.avg:.5f} - Validation loss = {val_losses_relative.avg:.5f}'\
                  f' - Training accuracy = {train_losses_absolute.avg:.5f} - Validation accuracy = {val_losses_absolute.avg:.5f}')

            # save the model
            if val_losses_relative.avg < best_loss:
                best_loss = val_losses_relative.avg
                torch.save(model.state_dict(), f'{dir_logs}models/bestmodel-{name_config}-epochs{epochs}.pt')
   
            if early_stopper.early_stop(val_losses_relative.avg):
                break

            # save intermediate losses
            if ep%2==0:
                with h5py.File(f'{dir_logs}loss/loss-{name_config}-epoch{ep}on{epochs}.h5', 'w') as f:
                    f.create_dataset('epochs', data=np.arange(start_epoch, ep+1))
                    f.create_dataset('train_loss_relative', data=train_history['loss_relative'])
                    f.create_dataset('train_loss_absolute', data=train_history['loss_absolute'])
                    f.create_dataset('val_loss_relative', data=val_history['loss_relative'])
                    f.create_dataset('val_loss_absolute', data=val_history['loss_absolute'])

                # remove the previous losses saved
                if ep>start_epoch+1:
                    os.remove(f'{dir_logs}loss/loss-{name_config}-epoch{ep-2}on{epochs}.h5')

                last_epoch_saved = ep # to remove the last intermediate save at the end
                

    if hvd.rank() == 0:
        # save the final loss
        with h5py.File(f'{dir_logs}loss/loss-{name_config}-epochs{ep+1}.h5', 'w') as f:
            f.create_dataset('epochs', data=np.arange(start_epoch, ep+1))
            f.create_dataset('train_loss_relative', data=train_history['loss_relative'])
            f.create_dataset('train_loss_absolute', data=train_history['loss_absolute'])
            f.create_dataset('val_loss_relative', data=val_history['loss_relative'])
            f.create_dataset('val_loss_absolute', data=val_history['loss_absolute'])
            
        os.remove(f'{dir_logs}loss/loss-{name_config}-epoch{last_epoch_saved}on{epochs}.h5')
            
        # rename the model in case the training stopped before the final epoch
        if ep!= epochs-1:
            os.rename(f'{dir_logs}models/bestmodel-{name_config}-epochs{epochs}.pt', f'{dir_logs}models/bestmodel-{name_config}-epochs{ep+1}.pt')
            os.rename(f'{dir_logs}models/architecture-{name_config}-epochs{epochs}.json', f'{dir_logs}models/architecture-{name_config}-epochs{ep+1}.json')
