#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

CPU_DEVICE = 'CPU'
CPU_EXTENSION = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ## Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, console_output=True):
        ### load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)

        if not all_layers_supported(self.plugin, self.network, console_output=console_output):
            if console_output:
                print('### Adding cpu extension...')
            self.plugin.add_extension(CPU_EXTENSION, CPU_DEVICE)
            if not all_layers_supported(self.plugin, self.network, console_output=console_output):
                if console_output:
                    print('ERROR: Not all layers supported with extension!')
                return

        self.exec_network = self.plugin.load_network(self.network, CPU_DEVICE)
        # if console_output:
        print('### Network loaded successfully!')

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return 
    

    def get_input_shape(self):
        ### Return the shape of the input layer ###
    
        input_shapes = {}
        for inp in self.network.inputs:
            input_shapes[inp] = (self.network.inputs[inp].shape)
        return input_shapes
    

    def exec_net(self, net_input):
        ###  Start an asynchronous request ###
        self.infer_request_handle = self.exec_network.start_async(
                request_id=0, 
                inputs=net_input)
        return 


    def wait(self):
        ### Wait for the request to be complete. ###
        status = self.infer_request_handle.wait()
        return status
    

    def get_output(self):
        ### Extract and return the output results
        res = self.infer_request_handle.outputs[self.output_blob]
        return res


def all_layers_supported(engine, network, console_output=False):
    ###  check if all layers are supported
    layers_supported = engine.query_network(network, device_name='CPU')
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False
            if console_output:
                print('### Layer', l, 'is not supported')
    if all_supported and console_output:
        print('### All layers supported!')
    return all_supported