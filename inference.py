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


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.plugin = None 
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None


    def load_model(self, model, device="MYRIAD", cpu_ext=None):
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0]+".bin"
    
        # Init plugin
        self.plugin = IECore()

        ### TODO: Check for supported layers ###
        # supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        # unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        # if len(unsupported_layers) != 0:
        #    print('Unsupported layers: {}'.format(unsupported_layers))
        #    exit(1)

        ### Add any necessary extensions ###
        if "CPU" in device and cpu_ext:
            self.plugin.add_extension(cpu_ext, device)

        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.exec_network = self.plugin.load_network(self.network, device)

        ### Return the loaded inference plugin ###
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        shape = self.network.inputs[self.input_blob].shape
        return shape

    def exec_net(self, img):
        ### Start an asynchronous request ###
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: img})
        ### Return any necessary information ###
        return

    def wait(self):
        ### Wait for the request to be complete. ###
        status = self.exec_network.requests[0].wait(-1)
        ### Return any necessary information ###    
        return status

    def get_output(self):
        ### Extract and return the output results
        out = self.exec_network.requests[0].outputs[self.output_blob]
        return out
