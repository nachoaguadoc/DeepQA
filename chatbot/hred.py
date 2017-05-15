# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf
import numpy as np
from chatbot.textdata import Batch


class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args, textData):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        # Placeholders
        self.utteranceEncInputs  = None
        self.contextEncInputs  = None
        self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size
        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = []  # Outputs of the network, list of probability for each words
        self.numberUtterances = 0 # Should be in chatbot.py args 

        # Construct the graphs        
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        def get_state_variables(batch_size, cell):
            # For each layer, get the initial state and make a variable out of it
            # to enable updating its value.
            state_variables = []
            for state_c, state_h in cell.zero_state(batch_size, tf.float32):
                state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False),
                    tf.Variable(state_h, trainable=False)))
            # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
            return tuple(state_variables)

        def get_state_update_op(state_variables, new_states):
            # Add an operation to update the train states with the last state tensors
            update_ops = []
            for state_variable, new_state in zip(state_variables, new_states):
                # Assign the new state to the state variables on this layer
                update_ops.extend([state_variable[0].assign(new_state[0]),
                                   state_variable[1].assign(new_state[1])])
            # Return a tuple in order to combine all update_ops into a single operation.
            # The tuple's actual value should not be used.
            return tf.tuple(update_ops)

        def get_state_reset_op(state_variables, cell, batch_size):
            # Return an operation to set each variable in a list of LSTMStateTuples to zero
            zero_states = cell.zero_state(batch_size, tf.float32)
            return get_state_update_op(state_variables, zero_states)


        init_op = tf.global_variables_initializer()

        with tf.name_scope('placeholder_batch'):
            self.batchSize = tf.placeholder(tf.int32)
        if self.args.test:
            self.numberUtterances = 1
        else: 
            self.numberUtterances = 2

        def create_rnn_cell(scope):
            with tf.variable_scope(scope):
                encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                    self.args.hiddenSize
                )

                if not self.args.test:  # TODO: Should use a placeholder instead
                    encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                        encoDecoCell,
                        input_keep_prob=1.0,
                        output_keep_prob=self.args.dropout
                    )
                if scope=='decoder':
                    encoDecoCell = tf.contrib.rnn.OutputProjectionWrapper(
                        encoDecoCell,
                        output_size=self.textData.getVocabularySize()
                    )
                if scope=='utterance_encoder':
                    encoDecoCell = tf.contrib.rnn.EmbeddingWrapper(
                        encoDecoCell,
                        embedding_classes=self.textData.getVocabularySize(),
                        embedding_size=self.args.embeddingSize,
                    )
                return encoDecoCell

        utterance_encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell('utterance_encoder') for _ in range(self.args.numLayers)],
        )
        context_encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell('context_encoder') for _ in range(self.args.numLayers)],
        )        
        decoder_cell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell('decoder') for _ in range(self.args.numLayers)],
        )
        lastContextState = get_state_variables(self.batch_size, context_encoder_cell)
        # Creation of the Utterance encoder
        # cell: MultiRnnCell([ BasicLSTMCell(hiddenSize) ])
        # inputs: [ batchSize * sentenceLength * wordDimensions ]
        # lengths: [ batchSize ] length of each sentence in the batch
        def utterance_encoder(cell, inputs, sequence_length, reset, batch_size=1, dtype=tf.float32, scope='utterance_encoder'):

            initial_state = cell.zero_state(batch_size, tf.float32)
            outputs, state = tf.nn.dynamic_rnn(
                cell=cell,
                time_major=True,
                dtype=dtype,
                sequence_length=sequence_length,
                inputs=inputs,
                initial_state=initial_state,
                scope=scope,
            )
            return outputs, state

        # Creation of the Utterance encoder
        # cell: MultiRnnCell([ BasicLSTMCell(hiddenSize) ])
        # inputs: [ batchSize * sentenceLength * wordDimensions ]
        # lengths: [ batchSize ] length of each sentence in the batch (without taking padding into account)
        def context_encoder(cell, inputs, reset, batch_size=1, dtype=tf.float32, scope='context_encoder'):
            initial_state = cell.zero_state(self.batchSize, tf.float32) if reset else self.lastContextState
            outputs, state = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=dtype,
                time_major=True,
                inputs=inputs,
                initial_state = initial_state,
                scope=scope)
            return outputs, state
            
        # Creation of the Utterance encoder
        # cell: MultiRnnCell([ BasicLSTMCell(hiddenSize) ])
        # inputs: [ batchSize * sentenceLength * wordDimensions ]
        # lengths: [ batchSize ] length of each sentence in the batch (without taking padding into account)
        def decoder(cell, inputs, initial_state, batch_size=1, dtype=tf.int32, feed_previous=True, scope=None):
            outputs, state = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(
                inputs,
                initial_state,
                cell,
                self.textData.getVocabularySize(),
                output_projection=None,
                embedding_size=self.args.embeddingSize,
                feed_previous=feed_previous,
                scope=scope)
            return outputs, state

        # Network input (placeholders)
        
        with tf.name_scope('placeholder_utterance_encoder'):
            self.utteranceEncInputs  = tf.placeholder(tf.int32, [self.args.maxLengthEnco, None])
            self.utteranceEncLengths = tf.placeholder(tf.int32, [None]) # Batch size

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]
            

        self.lossFct = 0

        with tf.variable_scope('utterances', reuse=reuse):
            utteranceEncInput = tf.reshape(self.utteranceEncInputs[:,:], [self.args.maxLengthEnco, self.batchSize, 1])
            utteranceEncLength = tf.reshape(self.utteranceEncLengths[:], [self.batchSize])

            utteranceEncOutputs, utteranceEncState = utterance_encoder(
                cell=utterance_encoder_cell,
                inputs=utteranceEncInput,
                sequence_length=utteranceEncLength,
                reset=True,
                batch_size=self.batchSize
            )

        with tf.variable_scope('context', reuse=reuse):
            self.contextEncInputs = tf.reshape(utteranceEncOutputs[-1], [1, self.batchSize, self.args.hiddenSize])

            contextEncOutputs, contextEncState = context_encoder(
                cell=context_encoder_cell,
                inputs=self.contextEncInputs,
                reset=reset
            )
            reset_op = get_state_reset_op(lastContextState, context_encoder_cell, self.batchSize)
            update_op = get_state_update_op(lastContextState, contextEncState)

        with tf.variable_scope('decoders', reuse=reuse):
            decoderOutputs, decoderState = decoder(
                cell=decoder_cell,
                inputs=self.decoderInputs,
                initial_state=lastContextState,
                feed_previous=bool(self.args.test),
                dtype=tf.int32
            )
        if not self.args.test:
            self.lossFct += tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function= None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost
    
            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.resetOps = reset_op
            self.updateOps = update_op
            self.optOp = opt.minimize(self.lossFct)
        else: 
            self.outputs = decoderOutputs

    def step(self, batch):

        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """
        # Feed the dictionary
        feedDict = {}
        ops = None
        if not self.args.test:  # Training
            feedDict[self.batchSize] = batch.batchSize
            feedDict[self.utteranceEncLengths] = batch.encoderLengths
            feedDict[self.utteranceEncInputs] = batch.encoderSeqs
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            resetOps = (self.optOp, self.resetOps, self.lossFct)
            updateOps = (self.optOp, self.updateOps, self.lossFct)
        else:  # Testing (batchSize == 1)
            feedDict[self.batchSize] = batch.batchSize
            feedDict[self.utteranceEncLengths] = np.reshape(batch.encoderLengths, [batch.batchSize,1])
            feedDict[self.utteranceEncInputs] = np.reshape(batch.encoderSeqs, [self.args.maxLengthEnco, batch.batchSize, 1])
            feedDict[self.decoderInputs[0][0]]  = [self.textData.goToken]
            ops = (self.outputs,)
        # Return one pass operator

        return resetOps, updateOps, feedDict
