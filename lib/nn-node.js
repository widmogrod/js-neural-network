/**
 * Simple implementation of Feed forward neural network.
 *
 * @date 2012.06.23
 * @author Gabriel Habryn <widmogrod@gmail.com>
 */
 (function(app){

     function neuron(inputsNumber)
     {
         var weights_ = [],
             count_ = inputsNumber,
             bias_ = -1;

         for (var i = 0; i < inputsNumber; i++) {
             weights_[i] = null;
         }
         // weight for bias
         weights_[i] = null;

         function update(inputsArray, activationFunction)
         {
             var input = 0;

             for (var i = 0; i < count_; i++) {
                 input += weights_[i] * inputsArray[i];
             }

             input += weights_[i] * bias_;

             return activationFunction(input);
         }

         return {
             'count': function() {return count_;},
             'update': update,
             'getWeights': function() {return weights_},
             'setWeights': function(weights) {weights_ = weights;},
             'setWeightForInput': function(weight, inputNumber) {weights_[inputNumber] = weight; },
             'setBias': function(bias) {bias_ = bias;},
             'getBias': function() {return bias_;}
         };
     };

     function neuronLayer(neuronsNumber, inputsNumberPerNeuron)
     {
         var neurons_ = [],
             count_ = neuronsNumber,
             bias_ = -1;

         for (var i = 0; i < neuronsNumber; i++) {
             neurons_[i] = new app.nn.neuron(inputsNumberPerNeuron);
         }

         function update(inputsArray, activationFunction)
         {
             var outputArray = [];

             for (var neuronNumber = 0; neuronNumber < neurons_.length; neuronNumber++) {
                 outputArray[neuronNumber] = neurons_[neuronNumber].update(inputsArray, activationFunction);
             }

             return outputArray;
         }

         function getWeights()
         {
             var weightsArray = [];

             for (var neuronNumber = 0; neuronNumber < count_; neuronNumber++) {
                 weightsArray[neuronNumber] = neurons_[neuronNumber].getWeights();
             }

             return weightsArray;
         }

         function setWeights(weightsArray)
         {
             for (var neuronNumber = 0; neuronNumber < count_; neuronNumber++) {
                 neurons_[neuronNumber].setWeights(
                     weightsArray[neuronNumber]
                 );
             }
         }

         function setBias(bias)
         {
             bias_ = bias;
             for (var neuronNumber = 0; neuronNumber < count_; neuronNumber++) {
                neurons_[neuronNumber].setBias(bias_);
             }
         }

         return {
             'count': function() {return count_;},
             'update': update,
             'getNeurons' : function() {return neurons_;},
             'setBias': setBias,
             'getBias': function() {return bias_;},
             'getWeights': getWeights,
             'setWeights': setWeights
         };
     };

     function neuralNet()
     {
         var args = Array.prototype.slice.call(arguments),
             layers_ = [],
             // first is inputs number
             inputsNumber_ = args.shift(),
             // number of hidden and output layer
             layersNumber_ = args.length,
             // last is outputs number
             outputsNumber_ = args.pop(),

             hiddenLayers_ = args,
             weightsNumber_ = null,
             activationFunction_ = function() {},
             derivativeOfActivationFunction_ = function() {},
             bias_ = -1;

         var inputsNumberPerNeuron = inputsNumber_;

         // hidden layers
         for (var layerNumber = 0; layerNumber < hiddenLayers_.length; layerNumber++)
         {
             // current number of neurons
             var numberOfNeurons = hiddenLayers_[layerNumber];

             layers_[layerNumber] = new app.nn.neuronLayer(
                 numberOfNeurons,
                 inputsNumberPerNeuron
             );

             inputsNumberPerNeuron = numberOfNeurons
         }

         // output layer
         layers_[layerNumber] = new app.nn.neuronLayer(
             outputsNumber_,
             layers_[hiddenLayers_.length-1].count()
         );

         function update(inputsArray)
         {
             var result = [],
                outputArray;

             for (var layerNumber_ = 0; layerNumber_ < layers_.length; layerNumber_++)
             {
                 outputArray = layers_[layerNumber_].update(inputsArray, activationFunction_);
                 // output is input for next layer
                 inputsArray = outputArray;

                 result[layerNumber_] = outputArray;
             }

             return result;
         }

         function getWeights()
         {
             var weightsArray = [];
             for (var layerNumber_ = 0; layerNumber_ < layers_.length; layerNumber_++) {
                 weightsArray[layerNumber_] = layers_[layerNumber_].getWeights();
             }

             return weightsArray;
         }

         function setWeights(weightsArray)
         {
             for (var i = 0; i < layers_.length; i++)
             {
                 layers_[i].setWeights(
                     weightsArray[i]
                 );
             }
         }

         function weightsCount()
         {
             if (null !== weightsNumber_) {
                 return weightsNumber_;
             }

             for (var layerNumber_ = 0; layerNumber_ < layers_.length; layerNumber_++)
             {
                 var neurons_ = layers_[layerNumber_].getNeurons();
                 for (var neuronNumber_ = 0; neuronNumber_ < neurons_.length; neuronNumber_++)
                 {
                     weightsNumber_ += neurons_[neuronNumber_].count();
                     // bias
                     weightsNumber_ += 1;
                 }
             }

             return weightsNumber_;
         }

         function setBias(bias)
         {
             bias_ = bias;
             for (var layerNumber_ = 0; layerNumber_ < layers_.length; layerNumber_++) {
                 layers_[layerNumber_].setBias(bias_);
             }
         }

         return {
             'count': function() {return layersNumber_;},
             'update': update,
             'getLayers': function() {return layers_;},
             'getWeights': getWeights,
             'setWeights': setWeights,
             'weightsCount': weightsCount,
             'setActivationFunction': function(func) { activationFunction_ = func; },
             'setDerivativeOfActivationFunction': function(func) { derivativeOfActivationFunction_ = func; },
             'getDerivativeOfActivationFunction': function () { return derivativeOfActivationFunction_; },
             'derivativeOfActivationFunction': function(x) {return derivativeOfActivationFunction_(x)},
             'setBias': setBias,
             'getBias': function() {return bias_;},
             'getInputsNumber' : function() { return inputsNumber_; },
             'getOutputsNumber' : function() { return outputsNumber_; }
         };
     };

     function backPropagation(neuralNet)
     {
         var learnData_ = [],
             neuralNet_ = neuralNet,
             learningRate_ = 0.9,
             momentum_ = 0.9;

         var previousDeltaWeight = [];

         function learn()
         {
             var inputsNumber_ = neuralNet_.getInputsNumber(),
                 outputsNumber_ = neuralNet_.getOutputsNumber();

             var err = 0;

             for (var i = 0; i < learnData_.length; i++)
             {
                 var learnInput_ = learnData_[i].slice(0, inputsNumber_),
                     learnOutput_ = learnData_[i].slice(inputsNumber_);

                 var output_ = neuralNet_.update(learnInput_);
                 var deltas_ = deltaForOutputLayer(output_, learnOutput_);
                 var newWeights_ = modifyWeights(deltas_, learnInput_, output_);
                 neuralNet_.setWeights(newWeights_);

                 // @todo how to calculate the error for more than one output?
                 var delta = (learnOutput_[0] - output_.pop().pop());
                 err += (delta*delta)/2;
             }

             return err;
         }

         function modifyWeights(deltas, learnInput, outputs_)
         {
             var layers_ = neuralNet_.getLayers(),
                 weights_ = neuralNet_.getWeights(),
                 df_ = neuralNet_.getDerivativeOfActivationFunction();

             var __nw__ = [];
             for (var layerNumber_ = 0; layerNumber_ < layers_.length; layerNumber_++)
             {
                 if (!previousDeltaWeight[layerNumber_]) {
                     previousDeltaWeight[layerNumber_] = [];
                 }


                 __nw__[layerNumber_] = [];
                 var neurons_ = layers_[layerNumber_].getNeurons();
                 for (var neuronNumber_ = 0; neuronNumber_ < neurons_.length; neuronNumber_++)
                 {
                     if (!previousDeltaWeight[layerNumber_][neuronNumber_]) {
                         previousDeltaWeight[layerNumber_][neuronNumber_] = [];
                     }

                     __nw__[layerNumber_][neuronNumber_] = [];
                     var count_ = weights_[layerNumber_][neuronNumber_].length -1;
                     for (var i = 0; i < count_; i++)
                     {
                         var inputValue_ = (layerNumber_ < 1) ? learnInput[i] : outputs_[layerNumber_][neuronNumber_];

                         __nw__[layerNumber_][neuronNumber_][i] = weights_[layerNumber_][neuronNumber_][i];
                         __nw__[layerNumber_][neuronNumber_][i] += (
                                learningRate_
                                 * deltas[layerNumber_][neuronNumber_]
                                 * inputValue_
                         );

                         // momentum
                         __nw__[layerNumber_][neuronNumber_][i] += (
                             momentum_ * (previousDeltaWeight[layerNumber_][neuronNumber_][i] ? previousDeltaWeight[layerNumber_][neuronNumber_][i] : 0)
                         );

                         previousDeltaWeight[layerNumber_][neuronNumber_][i] = __nw__[layerNumber_][neuronNumber_][i] - weights_[layerNumber_][neuronNumber_][i];
                     }

                     __nw__[layerNumber_][neuronNumber_][i] = weights_[layerNumber_][neuronNumber_][i];
                     __nw__[layerNumber_][neuronNumber_][i] += (
                            learningRate_
                             * deltas[layerNumber_][neuronNumber_]
                             * neuralNet_.getBias()
                     );


                     // momentum
                     __nw__[layerNumber_][neuronNumber_][i] += (
                         momentum_ * (previousDeltaWeight[layerNumber_][neuronNumber_][i] ? previousDeltaWeight[layerNumber_][neuronNumber_][i] : 0)
                     );

                     previousDeltaWeight[layerNumber_][neuronNumber_][i] = __nw__[layerNumber_][neuronNumber_][i] - weights_[layerNumber_][neuronNumber_][i];
                 }
             }

             return __nw__;
         }

         // update delta rule
         function deltaForOutputLayer(netOutputs, desiredOutput)
         {
             var layers_ = neuralNet_.getLayers(),
                 weights_ = neuralNet_.getWeights(),
                 outputLayerNumber_ = layers_.length- 1,
                 deltas_ = {};

             // go from output layer to input layer
             for (var layerNumber_ = layers_.length-1; layerNumber_ >= 0; layerNumber_--)
             {
                 deltas_[layerNumber_] = {};

                 var neurons_ = layers_[layerNumber_].getNeurons();
                 for (var neuronNumber_ = 0; neuronNumber_ < neurons_.length; neuronNumber_++)
                 {
                     if (outputLayerNumber_ == layerNumber_)
                     {
                         deltas_[layerNumber_][neuronNumber_]
                             = neuralNet_.derivativeOfActivationFunction(netOutputs[layerNumber_][neuronNumber_])
                                * (desiredOutput[neuronNumber_] - netOutputs[layerNumber_][neuronNumber_]);

                     }
                     else
                     {
                         // number of neurons next in layer
                         var count_ = layers_[layerNumber_+1].count();

                         deltas_[layerNumber_][neuronNumber_] = 0;
                         for (var j = 0; j < count_; j++) {
                             deltas_[layerNumber_][neuronNumber_] += weights_[layerNumber_+1][j][neuronNumber_] * deltas_[layerNumber_+1][j];
                         }

                         deltas_[layerNumber_][neuronNumber_]
                            = neuralNet_.derivativeOfActivationFunction(netOutputs[layerNumber_][neuronNumber_])
                                * deltas_[layerNumber_][neuronNumber_];
                     }
                 }
             }

             return deltas_;
         }

         return {
             'setLearnData': function(data) { learnData_ = data; },
             'getLearnData': function() { return learnData_; },
             'setMomentum': function(momentum) { momentum_ = momentum; },
             'getMomentum': function() { return momentum_; },
             'setLearningRate': function(learningRate) { learningRate_ = learningRate; },
             'getLearningRate': function() { return learningRate_; },
             'learn': learn
         };
     }

     app.nn = app.nn || {};
     app.nn.util = app.nn.util || {
         'getRandomWeight': function () {
             if (!this.random) {
                 this.random = Math.random() - Math.random();
             } else {
                 var old = this.random;;
                 while (this.random == old) {
                     this.random = Math.random() - Math.random();
                 }
             }
             return this.random;
         },
         'setRandomWeights': function(neuralNet)
         {
             var layers_ = neuralNet.getLayers();

             for (var layerNumber_ = 0; layerNumber_ < layers_.length; layerNumber_++)
             {
                 var neurons_ = layers_[layerNumber_].getNeurons();
                 for (var neuronNumber_ = 0; neuronNumber_ < neurons_.length; neuronNumber_++)
                 {
                     var inputsNumber_ = neurons_[neuronNumber_].count();
                     for (var inputNumber_ = 0; inputNumber_ < inputsNumber_; inputNumber_++)
                     {
                         neurons_[neuronNumber_].setWeightForInput(
                             app.nn.util.getRandomWeight(),
                             inputNumber_
                         );
                     }

                     neurons_[neuronNumber_].setWeightForInput(
                         app.nn.util.getRandomWeight(),
                         inputNumber_
                     );
                 }
             }
         }
     };

     app.nn.neuron = neuron;
     app.nn.neuronLayer = neuronLayer;
     app.nn.neuralNet = neuralNet;
     app.nn.backPropagation = backPropagation;

 })(exports);