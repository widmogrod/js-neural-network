var normalize = require('../lib/normalize.js').normalize,
    nn = require('../lib/nn-node.js').nn;

var net = new nn.neuralNet(2, 6, 1);
net.setBias(1);

nn.util.setRandomWeights(net);

// Sigmoid
net.setActivationFunction(function(input) {
    return ( 1 / ( 1 + Math.exp(-input)))
});
net.setDerivativeOfActivationFunction(function(x) {
    // f(x)*(1-f(x))
    return (x)*(1 - x);
});

// xor
var learnData_ = [
    //[input1, input2, output]
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
];

// xor
var testData_ = [
    //[input1, input2]
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

var threshold_ = 0.008,
    epoch_ = 10000;

var error;
for (var e = 0; e < epoch_; e++)
{
    var bp = nn.backPropagation(net);
    bp.setLearningRate(0.7);
    bp.setMomentum(0.6);
    bp.setLearnData(learnData_);
    error = bp.learn();
    if (error < threshold_)
    {
        console.log('learn in epoch', e, 'error', error)
        break;
    }
}

console.log('test net.', 'epoch', e, 'error', error);

for (var x = 0; x < testData_.length; x++)
{
    console.log(
        testData_[x],
        net.update(testData_[x]).pop().pop()
    );
}