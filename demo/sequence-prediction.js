var normalize = require('../lib/normalize.js').normalize,
    nn = require('../lib/nn-node.js').nn;

normalize.min = 0;
normalize.max = 5;

var net = new nn.neuralNet(3, 6, 1);
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

// Sequense: 1, 2, 3, 4, 3, 2, 1, 2, 3, 4
var learnData_ = [
    //[input1, input2, input3, output]
    [normalize.number(1), normalize.number(2), normalize.number(3), normalize.number(4)],
    [normalize.number(2), normalize.number(3), normalize.number(4), normalize.number(3)],
    [normalize.number(3), normalize.number(4), normalize.number(3), normalize.number(2)],
    [normalize.number(4), normalize.number(3), normalize.number(2), normalize.number(1)]
];

// test data
var testData_ = [
    [4, 3, 2],
    [3, 2, 1]
];

var normalizedTestData_ = [
    //[input1, input2]
    [normalize.number(4), normalize.number(3), normalize.number(2)],
    [normalize.number(3), normalize.number(2), normalize.number(1)]
];

var threshold_ = 0.0001,
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
console.log('number sequence: 1, 2, 3, 4, 3, 2, 1, 2, 3, 4');

for (var x = 0; x < normalizedTestData_.length; x++)
{
    var out_ = net.update(normalizedTestData_[x]).pop().pop();
    console.log(
        'input:',               testData_[x],
        '"predicted" output: round(' + Math.round(normalize.revertNumber(out_)) + '), raw(' + normalize.revertNumber(out_) + ')',

        'normalized input:',    normalizedTestData_[x],
        'normalized output:',   out_
    );
}