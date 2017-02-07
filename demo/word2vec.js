var normalize = require('../lib/normalize.js').normalize,
    nn = require('../lib/nn-node.js').nn;

function id(x) {
    return x;
}

function unique(xxs) {
    return Object.keys(
        xxs.reduce(function(agg, x) {
            agg[x] = true;
            return agg;
        }, {}));
}

function flatMap(fn, data) {
    return data.reduce(function(aggregate, item) {
        fn(item).map((x) => aggregate.push(x));
        return aggregate;
    }, []);
}

function range(size) {
    return new Array(size);
}

function fill(value, list) {
    return list.fill(value);
}

function set(key, value, list) {
    list[key] = value;
    return list;
}

function zip(listA, listB) {
    return listA.map((a, i) => [a, listB[i]]);
}

function indexTuple(listOfTuple) {
    return listOfTuple.reduce((m, [head, tail]) => {
        m[head] = tail;
        return m;
    }, {});
}

var rawData = 'king|kindom,queen|kindom,king|palace,queen|palace,king|royal,queen|royal,king|George,queen|Mary,man|rice,woman|rice,man|farmer,woman|farmer,man|house,woman|house,man|George,woman|Mary';
rawData = rawData.split(',');
rawData = rawData.map((x) => x.split('|'));

var wordsList = unique(flatMap(id, rawData));
var wordsInput = wordsList.map((word, index) => {
    return set(index, 1, fill(0, range(wordsList.length)));
});
var wordsInputTuple = zip(wordsList, wordsInput);
var wordsInputIndexed = indexTuple(wordsInputTuple);

var inputNeurons = wordsInput.length;
var outputNeurons = wordsInput.length;

var net = new nn.neuralNet(inputNeurons, 5, outputNeurons);
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
var learnData_ = rawData.map(([inWord, outWord]) => {
    return wordsInputIndexed[inWord].concat(wordsInputIndexed[outWord]);
});

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


var word2vec = wordsInputTuple.map(([word, inputWordVector]) => {
    return [word, net.update(inputWordVector)[0]];
});

console.log('\nLabels:\n')
wordsList.forEach((w) => {
    console.log(w);
});

console.log('\nword2vec as CSV:\n')
word2vec.forEach(([_, vec]) => {
    console.log(vec.join(','));
});

console.log('\nPut data into the form to visualize:');
console.log('- http://cs.stanford.edu/people/karpathy/tsnejs/csvdemo.html');


