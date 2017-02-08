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
    return new Array(size).fill(null);
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

function window(size, list) {
    return range(list.length/size | 0).map((_, offset) => {
        return range(size).map((_, index) => list[offset + index]);
    });
}

var sentences = require('fs').readFileSync(__dirname + '/sentences.txt').toString('utf8');
sentences = sentences.toLowerCase();
sentences = sentences.replace(/[^\w \n]+/gi, '');
sentences = sentences.split('\n');
sentences = sentences.map((sentence) => sentence.split(' '));
sentences = sentences.map((list) => list.filter(x => x));
sentences = sentences.filter(x => x.length);

var skipGram = sentences.map((list) => window(5, list));
skipGram = flatMap(id, skipGram);
skipGram = skipGram.map(([one, two, three]) => [[two], [one, three]]);

var wordsList = flatMap(id, sentences).filter((x) => x);
wordsList = unique(wordsList);

var wordsVectorMap = wordsList.map((word, index) => {
    return set(index, 1, fill(0, range(wordsList.length)));
}).reduce((m, vector, index) => {
    m[wordsList[index]] = vector;
    return m;
}, {});

var wordsInput = skipGram.map(([inputs]) => {
    return inputs.reduce((vector, word) => {
        return set(wordsList.indexOf(word), 1, vector);
    }, fill(0, range(wordsList.length)));
});
var wordsOutput = skipGram.map(([_, outs]) => {
    return outs.reduce((vector, word) => {
        return set(wordsList.indexOf(word), 1, vector);
    }, fill(0, range(wordsList.length)));
});

var inputNeurons = wordsList.length;
var outputNeurons = wordsList.length;

var net = new nn.neuralNet(inputNeurons, 7, outputNeurons);
net.setBias(1);

nn.util.setRandomWeights(net);

// Sigmoid
net.setActivationFunction(function(input) {
    return ( 1 / ( 1 + Math.exp(-input)))
});
net.setDerivativeOfActivationFunction(function(x) {
    return (x)*(1 - x);
});


// xor
var learnData_ = wordsInput.map((_, index) => {
    return wordsInput[index].concat(wordsOutput[index]);
});

var threshold_ = 0.008,
    epoch_ = 100000;

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


var word2vec = wordsList.map((word) => {
    return [word, net.update(wordsVectorMap[word])[0]];
});

console.log('\nLabels:\n')
word2vec.forEach(([label]) => {
    console.log(label);
});

console.log('\nword2vec as CSV:\n')
word2vec.forEach(([_, vec]) => {
    console.log(vec.join(','));
});

console.log('\nPut data into the form to visualize:');
console.log('- http://cs.stanford.edu/people/karpathy/tsnejs/csvdemo.html');


