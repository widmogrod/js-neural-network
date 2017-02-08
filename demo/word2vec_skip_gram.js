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

var sentences =`
I see a queen
It's the queen
I see the queen
He's a drama queen
She's a drama queen
He bowed to the Queen
I remember seeing the queen
I remember meeting the queen
Queen Elizabeth died in 1603
The queen visited the museum
The king and queen are coming
I remember that I met the queen
The queen stood beside the king
Mary was elected queen of the prom
The Queen's crown was made of gold
I name this ship the Queen Elizabeth
The Queen lives in Buckingham Palace
Queen Elizabeth I passed away in 1603
It was given to me by the Queen herself
The rose is called the queen of flowers
This magazine has a picture of the Queen
Queen Liliuokalani was forced to surrender
The queen was gracious enough to invite us
Three ships were given to him by the queen
He's the prom king and she's the prom queen
In each beehive there can only be one queen
In 1891, Liliuokalani became queen of Hawaii
They hung out the flag for the Queen's visit
This is the palace the king and queen live in
They named the ship Queen Mary after the Queen
I had a jack, a king and three queens in my hand
The queen was wearing a magnificent silver dress
The queen is going to address parliament next week
Three ships were given to Columbus by Queen Isabella
There was a time when kings and queens reigned over the world
The president was greeted by the queen on arrival at the palace
The Queen made an address to the nation on television yesterday
He was a good king
The king was executed
Tom lives like a king
We'll live like kings
The king abused his power
She treated him like a king
He acts as if he were a king
He acted the part of King Lear
When I grow up, I want to be a king
The people rebelled against the king
The king will appear in person tomorrow evening
Down with the king
The king is coming
He lives like a king
He was voted prom king
The king got undressed
He was more than a king
We want to see the king
He was every inch a king
To me, he is like a king
The eagle is king of birds
The King of France is bald
The lion is king of beasts
Who died and made you king
He was named after the king
The king crushed his enemies
The king's son was kidnapped
He served his king faithfully
King size beds are really big
The king always wears a crown
The king and queen are coming
The king governed the country
The king oppressed his people
The king took his clothes off
The lion is the king of beasts
The queen stood beside the king
He was the king of rock-and-roll
The king reigned over the island
They defied the laws of the king
He made up a story about the king
The king ruled his kingdom justly
The prince became a king that day
When I grow up, I want to be king
The king once lived in that palace
The king was deprived of his power
The king was stripped of his power
The king went hunting this morning
The lion is the king of the beasts
The lion is the king of the jungle
They are plotting to kill the king
`;

sentences = sentences.replace(/[^\w \n]+/gi, '');
sentences = sentences.split('\n');
sentences = sentences.map((sentence) => sentence.split(' '));
sentences = sentences.map((list) => list.filter(x => x));
sentences = sentences.filter(x => x.length);

var skipGram = sentences.map((list) => window(3, list));
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

var net = new nn.neuralNet(inputNeurons, 5, outputNeurons);
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


