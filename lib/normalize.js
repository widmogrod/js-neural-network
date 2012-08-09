(function(exports){

    normalize = {};
    normalize.max = 100;
    normalize.min = 100;
    normalize.number = function(n) {
//        return n/1000;
        return (n - 0)/(normalize.max - normalize.min)*(1-0) + 0;
    };
    normalize.numbers = function(numbers) {
        normalize.min = Math.min.apply(null, numbers);
        normalize.max = Math.max.apply(null, numbers);
        return numbers.map(normalize.number);
    };
    normalize.revertNumber = function(n)
    {
//        return n* 1000;
        return (n - 0)/(1 - 0)*(normalize.max - normalize.min) + normalize.min;
    };

    exports.normalize = normalize;

})(exports);