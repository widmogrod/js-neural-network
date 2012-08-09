# Introduction

JavaSript implementation of feedforward neural network with backpropagarion algorithm.

## Note
    
This project is still in development phase. Further development should bring extended  description, add more use cases and involve code refactoring and optimalisation.

P.S. Sorry for my english. If You wish to help me with this project or correct my english description - You are welcome :)

# Requirements
 
   * node js - Basically this is not strong requirement. There is no problem to run this library in modern browser. I decide to use node.js because there is more CPU power to use by JavaScript

   
# Demo

## Learning the xor function

Run this command in the project main directory:

```
node demo/xor.js
```

Example output:

```
learn in epoch 1647 error 0.007963763295449296
test net. epoch 1647 error 0.007963763295449296
[ 0, 0 ] 0.06878616562023242
[ 0, 1 ] 0.9628854533576559
[ 1, 0 ] 0.9301744412288724
[ 1, 1 ] 0.06920811758028442
```

## Learning neural network to predict next number

_Objective_: learn neural network of simple number sequence and use this net to "predict" next number in test sequence

Run this command in the project main directory:

```
node demo/sequence-prediction.js
```

```
learn in epoch 3334 error 0.00009983697113540062
test net. epoch 3334 error 0.00009983697113540062
number sequence: 1, 2, 3, 4, 3, 2, 1, 2, 3, 4
input: [ 4, 3, 2 ] "predicted" output: round(1), raw(1.04923740569591) normalized input: [ 0.8, 0.6, 0.4 ] normalized output: 0.209847481139182
input: [ 3, 2, 1 ] "predicted" output: round(2), raw(1.7728240967319415) normalized input: [ 0.6, 0.4, 0.2 ] normalized output: 0.3545648193463883
```