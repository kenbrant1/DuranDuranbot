# NeuralNet.sc
Basic ANN implementation in SuperCollider.

Usage: 
```
n = NeuralNet.new(inputSize : Integer, outputSize : Integer, [hiddenLayers : Int], bias : Boolean, learningRate : Float, momentum : Float, errorMargin : Float);
n.train([ [trainingInputs : Float] ], [ [trainingOutputs : Float] ]);
~output = n.evaluateInputs([input : Float]);
```

Also supported: \*.fromFile and .toFile.
