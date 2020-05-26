// NeuralNet.sc written and adapted for SuperCollider 3.6 by Brian Heim

NeuralNet {
	const token = ",";

	var bias;
	var learningRate;
	var momentum;
	var errorMargin;

	var >gradients, >prevGradients;
	var <>nodes, <>weights, >nodeDeltas;
	var >layerIndex, >layerSize;
	var nIns, nOuts;

	*new {
		arg inputSize, outputSize, hiddenSizes, bias = true, learningRate = 0.8, momentum = 0.7, errorMargin = 1e-4;
		^super.new.init(inputSize, outputSize, hiddenSizes, bias, learningRate, momentum, errorMargin);
	}

	init {
		arg inputSize, outputSize, hiddenSizes, inbias, inlearningRate, inmomentum, inerrorMargin;
		var index = 0, nWeights = 0;

		nIns = inputSize;
		nOuts = outputSize;
		bias = inbias;
		learningRate = inlearningRate;
		momentum = inmomentum;
		errorMargin = inerrorMargin;

		layerIndex = Array.newClear(hiddenSizes.size + 2);
		layerSize = Array.newClear(layerIndex.size);

		layerIndex[0] = index;
		layerSize[0] = inputSize + bias.binaryValue;
		index = index + layerSize[0];
		layerIndex[1] = index;

		hiddenSizes.size.do {
			|i|
			layerSize[i + 1] = hiddenSizes[i] + bias.binaryValue;
			index = index + layerSize[i + 1];
			layerIndex[i + 2] = index;
		};

		layerSize[layerSize.size - 1] = outputSize;
		index = index + outputSize;
		// index now holds the total number of nodes in the network,
		// but we don't need that for layerIndex

		nodes = Array.fill(index, 0);
		nodeDeltas = Array.fill(index, 0);

		// calculate the number of weights
		nWeights = 0;
		(layerSize.size - 2).do {
			|i|
			nWeights = nWeights + (layerSize[i] * (layerSize[i + 1] - bias.binaryValue));
		};
		nWeights = nWeights + (layerSize[layerSize.size - 2] * layerSize[layerSize.size - 1]);

		weights = Array.fill(nWeights, 0);
		gradients = Array.fill(nWeights, 0);
		prevGradients = Array.fill(nWeights, 0);

		this.randomizeWeights;
	}

	randomizeWeights {
		weights.do {|item, i| weights[i]= -1.0.rrand(1.0)};
	}

	train {
		// trainingInputs is a double array of input sets, training out is a double array of output sets
		arg trainingIn, trainingOut;
		var error, iteration = 0;

		"calculate error".postln;
		error = this.calculateError(trainingIn, trainingOut);
		while {error > errorMargin} {
			("Iteration " ++ iteration ++ ". Error: " ++ error).postln;
			//"train once".postln;
			this.trainOnce(trainingIn, trainingOut);
			//"calculate error".postln;
			error = this.calculateError(trainingIn, trainingOut);
			iteration = iteration+1;
		};
		("Training set completed successfully with an error of: " ++ error).postln;
	}

	gradientCalculation {
		arg expectedOut;
		var iNode1, iNode2, iWeight, bBiasAdj, deltaSum, biasAdjustment;
		// calculate error for each output
		nOuts.do {
			|i|
			iNode1 = layerIndex[layerIndex.size - 1] + i;
			nodeDeltas[iNode1] = NeuralNet.activationDerivative(nodes[iNode1]) * (expectedOut[i] - nodes[iNode1]);
		};

		iWeight = weights.size - 1;

		forBy(layerIndex.size-2, 0, -1) {
			|iLayer|
			bBiasAdj = bias&&(iLayer!=(layerIndex.size - 2));
			forBy(layerSize[iLayer]-1, 0, -1) {
				|iLayerNode|
				iNode1 = layerIndex[iLayer] + iLayerNode;
				deltaSum = 0;
				forBy(layerSize[iLayer+1]-bBiasAdj.binaryValue-1, 0, -1) {
					|iNextLayerNode|
					iNode2 = layerIndex[iLayer + 1] + iNextLayerNode;
					deltaSum = deltaSum + (nodeDeltas[iNode2] * weights[iWeight]);
					iWeight = iWeight - 1;
				};
				nodeDeltas[iNode1] = deltaSum *  ((iLayerNode!=(layerSize[iLayer]-1))&&(iLayer!=0)).if({NeuralNet.activationDerivative(nodes[iNode1])}, {nodes[iNode1]})
			};
		};
		// now we have the node deltas for all the nodes except input (unnecessary)

		// calculate gradients
		iWeight = 0;
		(layerIndex.size-1).do {
			|iLayer|
			biasAdjustment = 0-(bias && (iLayer!=(layerIndex.size-2))).binaryValue;
			layerSize[iLayer].do {
				|iLayerNode|
				(layerSize[iLayer+1]+biasAdjustment).do {
					|iNextLayerNode|
					iNode1 = layerIndex[iLayer] + iLayerNode;
					iNode2 = layerIndex[iLayer + 1] + iNextLayerNode;
					gradients[iWeight] = gradients[iWeight] + (nodeDeltas[iNode2] * nodes[iNode1]);
					iWeight = iWeight + 1;
				};
			};
		};
	}

	updateWeights {
		weights.size.do {
			|i|
			gradients[i] = (gradients[i] * learningRate) + (prevGradients[i] * momentum);
			weights[i] = weights[i] + gradients[i];
		};
		prevGradients.size.do {
			|i|
			prevGradients[i] = gradients[i];
		};
	}

	///////////////////////////////////////
	//////////// I/O METHODS //////////////
	///////////////////////////////////////

	toFile {
		arg filename;

		var file = File(filename, "w"), str = "";
		file.isOpen;
		[bias, learningRate, momentum, errorMargin, nIns, nOuts].do {
			|element|
			str = str ++ element.asInteger ++ token;
		};
		str[str.size-1] = $\n; // replace last comma with newline
		file.write(str);

		[gradients, prevGradients, nodes, weights, nodeDeltas, layerIndex, layerSize].do {
			|item, i| // identical process for each array in data
			item.size.postln;
			str = "";
			item.do {
				|element|
				str = str ++ element ++ token;
			};
			str[str.size-1] = $\n; // same as above
			file.write(str);
		};
		file.close;
	}

	*fromFile {
		arg filename;

		var reader, data, net;

		// these in the first line
		/*var bias;
		var learningRate;
		var momentum;
		var errorMargin;
		var nIns, nOuts */

		// these each on their own line
		/*var gradients, prevGradients;
		var nodes, weights, nodeDeltas;
		var layerIndex, layerSize; */

		reader = CSVFileReader(File(filename, "r"));
		data = reader.read(true, true);
		reader.close;

		net = NeuralNet.new(data[0][4].asInteger,
			data[0][5].asInteger,
			[1],
			data[0][0].asInteger.asBoolean,
			data[0][1].asInteger,
			data[0][2].asInteger,
			data[0][3].asInteger);
		net.gradients = data[1].collect(_.asFloat);
		net.prevGradients = data[2].collect(_.asFloat);
		net.nodes = data[3].collect(_.asFloat);
		net.weights = data[4].collect(_.asFloat);
		net.nodeDeltas = data[5].collect(_.asFloat);
		net.layerIndex = data[6].collect(_.asInteger);
		net.layerSize = data[7].collect(_.asInteger);

		^net;
	}

	//////////////////////////////////////////
	//////////// HELPER METHODS //////////////
	//////////////////////////////////////////

	trainOnce {
		arg trainingIn, trainingOut;
		gradients = Array.fill(weights.size, 0.0);
		trainingIn.size.do {
			|i|
			//("training set: " ++ i).postln;
			this.process(trainingIn[i]);
			this.gradientCalculation(trainingOut[i]);
		};
		this.updateWeights;
	}

	*activationFunction {
		arg vals, start, size;
		size.do {
			|i|
			vals[start + i] = 1.0 / (1.0 + (-1.0 * vals[start + i]).exp);
		};
	}

	*activationDerivative {
		arg val;
		^val * (1.0 - val) + 0.1;
	}

	calculateError {
		arg trainingIn, trainingOut;
		var globalError = 0.0, actualOut, runningErrorTotal;
		// run all tests, find error
		// mean squared error
		trainingIn.size.do {
			|i|
			//("training set " ++ i).postln;
			this.process(trainingIn[i]);
			actualOut = this.getOutput;
			runningErrorTotal = 0.0;
			trainingOut[i].size.do {
				|j|
				runningErrorTotal = runningErrorTotal + (trainingOut[i][j] - actualOut[j]).squared;
			};
			globalError = globalError + runningErrorTotal;
		};
		^(globalError / trainingIn.size);
	}

	resetNeurons {
		var lastLayer, lastNode, iNode;
		layerIndex.size.do {
			|iLayer|
			lastLayer = iLayer == (layerIndex.size - 1);
			layerSize[iLayer].do {
				|iLayerNode|
				lastNode = iLayerNode == (layerSize[iLayer] - 1);
				iNode = layerIndex[iLayer] + iLayerNode;
				nodes[iNode] = ((bias&&(lastLayer.not))&&lastNode).asInteger.asFloat;
			};
		};
	}

	evaluateInputs {
		arg in;
		this.process(in);
		^this.getOutput;
	}

	getOutput {
		^Array.fill(nOuts, { |i| nodes[layerIndex[layerIndex.size - 1] + i]; });
	}

	process {
		arg inputs;
		var iWeight = 0, biasAdjustment, node1, node2;

		this.resetNeurons;
		nIns.do { |i| nodes[i] = inputs[i] };

		(layerIndex.size - 1).do {
			|iLayer|
			// iLayer is the index of which layer we are computing
			biasAdjustment = 0-(bias&&(iLayer!=(layerIndex.size - 2))).binaryValue;

			layerSize[iLayer].do {
				|iLayerNode|
				// iLayerNode is the index of the node we are dealing with relative to the
				// overall start position of the layer

				(layerSize[iLayer + 1] + biasAdjustment).do {
					|iNextLayerNode|
					// iNextLayerNode is the index of the node in the next layer, relative to its
					// overall layer starting index
					node1 = layerIndex[iLayer] + iLayerNode;
					node2 = layerIndex[iLayer + 1] + iNextLayerNode;
					nodes[node2] = nodes[node2] + (weights[iWeight] * nodes[node1]);
					iWeight = iWeight + 1;
				};
			};
			NeuralNet.activationFunction(nodes,
				layerIndex[iLayer + 1],
				layerSize[iLayer + 1] + biasAdjustment);
		};
	}
}