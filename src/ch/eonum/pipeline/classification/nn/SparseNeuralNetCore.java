package ch.eonum.pipeline.classification.nn;

import java.io.IOException;
import java.util.LinkedHashMap;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Entry;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Core functionality (all the math) of the feed forward 3 layer
 * @see SparseNeuralNet.
 * 
 * @author tim
 *
 * @param <E>
 */
public class SparseNeuralNetCore<E extends Instance> extends NeuralNetCore<E> {

	private Entry[][] inp;
	private Entry[] tar;
	private Entry[][] inp_t;
	private Entry[] tar_t;
	
	private Entry[] inputNeurons;

	public SparseNeuralNetCore(String threadName, String baseDir,
			NeuralNet<E> parent, int seed, boolean dropout, boolean classify) {
		super(threadName, baseDir, parent, seed, dropout, classify);
	}

	private void feedForward(Entry inputs[]) {
		inputNeurons = new Entry[inputs.length + 1];
		inputNeurons[inputs.length] = new Entry(nInput, -1);
		
		for (int i = 0; i < inputs.length; i++)
			inputNeurons[i] = inputs[i];

		/** hidden layer. */
		for (int j = 0; j < nHidden; j++) {
			hiddenNeurons[j] = 0;
			if(!dropout || dropouts[j]){
				for (int i = 0; i < inputNeurons.length; i++)
					hiddenNeurons[j] += inputNeurons[i].value * wInputHidden[inputNeurons[i].index][j];
				hiddenNeurons[j] = activationFunction(hiddenNeurons[j]);
			}
		}

		/** output layer. */
		for (int k = 0; k < nOutput; k++) {
			outputNeurons[k] = 0;
			for (int j = 0; j <= nHidden; j++)
				if(!dropout ||  nHidden == j || dropouts[j])
					outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];
			if (softmax)
				outputNeurons[k] = Math.exp(outputNeurons[k]);
			else
				outputNeurons[k] = activationFunction(outputNeurons[k]);
		}
		/**
		 * normalize all output units to sum up to 1 when using the softmax
		 * activation function.
		 */
		if(softmax){
			double total = 0.0;
			for (int k = 0; k < outputNeurons.length; k++)
				total += outputNeurons[k];
			for (int k = 0; k < outputNeurons.length; k++)
				outputNeurons[k] /= total;
		}
	}

	private void backpropagate(Entry desiredValue) {
		/** output to hidden layer. */
		for (int k = 0; k < nOutput; k++) {
			outputErrorGradients[k] = getOutputErrorGradient(
					desiredValue.index == k ? desiredValue.value : 0.0,
					outputNeurons[k]);
			for (int j = 0; j <= nHidden; j++) {
				if (!dropout || nHidden == j || dropouts[j])
					deltaHiddenOutput[j][k] += learningRate
							* hiddenNeurons[j]
							* (outputErrorGradients[k] + lambda
									* wHiddenOutput[j][k]);
			}
		}

		/** hidden to input layer. */
		for (int j = 0; j < nHidden; j++) {
			if(!dropout || dropouts[j]){
				hiddenErrorGradients[j] = getHiddenErrorGradient(j);
				for (int i = 0; i < inputNeurons.length; i++) {
					deltaInputHidden[inputNeurons[i].index][j] += learningRate * inputNeurons[i].value
								* (hiddenErrorGradients[j] + lambda * wInputHidden[inputNeurons[i].index][j]);
				}
			}
		}

		/** if using online learning update the weights immediately. */
		if (!useBatch)
			updateWeights();
	}
	
	private void runTrainingEpoch() {
		double mse = 0;
		for (example = 0; example < (int) inp.length; example++) {	
			if(dropout){
				dropouts = new boolean[nHidden];
				for(int i = 0; i < dropouts.length; i++)
					dropouts[i] = this.rand.nextBoolean();
			}

			feedForward(inp[example]);
			backpropagate(tar[example]);


			for (int k = 0; k < nOutput; k++)
				mse += Math.pow((outputNeurons[k] - (tar[example].index == k ? tar[example].value : 0.0)), 2);
			
			if (useBatch
					&& ((example + batchSize + 1) % batchSize == 0 || example == inp.length - 1))
				updateWeights();
		}		

		trainingSetMSE = mse / (nOutput * inp.length);
	}
	
	/**
	 * get MSE of a set in case of regression or cross entropy in case of classification.
	 * @param inputs
	 * @param targets
	 * @return
	 */
	private double getSetAccuracy(Entry[][] inputs, Entry[] targets) {
		double mse = 0;
		if(dropout){
			dropouts = new boolean[nHidden];
			for(int i = 0; i < dropouts.length; i++)
				dropouts[i] = true;
			this.multiplyHiddenWeigths(0.5);
		}
		for (int tp = 0; tp < (int) inputs.length; tp++) {
			feedForward(inputs[tp]);
			if (softmax)
				mse -= targets[tp].value
						* Math.log(outputNeurons[targets[tp].index]);
			else
				for (int k = 0; k < nOutput; k++)
					mse += Math.pow((outputNeurons[k] - (targets[tp].index == k ? targets[tp].value : 0.0)), 2);
		}
		if(dropout)
			multiplyHiddenWeigths(2.0);
		return mse / (nOutput * inputs.length);
	}
	
	public void train() throws IOException {
		this.setParameters();
		this.init();
		validationEpochs = new LinkedHashMap<Integer, Double>();
		
		double minMSE = Double.POSITIVE_INFINITY;
		
		stepsSinceMin = 0;
		epoch = 0;			
		while (epoch < maxEpochs && stepsSinceMin < maxEpochsAfterMin) {
			runTrainingEpoch();
			epoch_err = getSetAccuracy(inp_t, tar_t);
			validationEpochs.put(epoch, Math.sqrt(epoch_err));
			Gnuplot.plotOneDimensionalCurve(validationEpochs, "validationEpochs", this.baseDir + "validationEpochs");
			
			Log.puts("Thread " + threadName + " Epoch :" + epoch);
			Log.puts(" TrainSet " + (softmax ? "cross entropy: " : "MSE: ") + Math.sqrt(trainingSetMSE));
			Log.puts(" TestSet " + (softmax ? "cross entropy: " : "MSE: ") + Math.sqrt(epoch_err));
			
			if(epoch_err < minMSE){
				minMSE = epoch_err;
				stepsSinceMin = 0;
				this.save(baseDir + "maxnet.txt");
			} else
				stepsSinceMin++;
			
			epoch++;
		}

		
		this.load(baseDir + "maxnet.txt");

		this.inp = null;
		this.tar = null;
		this.inp_t = null;
		this.tar_t = null;
	}
	
	public void test(DataSet<E> testData) {
		if(dropout){
			dropouts = new boolean[nHidden];
			for(int i = 0; i < dropouts.length; i++)
				dropouts[i] = true;
			this.multiplyHiddenWeigths(0.5);
		}
		
		for (int tp = 0; tp < inp_t.length; tp++) {
			feedForward(inp_t[tp]);
			for(int k = 0; k < nOutput; k++)
				testData.get(tp).putResult("resultNode" + k, testData.get(tp).getResult("resultNode" + k) + outputNeurons[k]);
		}
		if(dropout)
			this.multiplyHiddenWeigths(2.0);
	}

	public void setTrainingData(Entry[][] input, Entry[] target) {
		this.inp = input;
		this.tar = target;
	}

	public void setTestData(Entry[][] input, Entry[] target) {
		this.inp_t = input;
		this.tar_t = target;
	}
}
