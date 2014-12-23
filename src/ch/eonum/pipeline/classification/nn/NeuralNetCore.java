package ch.eonum.pipeline.classification.nn;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.StringTokenizer;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Core functionality (all the math) of the feed forward 3 layer
 * Neural Network.
 * 
 * @author tim
 * 
 */
public class NeuralNetCore<E extends Instance> implements Runnable {
	/** random number generator with seed. */
	protected Random rand;
	/** Thread Name/Number. */
	protected String threadName;
	/** base directory, where we store the net. */
	protected String baseDir;
	/** number of epochs seen by the net */
	protected int epoch;
	/** maximum number of epochs. */
	protected long maxEpochs;
	/** learning rate */
	protected double learningRate;
	/** momentum for weight delta. */
	protected double momentum;
	/**
	 * matrix for storing the target per training element
	 * per output component
	 */
	private double tar[][];
	/**
	 * matrix for storing the input per training element per
	 * input component
	 */
	private double inp[][];
	/**
	 * matrix for storing the target per test element per
	 * output component
	 */
	private double tar_t[][];
	/**
	 * matrix for storing the input per test element per
	 * input component
	 */
	private double inp_t[][];
	/** mean square error per epoch */
	protected double epoch_err;
	/** current validation set error minimum during training. */
	private double currentMin;
	/** number of epochs since we reached the minimum of the validation error. */
	protected int stepsSinceMin;
	/** validation error for each epoch. */
	protected Map<Integer, Double> validationEpochs;
	/** parent neural net. all parameters can be obtained from there. */
	private NeuralNet<E> parent;
	/**
	 * true if weights are to be initialized randomly. false if weights are set.
	 */
	private boolean randomWeightInitialization;
	/** number of units. */
	protected int nInput, nHidden;
	protected int nOutput;
	/** units. */
	private double[] inputNeurons;
	protected double[] hiddenNeurons;
	protected double[] outputNeurons;
	/** weight matrix between input and hidden unit. */
	protected double[][] wInputHidden;
	/** weight matrix between hidden and output unit. */
	protected double[][] wHiddenOutput;

	/** weight gradients. */
	protected double[][] deltaInputHidden;
	protected double[][] deltaHiddenOutput;

	/** error gradients. */
	protected double[] hiddenErrorGradients;
	protected double[] outputErrorGradients;

	/** accuracy stats per epoch. */
	protected double trainingSetAccuracy;
	protected double validationSetAccuracy;
	protected double trainingSetMSE;

	/** batch learning flag. */
	protected boolean useBatch;
	/** batch size if using mini-batch learning. */
	protected int batchSize;
	/**
	 * maximum number of epochs to continue training after having reached a
	 * maximum in validation set accuraccy.
	 */
	protected double maxEpochsAfterMin;
	/** index of current training sample. */
	protected int example;
	/** lambda for weight decay. */
	protected double lambda;
	/** do dropout. */
	protected boolean dropout;
	/** do dropout on hidden unit i. */
	protected boolean[] dropouts;
	/**
	 * use the softmax activation function for the output layer. used for
	 * classification tasks.
	 */
	protected boolean softmax;

	public NeuralNetCore(String threadName, String baseDir, NeuralNet<E> parent,
			int seed, boolean dropout, boolean classify) {
		this.threadName = threadName;
		this.baseDir = baseDir;
		this.parent = parent;
		this.rand = new Random(seed);
		this.randomWeightInitialization = true;
		this.rand = new Random(seed);
		this.dropout = dropout;
		this.softmax = classify;
	}

	protected void setParameters() {
		this.nInput = parent.getFeatures().size();
		this.learningRate = parent.getDoubleParameter("learningRate");
		this.nHidden = (int) parent.getDoubleParameter("hidden");
		this.batchSize = (int) parent.getDoubleParameter("batchSize");
		this.maxEpochs = (long) parent.getIntParameter("maxEpochs");
		this.momentum = parent.getDoubleParameter("momentum");
		this.maxEpochsAfterMin = parent.getDoubleParameter("maxEpochsAfterMax");
		this.nOutput = parent.getNumberOfOutputs();
		this.lambda =parent.getDoubleParameter("lambda");
	}

	public void save(String fileName) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(fileName));
			p.println(wInputHidden.length + " " + wInputHidden[0].length);
			for(int i = 0; i < this.wInputHidden.length; i++)
				for(int j = 0; j < this.wInputHidden[i].length; j++)
					p.print(wInputHidden[i][j] + " ");
			p.println();
			p.println(wHiddenOutput.length + " " + wHiddenOutput[0].length);
			for(int i = 0; i < this.wHiddenOutput.length; i++)
				for(int j = 0; j < this.wHiddenOutput[i].length; j++)
					p.print(wHiddenOutput[i][j] + " ");
			p.println();
			p.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void load(String filename) throws IOException {
		this.setParameters();
		this.init();
		
		Scanner scanner = new Scanner(new File(filename));
		String line = scanner.nextLine();
		String[] lengths = line.split(" ");
		this.nInput = Integer.valueOf(lengths[0]) - 1;
		this.nHidden = Integer.valueOf(lengths[1]);
		wInputHidden = new double[nInput + 1][nHidden];
		line = scanner.nextLine();
		StringTokenizer st = new StringTokenizer(line);
		for (int i = 0; i < this.wInputHidden.length; i++)
			for (int j = 0; j < this.wInputHidden[i].length; j++)
				wInputHidden[i][j] = Double.parseDouble(st.nextToken());

		line = scanner.nextLine();
		lengths = line.split(" ");
		this.nOutput = Integer.valueOf(lengths[1]);
		wHiddenOutput = new double[nHidden + 1][nOutput];
		line = scanner.nextLine();
		st = new StringTokenizer(line);
		for (int i = 0; i < this.wHiddenOutput.length; i++)
			for (int j = 0; j < this.wHiddenOutput[i].length; j++)
				wHiddenOutput[i][j] = Double.parseDouble(st.nextToken());

		scanner.close();
	}

	/** initialize a random net. */
	protected void init() {
		inputNeurons = new double[nInput + 1];
		inputNeurons[nInput] = -1;
		hiddenNeurons = new double[nHidden + 1];
		hiddenNeurons[nHidden] = -1;
		outputNeurons = new double[nOutput];

		deltaInputHidden = new double[nInput + 1][nHidden];
		deltaHiddenOutput = new double[nHidden + 1][nOutput];

		hiddenErrorGradients = new double[nHidden + 1];
		outputErrorGradients = new double[nOutput + 1];

		if(this.randomWeightInitialization)
			initializeWeights();

		useBatch = batchSize > 1;
	}

	private void initializeWeights() {
		wInputHidden = new double[nInput + 1][nHidden];
		wHiddenOutput = new double[nHidden + 1][nOutput];
		
		for (int i = 0; i <= nInput; i++) {
			for (int j = 0; j < nHidden; j++) {
				wInputHidden[i][j] = rand.nextDouble() - 0.5;
				deltaInputHidden[i][j] = 0;
			}
		}

		for (int i = 0; i <= nHidden; i++) {
			for (int j = 0; j < nOutput; j++) {
				wHiddenOutput[i][j] = rand.nextDouble() - 0.5;
				deltaHiddenOutput[i][j] = 0;
			}
		}
	}

	protected void feedForward(double inputs[]) {
		for (int i = 0; i < nInput; i++)
			inputNeurons[i] = inputs[i];

		/** hidden layer. */
		for (int j = 0; j < nHidden; j++) {
			hiddenNeurons[j] = 0;
			if(!dropout || dropouts[j]){
				for (int i = 0; i <= nInput; i++)
					hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];
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

	private void backpropagate(double desiredValues[]) {
		/** output to hidden layer. */
		for (int k = 0; k < nOutput; k++) {
			outputErrorGradients[k] = getOutputErrorGradient(desiredValues[k],
					outputNeurons[k]) ;
			for (int j = 0; j <= nHidden; j++) {
				if(!dropout || nHidden == j || dropouts[j])
					deltaHiddenOutput[j][k] += learningRate * hiddenNeurons[j]
								* (outputErrorGradients[k] + lambda * wHiddenOutput[j][k]);
			}
		}

		/** hidden to input layer. */
		for (int j = 0; j < nHidden; j++) {
			if(!dropout || dropouts[j]){
				hiddenErrorGradients[j] = getHiddenErrorGradient(j);
				for (int i = 0; i <= nInput; i++) {
					deltaInputHidden[i][j] += learningRate * inputNeurons[i]
								* (hiddenErrorGradients[j] + lambda * wInputHidden[i][j]);
				}
			}
		}

		/** if using online learning update the weights immediately. */
		if (!useBatch)
			updateWeights();
	}

	protected void updateWeights() {
		for (int i = 0; i <= nInput; i++) {
			for (int j = 0; j < nHidden; j++) {
				wInputHidden[i][j] += deltaInputHidden[i][j];
				deltaInputHidden[i][j] *= momentum;
			}
		}

		for (int j = 0; j <= nHidden; j++) {
			for (int k = 0; k < nOutput; k++) {
				wHiddenOutput[j][k] += deltaHiddenOutput[j][k];
				deltaHiddenOutput[j][k] *= momentum;
			}
		}
	}

	/** sigmoid activation function. */
	protected double activationFunction(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	/**
	 * get error gradient for output layer. gradient of sigmoid activation
	 * function or softmax layer.
	 */
	protected double getOutputErrorGradient(double desiredValue,
			double outputValue) {
		if(softmax)
			return (desiredValue - outputValue);
		else
			return outputValue * (1 - outputValue) * (desiredValue - outputValue);
	}

	/** get error gradient for hidden layer. */
	protected double getHiddenErrorGradient(int j) {
		double weightedSum = 0;
		for (int k = 0; k < nOutput; k++)
			weightedSum += wHiddenOutput[j][k] * outputErrorGradients[k];
		return hiddenNeurons[j] * (1 - hiddenNeurons[j]) * weightedSum;
	}
	
	private void runTrainingEpoch() {
		double mse = 0;
		for (example = 0; example < (int) inp.length; example++) {	
			if(dropout){
				dropouts = new boolean[nHidden];
				for(int i = 0; i < dropouts.length; i++)
					dropouts[i] = this.rand.nextBoolean();
			}
			/** feed inputs through network and backpropagate errors. */
			feedForward(inp[example]);
			backpropagate(tar[example]);

			/** check all outputs from neural network against desired values. */
			for (int k = 0; k < nOutput; k++)
				mse += Math.pow((outputNeurons[k] - tar[example][k]), 2);
			
			if (useBatch
					&& ((example + batchSize + 1) % batchSize == 0 || example == inp.length - 1))
				updateWeights();
		}		

		trainingSetMSE = mse / (nOutput * inp.length);
	}
	
	/**
	 * get mean square error of a set in case of regression or cross entropy in
	 * case of classification.
	 * 
	 * @param inputs
	 * @param targets
	 * @return
	 */
	protected double getSetAccuracy(double[][] inputs, double[][] targets) {
		double mse = 0;
		if(dropout){
			dropouts = new boolean[nHidden];
			for(int i = 0; i < dropouts.length; i++)
				dropouts[i] = true;
			this.multiplyHiddenWeigths(0.5);
		}
		for (int tp = 0; tp < (int) inputs.length; tp++) {
			feedForward(inputs[tp]);

			for (int k = 0; k < nOutput; k++)
				if(softmax)
					mse -= targets[tp][k] * Math.log(outputNeurons[k]);
				else
					mse += Math.pow((outputNeurons[k] - targets[tp][k]), 2);
		}
		if(dropout)
			multiplyHiddenWeigths(2.0);
		return mse / (nOutput * inputs.length);
	}

	protected void multiplyHiddenWeigths(double factor) {
		for (int j = 0; j < nHidden; j++)
			for (int k = 0; k < wHiddenOutput[j].length; k++)
				wHiddenOutput[j][k] *= factor;
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
		/** release memory. */
		this.inp = null;
		this.tar = null;
		this.inp_t = null;
		this.tar_t = null;
	}

	public double getValidationError() {
		return this.currentMin;
	}

	public void test(DataSet<E> testData) {
		if(dropout){
			dropouts = new boolean[nHidden];
			for(int i = 0; i < dropouts.length; i++)
				dropouts[i] = true;
			this.multiplyHiddenWeigths(0.5);
		}
		
		for (int tp = 0; tp < inp_t.length; tp++) {
			/** feed inputs through network. */
			feedForward(inp_t[tp]);
			/** store output. */
			for(int k = 0; k < nOutput; k++)
				testData.get(tp).putResult("resultNode" + k, testData.get(tp).getResult("resultNode" + k) + outputNeurons[k]);
		}
		if(dropout)
			this.multiplyHiddenWeigths(2.0);
	}

	public void setTrainingData(double[][] input, double[][] target) {
		this.inp = input;
		this.tar = target;
	}

	public void setTestData(double[][] input, double[][] target) {
		this.inp_t = input;
		this.tar_t = target;
	}

	@Override
	public void run() {
		try {
			this.train();
		} catch (Exception e){
			e.printStackTrace();
			System.exit(-1);
		}
	}

}
