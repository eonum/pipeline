package ch.eonum.pipeline.classification.nn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Log;

/**
 * Fast Forward neural network with one hidden layer for regression and
 * classification tasks (softmax output layer).
 * 
 * Includes k-fold cross validation.
 * 
 * @author tim
 * 
 */
public class NeuralNet<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("hidden",
						"number of hidden layer neurons. (default 10)");
		PARAMETERS.put("maxEpochsAfterMax",
						"Number of epochs to continue training after the last maximum on validation set accuraccy is reached."
								+ "The training of a neural net is then stopped when we cannot improve accuraccy on the validation set. (default: 10)");
		PARAMETERS.put("maxEpochs", "global maximum of epochs (default: 5000)");
		PARAMETERS.put("learningRate", "learning rate alpha (default: 0.007)");
		PARAMETERS.put("numNets",
				"number of differently initialized nets that are trained for each net(default: 1.0)");
		PARAMETERS.put("numNetsTotal",
						"number of nets that are trained to build a network of nets. "
								+ "Equals k in k-fold cross validation. (default: 1.0, no cross validation)");
		PARAMETERS.put("normTarget",
						"normalize all targets by dividing by normTarget."
								+ " If set to -1.0, maxOutcome is used for norming (default: -1.0)");
		PARAMETERS.put("momentum", "learning momentum. Default: 0.0 (no momentum)");
		PARAMETERS.put("batchSize", "batch size for mini batch or batch learning: 1.0 (online backpropagation)");
		PARAMETERS.put("lambda", "lambda for weight decay. Default: 0.0 (no weight decay)");
		PARAMETERS.put("maxOutcome", "maximum outcome for regression tasks. used for normalization.");
	}

	/** the core nets. */
	protected List<NeuralNetCore<E>> nets;
	/** maximum outcome. norm factor. */
	protected double maxOutcome;
	/** do dropout. */
	protected boolean dropout;
	
	public NeuralNet(Features features) {
		super();
		this.setFeatures(features);
		this.setSupportedParameters(NeuralNet.PARAMETERS);
		this.putParameter("hidden", 10.0);
		this.putParameter("maxEpochsAfterMax", 10.0);
		this.putParameter("maxEpochs", 1000);
		this.putParameter("learningRate", 0.007);
		this.putParameter("numNets", 1.0);
		this.putParameter("numNetsTotal", 1.0);
		this.putParameter("normTarget", -1.0);
		this.putParameter("momentum", 0.0);
		this.putParameter("lambda", 0.0);
		this.putParameter("batchSize", 1.0);
		this.classify = false;
	}

	@Override
	public void train() {
		Log.puts("Neural Net: Start Training");
		if(classify)
			prepareClasses();
		
		this.maxOutcome = Double.NEGATIVE_INFINITY;
		for(Instance each : trainingDataSet)
			maxOutcome = Math.max(maxOutcome, each.outcome);
		if(this.getDoubleParameter("normTarget") > 0.0)
			maxOutcome = getDoubleParameter("normTarget");
		this.putParameter("maxOutcome", maxOutcome);
		
		int numNetsTotal = (int)this.getDoubleParameter("numNetsTotal");
		int numNets = (int)this.getDoubleParameter("numNets");
		
		double[][][] trainInputs = null;
		double[][][] trainTargets = null;
		
		double[][] input = null;
		double[][] target = null;
		double[][] inputTest = null;
		double[][] targetTest = null;
		
		/** create training and validation data. */
		if(numNetsTotal > 1){ /** k-fold cross validation. */
			trainInputs = new double[numNetsTotal][][];
			trainTargets = new double[numNetsTotal][][];
			int i = 0;
			for (DataSet<E> each : this.trainingDataSet.splitIntoNSubsets(numNetsTotal)) {
				trainInputs[i] = each.asDoubleArrayMatrix(features);
				trainTargets[i] = getTargets(each);
				i++;
			}
		} else { /** no cross validation. only one net. */
			input = this.trainingDataSet.asDoubleArrayMatrix(features);
			target = getTargets(this.trainingDataSet);
			inputTest = this.testDataSet.asDoubleArrayMatrix(features);
			targetTest = getTargets(this.testDataSet);
		}
		
		ExecutorService service = Executors.newFixedThreadPool(Runtime
				.getRuntime().availableProcessors());
		ArrayList<List<NeuralNetCore<E>>> allNets = new ArrayList<List<NeuralNetCore<E>>>();
		
		
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){
			/** select training and validation data if we do k-fold cross validation. */			
			if (numNetsTotal > 1) {
				inputTest = trainInputs[netNumber];
				targetTest = trainTargets[netNumber];
				
				int length = 0;
				for (int i = 0; i < trainInputs.length; i++)
					if(i != netNumber)
						length += trainInputs[i].length;
				input = new double[length][];
				target = new double[length][];
				int k = 0;
				for (int i = 0; i < trainInputs.length; i++)
					if (i != netNumber)
						for(int j = 0; j < trainInputs[i].length; j++){
							input[k] = trainInputs[i][j];
							target[k++] = trainTargets[i][j];
						}
			}
			
			/** create all differently initialized nets . */	
			ArrayList<NeuralNetCore<E>> randomNets = new ArrayList<NeuralNetCore<E>>();	
			for(int i = 0; i < numNets; i++){
				String name = netNumber + "-" + i;
				FileUtil.mkdir(this.getBaseDir() + name + "/");
				NeuralNetCore<E> net = createNet(name, i);
				net.setTrainingData(input, target);
				net.setTestData(inputTest, targetTest);
				randomNets.add(net);
				service.submit(net);
			}
			allNets.add(randomNets);
		}
		
		/** execute all nets. Training. */
		service.shutdown();
		try {
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		this.nets = new ArrayList<NeuralNetCore<E>>();
		
		/** Select the best performing nets. */
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){
			double max = Double.POSITIVE_INFINITY;
			int maxNet = -1;
			List<NeuralNetCore<E>> randomNets = allNets.get(netNumber);
			for (int numNet = 0; numNet < numNets; numNet++){
				if(randomNets.get(numNet).getValidationError() < max){
					max = randomNets.get(numNet).getValidationError();
					maxNet = numNet;
				}
			}
			
			randomNets.get(maxNet).save(this.getBaseDir() + "maxNNglobal" + netNumber);
			nets.add(randomNets.get(maxNet));
		}
	}

	protected NeuralNetCore<E> createNet(String name, int seed) {
		return new NeuralNetCore<E>(name, this.getBaseDir() + name
				+ "/", this, seed * 11, dropout, classify);
	}
	
	private double[][] getTargets(DataSet<E> dataset) {
		double[][] targets = new double[dataset.size()][this.getNumberOfOutputs()];
		if(classify){
			for(int i = 0; i < targets.length; i++)
				if(dataset.get(i).groundTruth != null) /** testing mode with no groundtruth. */
					targets[i][this.classes.getIndexFromFeature(dataset.get(i).groundTruth)] = 1.0;
		} else {
			for(int i = 0; i < targets.length; i++)	
				targets[i][0] = dataset.get(i).outcome / this.maxOutcome;
		}
		return targets;
	}

	@Override
	public DataSet<E> test(){
		double[][] inputsTest = testDataSet.asDoubleArrayMatrix(features);
		double[][] targetsTest = getTargets(testDataSet);
		
		for(Instance each : this.testDataSet)
			for(int k = 0; k < this.getNumberOfOutputs(); k++)
				each.putResult("resultNode" + k, 0.0);
		
		int numNetsTotal = (int)this.getDoubleParameter("numNetsTotal");
		for(int netNumber = 0; netNumber < numNetsTotal; netNumber++){
			NeuralNetCore<E> net = nets.get(netNumber);
			net.setTestData(inputsTest, targetsTest);
			net.test(this.testDataSet);
		}
		
		if(classify){
			for(Instance each : this.testDataSet){
				int maxIndex = -1;
				double maxValue = Double.NEGATIVE_INFINITY;
				for(int k = 0; k < this.getNumberOfOutputs(); k++){
					double value = each.getResult("resultNode" + k) / numNetsTotal;
					each.putResult("classProb" + classes.getFeatureByIndex(k), value);
					if(value > maxValue){
						maxValue = value;
						maxIndex = k;
					}
					each.removeResult("resultNode" + k);
				}
				each.label = classes.getFeatureByIndex(maxIndex);
				each.putResult("result", maxValue);
			}
		} else 
			for(Instance each : this.testDataSet)
				each.putResult("result", (each.getResult("resultNode0") / numNetsTotal) * this.maxOutcome);
		
		return this.testDataSet;
	}
	
	@Override
	public void save(String fileName) throws IOException {
		super.save(fileName);
		int i = 0;
		for (NeuralNetCore<E> each : nets)
			each.save(fileName + i++);
	}
	
	@Override
	public void loadSerializedState(File file) throws IOException {
		super.loadSerializedState(file);
		this.maxOutcome = this.getDoubleParameter("maxOutcome");
		this.load(file.getAbsolutePath());
	}
	
	@Override
	public void load(String fileName) throws IOException{
		super.load(fileName);
		nets = new ArrayList<NeuralNetCore<E>>();
		for(int i = 0; i < (int)this.getDoubleParameter("numNetsTotal"); i++){
			NeuralNetCore<E> net = createNet("net" + i, 11);
			net.load(fileName + i);
			nets.add(net);
		}
	}
	
	public void enableClassification(){
		this.classify = true;
	}

	public int getNumberOfOutputs() {
		return classify ? classes.size() : 1;
	}
	
	public void doDropout() {
		this.dropout = true;
	}

}
